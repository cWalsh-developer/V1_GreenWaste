from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .capture_manifest import load_metadata
from .size_estimation import estimate_size_from_roi, load_capture_images, parse_intrinsics

try:
    import torch
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "ultralytics and torch are required. Install with: pip install ultralytics"
    ) from exc


def xyxy_to_roi(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1_i = max(int(round(x1)), 0)
    y1_i = max(int(round(y1)), 0)
    x2_i = min(int(round(x2)), image_width)
    y2_i = min(int(round(y2)), image_height)
    return (x1_i, y1_i, max(x2_i - x1_i, 1), max(y2_i - y1_i, 1))


def yolo_label_to_roi(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    line = label_path.read_text().splitlines()[0]
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")

    _, x_center, y_center, width, height = parts[:5]
    x_center_px = float(x_center) * image_width
    y_center_px = float(y_center) * image_height
    width_px = float(width) * image_width
    height_px = float(height) * image_height
    x1 = x_center_px - width_px / 2.0
    y1 = y_center_px - height_px / 2.0
    x2 = x_center_px + width_px / 2.0
    y2 = y_center_px + height_px / 2.0
    return xyxy_to_roi((x1, y1, x2, y2), image_width, image_height)


def size_category(width_cm: float, height_cm: float, depth_cm: float) -> str:
    largest_dimension = max(width_cm, height_cm, depth_cm)
    if largest_dimension < 50:
        return "small"
    if largest_dimension < 120:
        return "medium"
    return "large"


def detect_items(
    model_path: Path,
    image_path: Path,
    confidence: float,
    image_size: int,
) -> list[dict[str, Any]]:
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    result = model.predict(
        source=str(image_path),
        conf=confidence,
        iou=0.7,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]

    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "item_class": result.names[class_id],
                "class_id": class_id,
                "confidence": float(box.conf.item()),
                "xyxy": (x1, y1, x2, y2),
            }
        )

    return sorted(detections, key=lambda row: row["confidence"], reverse=True)


def estimate_capture_items(
    capture_dir: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
    max_detections: int,
    label_file: Path | None = None,
    label_item_class: str = "manual_roi",
) -> list[dict[str, Any]]:
    metadata = load_metadata(capture_dir)
    rgb, depth = load_capture_images(capture_dir)
    image_height, image_width = rgb.shape[:2]
    intrinsics = parse_intrinsics(metadata)
    depth_scale = float(metadata.get("depth_scale", 0.001))

    if label_file is not None:
        roi = yolo_label_to_roi(
            label_path=label_file,
            image_width=image_width,
            image_height=image_height,
        )
        detections = [
            {
                "item_class": label_item_class,
                "class_id": None,
                "confidence": None,
                "xyxy": (
                    float(roi[0]),
                    float(roi[1]),
                    float(roi[0] + roi[2]),
                    float(roi[1] + roi[3]),
                ),
            }
        ]
    else:
        detections = detect_items(
            model_path=model_path,
            image_path=capture_dir / "rgb.png",
            confidence=confidence,
            image_size=image_size,
        )

    outputs: list[dict[str, Any]] = []
    for detection in detections[:max_detections]:
        if label_file is not None:
            roi = yolo_label_to_roi(
                label_path=label_file,
                image_width=image_width,
                image_height=image_height,
            )
        else:
            roi = xyxy_to_roi(
                xyxy=detection["xyxy"],
                image_width=image_width,
                image_height=image_height,
            )
        estimate = estimate_size_from_roi(
            depth_image=depth,
            roi=roi,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
        )

        estimate_payload = asdict(estimate)
        outputs.append(
            {
                "capture_id": metadata.get("capture_id", capture_dir.name),
                "item_class": detection["item_class"],
                "confidence": detection["confidence"],
                "bbox_xyxy": [round(value, 2) for value in detection["xyxy"]],
                "roi": list(roi),
                "width_cm": estimate.width_cm,
                "height_cm": estimate.height_cm,
                "depth_cm": estimate.depth_cm,
                "distance_cm": estimate.distance_cm,
                "size_category": size_category(
                    width_cm=estimate.width_cm,
                    height_cm=estimate.height_cm,
                    depth_cm=estimate.depth_cm,
                ),
                "roi_refined": list(estimate.roi_refined),
                "size_estimate": estimate_payload,
            }
        )

    return outputs


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, capture_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{capture_id}_yolo_size.json"
    csv_path = output_dir / f"{capture_id}_yolo_size.csv"

    json_path.write_text(json.dumps(rows, indent=2))
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"size_estimate"}
        }
        for row in rows
    ]
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on a RealSense capture and estimate size from depth"
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/train-2/weights/best.pt"),
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--max-detections", type=int, default=1)
    parser.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help="Optional YOLO-format label file to use instead of model detection.",
    )
    parser.add_argument(
        "--label-item-class",
        type=str,
        default="manual_roi",
        help="Item class to report when --label-file is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/yolo_size_estimates"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = estimate_capture_items(
        capture_dir=args.capture_dir,
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        max_detections=args.max_detections,
        label_file=args.label_file,
        label_item_class=args.label_item_class,
    )
    capture_id = args.capture_dir.name
    write_outputs(rows=rows, output_dir=args.output_dir, capture_id=capture_id)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

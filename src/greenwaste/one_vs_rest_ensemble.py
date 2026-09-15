from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .capture_manifest import load_metadata
from .lca import (
    DEFAULT_LCA_FACTOR_PATH,
    DEFAULT_MATERIAL_PROFILE_PATH,
    build_composition_factor_rows,
    load_lca_factor_table,
    load_material_profile_table,
    select_lca_factors,
    select_material_profiles,
)
from .lca_estimation import CONDITION_STATUSES, build_lca_payload
from .reference_matching import build_match_payload, match_reference_items
from .size_estimation import estimate_size_from_roi, load_capture_images, parse_intrinsics
from .yolo_depth_size import size_category, xyxy_to_roi

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "opencv-python is required for ensemble visualisation. Install with: pip install opencv-python"
    ) from exc

try:
    import torch
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "ultralytics and torch are required. Install with: pip install ultralytics torch"
    ) from exc


@dataclass(frozen=True)
class EnsembleDetection:
    item_class: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    model_path: str


def parse_key_value_path(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CLASS=PATH, got: {value}")
        key, path = value.split("=", 1)
        parsed[key.strip()] = resolve_model_path(Path(path.strip()))
    return parsed


def resolve_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path.resolve()

    nested_runs_path = Path("runs") / "detect" / model_path
    if nested_runs_path.exists():
        return nested_runs_path.resolve()

    raise FileNotFoundError(
        "Specialist model weights were not found: "
        f"{model_path}. Also checked: {nested_runs_path}"
    )


def parse_key_value_float(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CLASS=VALUE, got: {value}")
        key, number = value.split("=", 1)
        parsed[key.strip()] = float(number)
    return parsed


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    intersection = inter_w * inter_h
    first_area = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    second_area = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def non_max_suppression(
    detections: list[EnsembleDetection],
    iou_threshold: float,
) -> list[EnsembleDetection]:
    kept: list[EnsembleDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(box_iou(detection.bbox_xyxy, other.bbox_xyxy) < iou_threshold for other in kept):
            kept.append(detection)
    return kept


def run_specialist_models(
    image: np.ndarray,
    model_paths: dict[str, Path],
    confidence: float,
    class_thresholds: dict[str, float],
    image_size: int,
    max_per_model: int,
) -> list[EnsembleDetection]:
    device = 0 if torch.cuda.is_available() else "cpu"
    detections: list[EnsembleDetection] = []
    for item_class, model_path in model_paths.items():
        model = YOLO(str(model_path))
        threshold = class_thresholds.get(item_class, confidence)
        result = model.predict(
            source=image,
            conf=threshold,
            iou=0.7,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]

        class_detections: list[EnsembleDetection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_detections.append(
                EnsembleDetection(
                    item_class=item_class,
                    confidence=float(box.conf.item()),
                    bbox_xyxy=(x1, y1, x2, y2),
                    model_path=str(model_path),
                )
            )
        detections.extend(
            sorted(class_detections, key=lambda item: item.confidence, reverse=True)[
                :max_per_model
            ]
        )
    return detections


def build_route_payload(
    capture_dir: Path,
    rgb_image_name: str,
    detection: EnsembleDetection,
    reference_path: Path,
    factor_path: Path,
    material_profile_path: Path,
    use_material_profiles: bool,
    condition_status: str,
    top_n: int,
) -> dict[str, Any]:
    metadata = load_metadata(capture_dir)
    rgb, depth = load_capture_images(capture_dir, rgb_image_name=rgb_image_name)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    image_height, image_width = rgb.shape[:2]
    roi = xyxy_to_roi(detection.bbox_xyxy, image_width, image_height)
    estimate = estimate_size_from_roi(
        depth_image=depth,
        roi=roi,
        intrinsics=parse_intrinsics(metadata),
        depth_scale=float(metadata.get("depth_scale", 0.001)),
    )
    size_label = size_category(
        width_cm=estimate.width_cm,
        height_cm=estimate.height_cm,
        depth_cm=estimate.depth_cm,
    )
    size_row = pd.Series(
        {
            "capture_id": metadata.get("capture_id", capture_dir.name),
            "item_class": detection.item_class,
            "confidence": detection.confidence,
            "width_cm": estimate.width_cm,
            "height_cm": estimate.height_cm,
            "depth_cm": estimate.depth_cm,
            "distance_cm": estimate.distance_cm,
            "size_category": size_label,
        }
    )
    reference_df = pd.read_csv(reference_path)
    matches = match_reference_items(
        reference_df=reference_df,
        item_class=detection.item_class,
        width_cm=estimate.width_cm,
        height_cm=estimate.height_cm,
        depth_cm=estimate.depth_cm,
        top_n=top_n,
    )
    match_payload = build_match_payload(size_row, matches)
    reference_summary = match_payload["reference_summary"]
    match_summary_row = pd.Series(
        {
            "capture_id": size_row["capture_id"],
            "item_class": detection.item_class,
            "composition_profile": match_payload["composition_profile"],
            "confidence": detection.confidence,
            "width_cm": estimate.width_cm,
            "height_cm": estimate.height_cm,
            "depth_cm": estimate.depth_cm,
            "distance_cm": estimate.distance_cm,
            "size_category": size_label,
            "candidate_count": reference_summary["candidate_count"],
            "material_family": reference_summary["material_family"],
            "weight_low_kg": reference_summary["weight_range_kg"][0],
            "weight_high_kg": reference_summary["weight_range_kg"][1],
            "weight_reference_count": reference_summary["weight_reference_count"],
            "weight_imputed_count": reference_summary["weight_imputed_count"],
            "weight_missing_count": reference_summary["weight_missing_count"],
            "weight_source_note": reference_summary["weight_source_note"],
            "reference_size_bin": reference_summary["size_bin"],
        }
    )

    factors = load_lca_factor_table(factor_path)
    if use_material_profiles:
        profiles = load_material_profile_table(material_profile_path)
        selected_profiles = select_material_profiles(
            profiles,
            str(match_summary_row.get("composition_profile", detection.item_class)),
        )
        lca_payload = build_lca_payload(
            match_summary_row,
            composition_factor_rows=build_composition_factor_rows(
                factors,
                selected_profiles,
            ),
            condition_status=condition_status,
        )
    else:
        lca_payload = build_lca_payload(
            match_summary_row,
            factor_rows=select_lca_factors(
                factors,
                str(match_summary_row.get("material_family", "unknown")),
            ),
            condition_status=condition_status,
        )

    return {
        "size_estimate": {
            "width_cm": estimate.width_cm,
            "height_cm": estimate.height_cm,
            "depth_cm": estimate.depth_cm,
            "distance_cm": estimate.distance_cm,
            "size_category": size_label,
            "roi": list(roi),
            "roi_refined": list(estimate.roi_refined),
        },
        "reference_match": match_payload,
        "lca_estimate": lca_payload,
    }


def draw_detections(
    image: np.ndarray,
    detections: list[EnsembleDetection],
    output_path: Path,
) -> None:
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        color = (40, 180, 80)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.item_class} {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def run_one_vs_rest_ensemble(
    model_paths: dict[str, Path],
    image_path: Path | None,
    capture_dir: Path | None,
    rgb_image_name: str,
    output_dir: Path,
    confidence: float,
    class_thresholds: dict[str, float],
    image_size: int,
    max_per_model: int,
    max_detections: int,
    ensemble_iou: float,
    reference_path: Path,
    factor_path: Path,
    material_profile_path: Path,
    use_material_profiles: bool,
    condition_status: str,
    top_n: int,
    save_image: bool,
) -> dict[str, Any]:
    if image_path is None and capture_dir is None:
        raise ValueError("Provide either --image or --capture-dir")
    if image_path is not None and capture_dir is not None:
        raise ValueError("Provide only one of --image or --capture-dir")

    if capture_dir is not None:
        source_image_path = capture_dir / rgb_image_name
    else:
        source_image_path = image_path
    assert source_image_path is not None

    image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {source_image_path}")

    raw_detections = run_specialist_models(
        image=image,
        model_paths=model_paths,
        confidence=confidence,
        class_thresholds=class_thresholds,
        image_size=image_size,
        max_per_model=max_per_model,
    )
    merged_detections = non_max_suppression(raw_detections, ensemble_iou)[:max_detections]
    selected = merged_detections[0] if merged_detections else None

    route_payload = None
    if selected is not None and capture_dir is not None:
        route_payload = build_route_payload(
            capture_dir=capture_dir,
            rgb_image_name=rgb_image_name,
            detection=selected,
            reference_path=reference_path,
            factor_path=factor_path,
            material_profile_path=material_profile_path,
            use_material_profiles=use_material_profiles,
            condition_status=condition_status,
            top_n=top_n,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_image_path.stem
    payload = {
        "source_image": str(source_image_path),
        "selected_detection": None if selected is None else asdict(selected),
        "detections": [asdict(detection) for detection in merged_detections],
        "raw_detections": [asdict(detection) for detection in raw_detections],
        "route_payload": route_payload,
    }
    json_path = output_dir / f"{stem}_one_vs_rest_ensemble.json"
    json_path.write_text(json.dumps(payload, indent=2))
    payload["output_json"] = str(json_path)

    if save_image:
        image_output_path = output_dir / f"{stem}_one_vs_rest_ensemble.png"
        draw_detections(image, merged_detections, image_output_path)
        payload["output_image"] = str(image_output_path)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a one-vs-rest YOLO specialist-model ensemble."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Specialist model mapping in the form class_name=path/to/best.pt. Repeat for each class.",
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument("--rgb-image-name", type=str, default="rgb.png")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/one_vs_rest_ensemble"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Optional class threshold in the form class_name=0.35. Repeat as needed.",
    )
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--max-per-model", type=int, default=3)
    parser.add_argument("--max-detections", type=int, default=1)
    parser.add_argument("--ensemble-iou", type=float, default=0.5)
    parser.add_argument("--reference", type=Path, default=Path("data/interim/ikea_reference_cleaned.csv"))
    parser.add_argument("--factor-table", type=Path, default=DEFAULT_LCA_FACTOR_PATH)
    parser.add_argument("--material-profiles", type=Path, default=DEFAULT_MATERIAL_PROFILE_PATH)
    parser.add_argument("--no-material-profiles", action="store_true")
    parser.add_argument(
        "--condition",
        choices=sorted(CONDITION_STATUSES),
        default="unknown",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--save-image", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_one_vs_rest_ensemble(
        model_paths=parse_key_value_path(args.model),
        image_path=args.image,
        capture_dir=args.capture_dir,
        rgb_image_name=args.rgb_image_name,
        output_dir=args.output_dir,
        confidence=args.confidence,
        class_thresholds=parse_key_value_float(args.threshold),
        image_size=args.image_size,
        max_per_model=args.max_per_model,
        max_detections=args.max_detections,
        ensemble_iou=args.ensemble_iou,
        reference_path=args.reference,
        factor_path=args.factor_table,
        material_profile_path=args.material_profiles,
        use_material_profiles=not args.no_material_profiles,
        condition_status=args.condition,
        top_n=args.top_n,
        save_image=args.save_image,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

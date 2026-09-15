"""Create calibrated single-class YOLO pseudo-labels from an unseen image pool."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from urllib.parse import unquote

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def excluded_images(exports: list[Path], source_root: Path) -> set[Path]:
    excluded = set()
    for export in exports:
        tasks = json.loads(export.read_text(encoding="utf-8"))
        for task in tasks:
            value = unquote(task["data"]["image"])
            relative = value.split("?d=", 1)[1]
            prefix = source_root.name + "/"
            if relative.startswith(prefix):
                relative = relative[len(prefix):]
            excluded.add((source_root / Path(relative)).resolve())
    return excluded


def pseudo_label_pool(
    input_dir: Path,
    source_root: Path,
    model_path: Path,
    output_dir: Path,
    exclude_exports: list[Path],
    threshold: float,
    class_id: int,
    image_size: int,
) -> dict[str, int | float]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    image_output = output_dir / "images" / "train"
    label_output = output_dir / "labels" / "train"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    excluded = excluded_images(exclude_exports, source_root)
    candidates = sorted(
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve() not in excluded
    )
    model = YOLO(str(model_path.resolve()))
    audit_rows = []
    accepted = 0
    for number, image_path in enumerate(candidates, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        result = model.predict(
            source=str(image_path),
            conf=0.001,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )[0]
        confidence = 0.0
        box_values = None
        if len(result.boxes):
            best_index = int(result.boxes.conf.argmax().item())
            confidence = float(result.boxes.conf[best_index].item())
            box_values = [float(value) for value in result.boxes.xyxy[best_index].tolist()]

        status = "accepted" if box_values and confidence >= threshold else "review"
        output_image = output_label = ""
        if status == "accepted":
            x1, y1, x2, y2 = box_values
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            shutil.copy2(image_path, image_output / image_path.name)
            label_path = label_output / f"{image_path.stem}.txt"
            label_path.write_text(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n",
                encoding="utf-8",
            )
            output_image = str(image_output / image_path.name)
            output_label = str(label_path)
            accepted += 1

        audit_rows.append(
            {
                "source_image": str(image_path),
                "status": status,
                "confidence": confidence,
                "bbox_xyxy": json.dumps(box_values) if box_values else "",
                "output_image": output_image,
                "output_label": output_label,
            }
        )
        if number % 50 == 0 or number == len(candidates):
            print(f"Processed {number}/{len(candidates)}; accepted {accepted}")

    with (output_dir / "pseudo_label_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "pool_images": len(candidates),
        "excluded_reviewed_or_holdout": len(excluded),
        "accepted": accepted,
        "sent_to_review": len(candidates) - accepted,
        "confidence_threshold": threshold,
        "class_id": class_id,
        "training_only": True,
    }
    (output_dir / "pseudo_label_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-export", action="append", type=Path, default=[])
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=960)
    args = parser.parse_args()
    summary = pseudo_label_pool(
        args.input_dir,
        args.source_root,
        args.model,
        args.output_dir,
        args.exclude_export,
        args.threshold,
        args.class_id,
        args.image_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

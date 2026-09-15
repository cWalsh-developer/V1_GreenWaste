"""Calibrate a single-class YOLO confidence threshold on held-out data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def read_ground_truth(label_path: Path, width: int, height: int) -> list[tuple[float, ...]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed label in {label_path}: {line}")
        _, x_center, y_center, box_width, box_height = map(float, parts)
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        boxes.append(
            (
                x_center - box_width / 2,
                y_center - box_height / 2,
                x_center + box_width / 2,
                y_center + box_height / 2,
            )
        )
    return boxes


def metrics_at_threshold(rows: list[dict], threshold: float, iou_threshold: float) -> dict:
    true_positive = false_positive = false_negative = 0
    for row in rows:
        prediction_kept = row["confidence"] >= threshold and row["prediction_box"] is not None
        ground_truth = row["ground_truth_boxes"]
        if prediction_kept and ground_truth:
            best_iou = max(box_iou(row["prediction_box"], box) for box in ground_truth)
            if best_iou >= iou_threshold:
                true_positive += 1
                false_negative += max(0, len(ground_truth) - 1)
            else:
                false_positive += 1
                false_negative += len(ground_truth)
        elif prediction_kept:
            false_positive += 1
        elif ground_truth:
            false_negative += len(ground_truth)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calibrate(
    model_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    target_precision: float,
    iou_threshold: float,
    image_size: int,
) -> dict:
    image_dir = dataset_dir / "images" / "test"
    label_dir = dataset_dir / "labels" / "test"
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    model = YOLO(str(model_path.resolve()))
    rows = []
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        ground_truth = read_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        result = model.predict(
            source=str(image_path), conf=0.001, iou=0.5, imgsz=image_size, verbose=False
        )[0]
        prediction_box = None
        confidence = 0.0
        if len(result.boxes):
            best_index = int(result.boxes.conf.argmax().item())
            confidence = float(result.boxes.conf[best_index].item())
            prediction_box = tuple(float(value) for value in result.boxes.xyxy[best_index].tolist())
        rows.append(
            {
                "image": str(image_path),
                "ground_truth_boxes": ground_truth,
                "prediction_box": prediction_box,
                "confidence": confidence,
            }
        )

    thresholds = sorted({0.0, 1.0, *[round(index / 1000, 3) for index in range(1, 1000)]})
    curve = [metrics_at_threshold(rows, threshold, iou_threshold) for threshold in thresholds]
    eligible = [point for point in curve if point["precision"] >= target_precision and point["true_positives"] > 0]
    if not eligible:
        raise ValueError(f"No threshold achieved target precision {target_precision}")
    selected = max(eligible, key=lambda point: (point["recall"], -point["threshold"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "confidence_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)
    prediction_rows = []
    for row in rows:
        best_iou = (
            max(box_iou(row["prediction_box"], box) for box in row["ground_truth_boxes"])
            if row["prediction_box"] and row["ground_truth_boxes"]
            else 0.0
        )
        prediction_rows.append(
            {
                "image": row["image"],
                "has_ground_truth": bool(row["ground_truth_boxes"]),
                "confidence": row["confidence"],
                "best_iou": best_iou,
            }
        )
    with (output_dir / "prediction_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)

    summary = {
        "model": str(model_path),
        "evaluation_images": len(images),
        "positive_instances": sum(len(row["ground_truth_boxes"]) for row in rows),
        "background_images": sum(not row["ground_truth_boxes"] for row in rows),
        "iou_threshold": iou_threshold,
        "target_precision": target_precision,
        "selected_operating_point": selected,
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=960)
    args = parser.parse_args()
    summary = calibrate(
        args.model,
        args.dataset_dir,
        args.output_dir,
        args.target_precision,
        args.iou_threshold,
        args.image_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

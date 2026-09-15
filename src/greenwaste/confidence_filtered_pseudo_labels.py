from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from PIL import Image

from .one_vs_rest_ensemble import box_iou, parse_key_value_float, parse_key_value_path

try:
    import torch
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "ultralytics and torch are required. Install with: pip install ultralytics torch"
    ) from exc


TARGET_CLASSES = [
    "beds_mattresses",
    "chair_seating",
    "sofa",
    "storage",
    "tables_desks",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ModelVote:
    model_path: str
    predicted_class: str | None
    confidence: float
    second_confidence: float
    margin: float
    bbox_xyxy: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class PseudoDecision:
    source_path: str
    expected_class: str
    status: str
    reason: str
    accepted_class: str | None
    confidence: float | None
    agreement_count: int
    teacher_count: int
    bbox_xyxy: tuple[float, float, float, float] | None
    box_area_ratio: float | None
    output_image_path: str | None
    output_label_path: str | None
    votes_json: str


def collect_candidates(input_dir: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    for item_class in TARGET_CLASSES:
        class_dir = input_dir / item_class
        if not class_dir.exists():
            continue
        candidates.extend(
            (path, item_class)
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return candidates


def limit_candidates_per_class(
    candidates: list[tuple[Path, str]],
    limit_per_class: int | None,
    sample_seed: int | None = None,
) -> list[tuple[Path, str]]:
    if limit_per_class is None:
        return candidates
    grouped = {item_class: [] for item_class in TARGET_CLASSES}
    for candidate in candidates:
        grouped[candidate[1]].append(candidate)
    selected: list[tuple[Path, str]] = []
    sampler = random.Random(sample_seed)
    for item_class in TARGET_CLASSES:
        class_candidates = grouped[item_class]
        if sample_seed is not None:
            class_candidates = sampler.sample(
                class_candidates, min(limit_per_class, len(class_candidates))
            )
        else:
            class_candidates = class_candidates[:limit_per_class]
        selected.extend(class_candidates)
    return selected


def filter_candidates_by_class(
    candidates: list[tuple[Path, str]],
    include_classes: list[str] | None,
) -> list[tuple[Path, str]]:
    if not include_classes:
        return candidates
    included = set(include_classes)
    return [candidate for candidate in candidates if candidate[1] in included]


def best_vote_from_result(result: Any, model_path: Path) -> ModelVote:
    best_by_class: dict[str, tuple[float, tuple[float, float, float, float]]] = {}
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(result.names[class_id])
        if class_name not in TARGET_CLASSES:
            continue
        confidence = float(box.conf.item())
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        current = best_by_class.get(class_name)
        if current is None or confidence > current[0]:
            best_by_class[class_name] = (confidence, xyxy)

    ranked = sorted(best_by_class.items(), key=lambda item: item[1][0], reverse=True)
    if not ranked:
        return ModelVote(str(model_path), None, 0.0, 0.0, 0.0, None)

    predicted_class, (confidence, xyxy) = ranked[0]
    second_confidence = ranked[1][1][0] if len(ranked) > 1 else 0.0
    return ModelVote(
        model_path=str(model_path),
        predicted_class=predicted_class,
        confidence=confidence,
        second_confidence=second_confidence,
        margin=confidence - second_confidence,
        bbox_xyxy=xyxy,
    )


def median_box(votes: Iterable[ModelVote]) -> tuple[float, float, float, float]:
    boxes = [vote.bbox_xyxy for vote in votes if vote.bbox_xyxy is not None]
    if not boxes:
        raise ValueError("At least one vote with a bounding box is required")
    return tuple(float(median(values)) for values in zip(*boxes))  # type: ignore[return-value]


def area_ratio(
    box: tuple[float, float, float, float], image_width: int, image_height: int
) -> float:
    x1, y1, x2, y2 = box
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    return area / max(float(image_width * image_height), 1.0)


def evaluate_votes(
    votes: list[ModelVote],
    expected_class: str,
    image_width: int,
    image_height: int,
    min_confidence: float,
    min_margin: float,
    min_agreement: int,
    min_box_iou: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[str, str, tuple[float, float, float, float] | None, float | None, int]:
    agreeing = [
        vote
        for vote in votes
        if vote.predicted_class == expected_class
        and vote.confidence >= min_confidence
        and vote.margin >= min_margin
        and vote.bbox_xyxy is not None
    ]
    if len(agreeing) < min_agreement:
        predicted = [vote.predicted_class for vote in votes if vote.predicted_class]
        if predicted and expected_class not in predicted:
            reason = "teacher_class_disagreement"
        else:
            reason = "insufficient_high_confidence_agreement"
        return "review", reason, None, None, len(agreeing)

    consensus_box = median_box(agreeing)
    overlapping = [
        vote
        for vote in agreeing
        if vote.bbox_xyxy is not None
        and box_iou(vote.bbox_xyxy, consensus_box) >= min_box_iou
    ]
    if len(overlapping) < min_agreement:
        return "review", "teacher_box_disagreement", None, None, len(overlapping)

    consensus_box = median_box(overlapping)
    ratio = area_ratio(consensus_box, image_width, image_height)
    if ratio < min_area_ratio:
        return "review", "box_too_small", consensus_box, ratio, len(overlapping)
    if ratio > max_area_ratio:
        return "review", "box_too_large", consensus_box, ratio, len(overlapping)
    return "accepted", "accepted", consensus_box, ratio, len(overlapping)


def xyxy_to_yolo(
    box: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return (
        (x1 + width / 2.0) / image_width,
        (y1 + height / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def safe_output_name(source_path: Path, expected_class: str) -> str:
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in source_path.stem
    )
    return f"{expected_class}__{safe_stem}"


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(TARGET_CLASSES))
    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "# Validation and test must come from the untouched manually labelled dataset.\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, decisions: list[PseudoDecision]) -> None:
    fieldnames = list(PseudoDecision.__dataclass_fields__)
    with (output_dir / "pseudo_label_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(asdict(decision))

    summary: dict[str, Any] = {
        "total": len(decisions),
        "accepted": sum(decision.status == "accepted" for decision in decisions),
        "review": sum(decision.status == "review" for decision in decisions),
        "by_class": {},
        "by_reason": {},
    }
    for decision in decisions:
        summary["by_class"].setdefault(decision.expected_class, {"accepted": 0, "review": 0})
        summary["by_class"][decision.expected_class][decision.status] += 1
        summary["by_reason"][decision.reason] = summary["by_reason"].get(decision.reason, 0) + 1
    (output_dir / "pseudo_label_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def generate_confidence_filtered_pseudo_labels(
    input_dir: Path,
    output_dir: Path,
    teacher_paths: list[Path],
    min_confidence: float = 0.80,
    class_thresholds: dict[str, float] | None = None,
    min_margin: float = 0.20,
    min_agreement: int = 3,
    min_box_iou: float = 0.50,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    image_size: int = 960,
    inference_confidence: float = 0.05,
    limit: int | None = None,
    limit_per_class: int | None = None,
    sample_seed: int | None = None,
    include_classes: list[str] | None = None,
) -> list[PseudoDecision]:
    if not teacher_paths:
        raise ValueError("At least one teacher model is required")
    if min_agreement < 1 or min_agreement > len(teacher_paths):
        raise ValueError("min_agreement must be between 1 and the teacher count")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )

    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    write_data_yaml(output_dir)

    resolved_teachers = [path.resolve() for path in teacher_paths]
    for path in resolved_teachers:
        if not path.exists():
            raise FileNotFoundError(f"Teacher weights not found: {path}")
    models = [YOLO(str(path)) for path in resolved_teachers]
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    candidates = filter_candidates_by_class(
        collect_candidates(input_dir), include_classes
    )
    candidates = limit_candidates_per_class(candidates, limit_per_class, sample_seed)
    if limit is not None:
        candidates = candidates[:limit]

    decisions: list[PseudoDecision] = []
    thresholds = class_thresholds or {}
    for index, (source_path, expected_class) in enumerate(candidates, start=1):
        with Image.open(source_path) as image:
            image_width, image_height = image.size

        votes = []
        for model, model_path in zip(models, resolved_teachers):
            result = model.predict(
                source=str(source_path),
                conf=inference_confidence,
                iou=0.5,
                imgsz=image_size,
                device=device,
                verbose=False,
            )[0]
            votes.append(best_vote_from_result(result, model_path))

        required_confidence = thresholds.get(expected_class, min_confidence)
        status, reason, box, ratio, agreement = evaluate_votes(
            votes=votes,
            expected_class=expected_class,
            image_width=image_width,
            image_height=image_height,
            min_confidence=required_confidence,
            min_margin=min_margin,
            min_agreement=min_agreement,
            min_box_iou=min_box_iou,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )

        output_image_path = None
        output_label_path = None
        accepted_confidence = None
        if status == "accepted" and box is not None:
            accepted_votes = [
                vote
                for vote in votes
                if vote.predicted_class == expected_class
                and vote.confidence >= required_confidence
                and vote.margin >= min_margin
            ]
            accepted_confidence = float(median(vote.confidence for vote in accepted_votes))
            output_stem = safe_output_name(source_path, expected_class)
            output_image = output_dir / "images" / "train" / f"{output_stem}{source_path.suffix.lower()}"
            output_label = output_dir / "labels" / "train" / f"{output_stem}.txt"
            yolo_box = xyxy_to_yolo(box, image_width, image_height)
            class_id = TARGET_CLASSES.index(expected_class)
            shutil.copy2(source_path, output_image)
            output_label.write_text(
                f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} "
                f"{yolo_box[2]:.6f} {yolo_box[3]:.6f}\n",
                encoding="utf-8",
            )
            output_image_path = str(output_image)
            output_label_path = str(output_label)

        decisions.append(
            PseudoDecision(
                source_path=str(source_path),
                expected_class=expected_class,
                status=status,
                reason=reason,
                accepted_class=expected_class if status == "accepted" else None,
                confidence=accepted_confidence,
                agreement_count=agreement,
                teacher_count=len(votes),
                bbox_xyxy=box,
                box_area_ratio=ratio,
                output_image_path=output_image_path,
                output_label_path=output_label_path,
                votes_json=json.dumps([asdict(vote) for vote in votes]),
            )
        )
        if index % 25 == 0 or index == len(candidates):
            accepted = sum(decision.status == "accepted" for decision in decisions)
            print(f"Processed {index}/{len(candidates)}; accepted {accepted}")

    write_manifest(output_dir, decisions)
    return decisions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create conservative, training-only YOLO pseudo labels using teacher consensus"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/realsense_for_annotation_grouped_20260807"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher", action="append", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument(
        "--class-threshold",
        action="append",
        default=[],
        metavar="CLASS=VALUE",
    )
    parser.add_argument("--min-margin", type=float, default=0.20)
    parser.add_argument("--min-agreement", type=int, default=3)
    parser.add_argument("--min-box-iou", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--inference-confidence", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="Balanced pilot limit applied independently to each target class.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Randomly sample balanced pilot images reproducibly.",
    )
    parser.add_argument(
        "--include-class",
        action="append",
        choices=TARGET_CLASSES,
        help="Process only this class. Repeat to include more than one class.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decisions = generate_confidence_filtered_pseudo_labels(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        teacher_paths=args.teacher,
        min_confidence=args.min_confidence,
        class_thresholds=parse_key_value_float(args.class_threshold),
        min_margin=args.min_margin,
        min_agreement=args.min_agreement,
        min_box_iou=args.min_box_iou,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        image_size=args.image_size,
        inference_confidence=args.inference_confidence,
        limit=args.limit,
        limit_per_class=args.limit_per_class,
        sample_seed=args.sample_seed,
        include_classes=args.include_class,
    )
    accepted = sum(decision.status == "accepted" for decision in decisions)
    print(f"Accepted {accepted}/{len(decisions)} pseudo labels")
    print(f"Review manifest: {args.output_dir / 'pseudo_label_manifest.csv'}")


if __name__ == "__main__":
    main()

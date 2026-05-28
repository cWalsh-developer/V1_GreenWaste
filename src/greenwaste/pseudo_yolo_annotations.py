from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:
    import torch
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "ultralytics and torch are required. Install with: pip install ultralytics"
    ) from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TARGET_CLASSES = [
    "beds_mattresses",
    "chair_seating",
    "sofa",
    "storage",
    "tables_desks",
]

YOLO_TO_BROAD = {
    "bed": "beds_mattresses",
    "bench": "chair_seating",
    "chair": "chair_seating",
    "couch": "sofa",
    "dining table": "tables_desks",
}


@dataclass(frozen=True)
class PseudoAnnotation:
    source_path: Path
    split: str
    raw_label: str
    broad_label: str
    yolo_class: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    image_width: int
    image_height: int
    output_image_path: Path
    output_label_path: Path


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def broad_item_label(raw_label: str) -> str | None:
    label = raw_label.lower()
    accessory_terms = (
        "accessories",
        "parts",
        "cover",
        "covers",
        "cushion",
        "pad",
        "blanket",
        "sheet",
        "textile",
        "curtain",
        "canopy",
        "topper",
    )
    if any(term in label for term in accessory_terms):
        return None

    if any(term in label for term in ("sofa", "loveseat", "sectional", "futon")):
        return "sofa"
    if any(term in label for term in ("bed", "crib", "mattress")):
        return "beds_mattresses"
    if any(
        term in label
        for term in (
            "storage",
            "cabinet",
            "wardrobe",
            "dresser",
            "drawer",
            "shelf",
            "shelves",
            "bookcase",
            "bookshelf",
            "cart",
            "basket",
            "organizer",
            "sideboard",
            "tv_stands",
        )
    ):
        return "storage"
    if any(
        term in label
        for term in (
            "computer_desks",
            "office_desks",
            "gaming_desks",
            "kids_desks",
            "children_s_desk",
            "desks_for_home",
            "desks_for_office",
            "laptop_tables",
            "conference_meeting_tables",
            "conference_tables",
            "meeting_tables",
            "dining_tables",
            "coffee_tables",
            "side_tables",
            "end_tables",
            "nesting_tables",
            "bar_tables",
            "round_tables",
            "extendable_tables",
            "multifunctional_tables",
            "dining_sets",
            "bistro_sets",
            "outdoor_dining_sets",
            "table_tops",
            "nightstands",
            "changing_tables",
            "vanities",
        )
    ):
        return "tables_desks"
    if any(
        term in label
        for term in (
            "dining_chairs",
            "upholstered_chairs",
            "outdoor_dining_chairs",
            "outdoor_patio_lounge_chairs",
            "gaming_chairs",
            "conference_chairs",
            "desk_chairs",
            "armchairs",
            "stools",
            "benches",
            "bar_stools",
            "high_chairs",
            "kids_chairs",
            "lounge_chairs",
            "reclining_chairs",
            "chaise",
        )
    ):
        return "chair_seating"

    return None


def safe_stem(path: Path, raw_label: str) -> str:
    label_part = "".join(ch if ch.isalnum() else "_" for ch in raw_label)[:80]
    stem_part = "".join(ch if ch.isalnum() else "_" for ch in path.stem)[:80]
    return f"{label_part}__{stem_part}"


def xyxy_to_yolo(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def expand_xyxy(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    pad_x = box_width * padding
    pad_y = box_height * padding

    return (
        max(x1 - pad_x, 0.0),
        max(y1 - pad_y, 0.0),
        min(x2 + pad_x, float(image_width)),
        min(y2 + pad_y, float(image_height)),
    )


def box_area_ratio(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> float:
    x1, y1, x2, y2 = xyxy
    box_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    image_area = max(float(image_width * image_height), 1.0)
    return box_area / image_area


def foreground_xyxy_from_white_background(
    image_path: Path,
    threshold: int,
    padding: float,
) -> tuple[float, float, float, float] | None:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        image_width, image_height = rgb_image.size
        pixels = rgb_image.load()

        x_min = image_width
        y_min = image_height
        x_max = -1
        y_max = -1

        for y in range(image_height):
            for x in range(image_width):
                red, green, blue = pixels[x, y]
                if red < threshold or green < threshold or blue < threshold:
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

    if x_max < x_min or y_max < y_min:
        return None

    return expand_xyxy(
        xyxy=(float(x_min), float(y_min), float(x_max + 1), float(y_max + 1)),
        image_width=image_width,
        image_height=image_height,
        padding=padding,
    )


def choose_matching_box(result, expected_broad_label: str, min_confidence: float):
    best_box = None
    best_confidence = 0.0
    best_yolo_class = None

    for box in result.boxes:
        class_id = int(box.cls.item())
        yolo_class = result.names[class_id]
        broad_label = YOLO_TO_BROAD.get(yolo_class)
        confidence = float(box.conf.item())

        if broad_label != expected_broad_label:
            continue
        if confidence < min_confidence:
            continue
        if confidence > best_confidence:
            best_box = box
            best_confidence = confidence
            best_yolo_class = yolo_class

    return best_box, best_yolo_class, best_confidence


def collect_candidates(input_dir: Path, max_per_class: int | None) -> list[tuple[Path, str, str]]:
    grouped: dict[str, list[tuple[Path, str, str]]] = {label: [] for label in TARGET_CLASSES}

    for label_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        broad_label = broad_item_label(label_dir.name)
        if broad_label is None:
            continue

        for image_path in image_files(label_dir):
            grouped[broad_label].append((image_path, label_dir.name, broad_label))

    candidates: list[tuple[Path, str, str]] = []
    for broad_label in TARGET_CLASSES:
        items = grouped[broad_label]
        if max_per_class is not None:
            items = items[:max_per_class]
        candidates.extend(items)

    return candidates


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(TARGET_CLASSES))
    data_yaml = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml)


def generate_pseudo_dataset(
    input_dir: Path,
    output_dir: Path,
    weights: Path,
    confidence: float,
    image_size: int,
    box_padding: float,
    foreground_fallback: bool,
    prefer_foreground_box: bool,
    foreground_threshold: int,
    foreground_padding: float,
    min_box_area_ratio: float,
    val_fraction: float,
    seed: int,
    max_per_class: int | None,
) -> list[PseudoAnnotation]:
    candidates = collect_candidates(input_dir=input_dir, max_per_class=max_per_class)
    random.Random(seed).shuffle(candidates)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else "cpu"
    annotations: list[PseudoAnnotation] = []

    for index, (image_path, raw_label, broad_label) in enumerate(candidates, start=1):
        split = "val" if random.Random(seed + index).random() < val_fraction else "train"
        result = model.predict(
            source=str(image_path),
            conf=confidence,
            iou=0.5,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]

        box, yolo_class, box_confidence = choose_matching_box(
            result=result,
            expected_broad_label=broad_label,
            min_confidence=confidence,
        )
        if box is None or yolo_class is None:
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        raw_xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        xyxy = expand_xyxy(
            xyxy=raw_xyxy,
            image_width=image_width,
            image_height=image_height,
            padding=box_padding,
        )
        should_use_foreground = foreground_fallback and (
            prefer_foreground_box
            or box_area_ratio(xyxy, image_width, image_height) < min_box_area_ratio
        )
        if should_use_foreground:
            foreground_xyxy = foreground_xyxy_from_white_background(
                image_path=image_path,
                threshold=foreground_threshold,
                padding=foreground_padding,
            )
            if foreground_xyxy is not None:
                xyxy = foreground_xyxy

        yolo_box = xyxy_to_yolo(
            xyxy=xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        class_id = TARGET_CLASSES.index(broad_label)

        output_stem = safe_stem(image_path, raw_label)
        output_image_path = output_dir / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
        output_label_path = output_dir / "labels" / split / f"{output_stem}.txt"

        shutil.copy2(image_path, output_image_path)
        output_label_path.write_text(
            f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
        )

        annotations.append(
            PseudoAnnotation(
                source_path=image_path,
                split=split,
                raw_label=raw_label,
                broad_label=broad_label,
                yolo_class=yolo_class,
                confidence=box_confidence,
                xyxy=xyxy,
                image_width=image_width,
                image_height=image_height,
                output_image_path=output_image_path,
                output_label_path=output_label_path,
            )
        )

        if index % 50 == 0:
            print(f"Processed {index}/{len(candidates)} images; accepted {len(annotations)}")

    write_data_yaml(output_dir)
    write_summary_csv(output_dir, annotations)
    return annotations


def write_summary_csv(output_dir: Path, annotations: list[PseudoAnnotation]) -> None:
    summary_path = output_dir / "pseudo_annotations.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_path",
                "split",
                "raw_label",
                "broad_label",
                "yolo_class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "image_width",
                "image_height",
                "output_image_path",
                "output_label_path",
            ],
        )
        writer.writeheader()
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.xyxy
            writer.writerow(
                {
                    "source_path": annotation.source_path,
                    "split": annotation.split,
                    "raw_label": annotation.raw_label,
                    "broad_label": annotation.broad_label,
                    "yolo_class": annotation.yolo_class,
                    "confidence": f"{annotation.confidence:.6f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "image_width": annotation.image_width,
                    "image_height": annotation.image_height,
                    "output_image_path": annotation.output_image_path,
                    "output_label_path": annotation.output_label_path,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate YOLO-format pseudo annotations from labelled image folders"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/realsense/labelled"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/yolo_pseudo"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/yolo11s.pt"),
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.25,
        help="Expand accepted YOLO boxes by this fraction of their width/height.",
    )
    parser.add_argument(
        "--foreground-fallback",
        action="store_true",
        help="Use a white-background foreground box when the YOLO box is too partial.",
    )
    parser.add_argument(
        "--prefer-foreground-box",
        action="store_true",
        help="Use the foreground box whenever it is available after YOLO confirms the class.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=245,
        help="RGB threshold for white-background foreground extraction.",
    )
    parser.add_argument(
        "--foreground-padding",
        type=float,
        default=0.08,
        help="Expand foreground fallback boxes by this fraction of their width/height.",
    )
    parser.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.45,
        help="Use foreground fallback when an expanded YOLO box covers less than this image fraction.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    annotations = generate_pseudo_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights=args.weights,
        confidence=args.confidence,
        image_size=args.image_size,
        box_padding=args.box_padding,
        foreground_fallback=args.foreground_fallback,
        prefer_foreground_box=args.prefer_foreground_box,
        foreground_threshold=args.foreground_threshold,
        foreground_padding=args.foreground_padding,
        min_box_area_ratio=args.min_box_area_ratio,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    print(f"Accepted pseudo annotations: {len(annotations)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

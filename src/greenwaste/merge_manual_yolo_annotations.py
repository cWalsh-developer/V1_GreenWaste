from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
TARGET_CLASSES = [
    "beds_mattresses",
    "chair_seating",
    "sofa",
    "storage",
    "tables_desks",
]
STORAGE_CLASS_ID = TARGET_CLASSES.index("storage")


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(TARGET_CLASSES))
    data_yaml = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml)


def copy_existing_dataset(source_dir: Path, output_dir: Path) -> int:
    copied = 0
    for split in ("train", "val"):
        source_images = source_dir / "images" / split
        source_labels = source_dir / "labels" / split
        target_images = output_dir / "images" / split
        target_labels = output_dir / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        if not source_images.exists() or not source_labels.exists():
            continue

        for image_path in source_images.iterdir():
            if not image_path.is_file():
                continue
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            target_image_path = target_images / f"pseudo__{image_path.name}"
            target_label_path = target_labels / f"pseudo__{label_path.name}"
            shutil.copy2(image_path, target_image_path)
            shutil.copy2(label_path, target_label_path)
            copied += 1

    return copied


def build_image_index(raw_image_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for image_path in raw_image_root.rglob(f"*{extension}"):
            image_index.setdefault(image_path.stem, image_path)
    return image_index


def remap_label_file(source_label_path: Path, target_label_path: Path, class_id: int) -> None:
    lines = []
    for line in source_label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, width, height = parts
        lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

    target_label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def merge_manual_annotations(
    pseudo_dir: Path,
    manual_label_groups: list[tuple[str, list[Path]]],
    raw_image_root: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> tuple[int, dict[str, int], list[Path], list[str]]:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    pseudo_count = copy_existing_dataset(pseudo_dir, output_dir)
    image_index = build_image_index(raw_image_root)

    rng = random.Random(seed)
    copied_manual: dict[str, int] = {}
    missing_images: list[Path] = []
    duplicate_stems: list[str] = []
    seen_manual_stems: set[str] = set()

    for class_name, label_dirs in manual_label_groups:
        class_id = TARGET_CLASSES.index(class_name)
        manual_label_paths = []
        for label_dir in label_dirs:
            manual_label_paths.extend(sorted(label_dir.glob("*.txt")))

        rng.shuffle(manual_label_paths)
        copied_manual.setdefault(class_name, 0)

        for label_path in manual_label_paths:
            if label_path.stem in seen_manual_stems:
                duplicate_stems.append(label_path.stem)
                continue
            seen_manual_stems.add(label_path.stem)

            source_image_path = image_index.get(label_path.stem)
            if source_image_path is None:
                missing_images.append(label_path)
                continue

            split = "val" if rng.random() < val_fraction else "train"
            output_stem = f"manual_{class_name}__{label_path.stem}"
            target_image_path = (
                output_dir / "images" / split / f"{output_stem}{source_image_path.suffix.lower()}"
            )
            target_label_path = output_dir / "labels" / split / f"{output_stem}.txt"

            shutil.copy2(source_image_path, target_image_path)
            remap_label_file(label_path, target_label_path, class_id=class_id)
            copied_manual[class_name] += 1

    write_data_yaml(output_dir)
    return pseudo_count, copied_manual, missing_images, duplicate_stems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge manual storage YOLO annotations with the pseudo YOLO dataset"
    )
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        default=Path("data/processed/yolo_pseudo"),
    )
    parser.add_argument(
        "--manual-storage-label-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--manual-tables-label-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--raw-image-root",
        type=Path,
        default=Path("data/raw/realsense/labelled"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/yolo_combined"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manual_label_groups = [
        ("storage", args.manual_storage_label_dir),
        ("tables_desks", args.manual_tables_label_dir),
    ]
    manual_label_groups = [
        (class_name, label_dirs)
        for class_name, label_dirs in manual_label_groups
        if label_dirs
    ]
    if not manual_label_groups:
        parser.error("At least one manual label directory must be supplied.")

    pseudo_count, manual_counts, missing_images, duplicate_stems = merge_manual_annotations(
        pseudo_dir=args.pseudo_dir,
        manual_label_groups=manual_label_groups,
        raw_image_root=args.raw_image_root,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Copied pseudo-labelled images: {pseudo_count}")
    for class_name, count in manual_counts.items():
        print(f"Copied manual {class_name} images: {count}")
    print(f"Skipped duplicate manual stems: {len(duplicate_stems)}")
    if duplicate_stems:
        for stem in duplicate_stems[:20]:
            print(f"Duplicate manual stem: {stem}")
    print(f"Missing manual source images: {len(missing_images)}")
    if missing_images:
        for path in missing_images[:20]:
            print(f"Missing image for: {path}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

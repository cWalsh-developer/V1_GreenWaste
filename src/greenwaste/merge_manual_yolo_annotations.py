from __future__ import annotations

import argparse
import csv
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


def safe_stem_part(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


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


def build_image_index(raw_image_roots: list[Path]) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for raw_image_root in raw_image_roots:
        for extension in IMAGE_EXTENSIONS:
            for image_path in raw_image_root.rglob(f"*{extension}"):
                image_index.setdefault(image_path.stem, image_path)
    return image_index


def remove_duplicate_pseudo_annotations(
    output_dir: Path,
    manual_stems: set[str],
) -> list[dict[str, str]]:
    removed: list[dict[str, str]] = []
    safe_manual_stems = {safe_stem_part(stem) for stem in manual_stems}

    for split in ("train", "val"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or not image_path.name.startswith("pseudo__"):
                continue

            pseudo_stem = image_path.stem.removeprefix("pseudo__")
            is_duplicate = any(
                pseudo_stem == manual_stem
                or pseudo_stem.endswith(f"__{manual_stem}")
                for manual_stem in safe_manual_stems
            )
            if not is_duplicate:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            image_path.unlink()
            if label_path.exists():
                label_path.unlink()
            removed.append(
                {
                    "split": split,
                    "removed_pseudo_stem": image_path.stem,
                }
            )

    return removed


def write_removed_duplicates_csv(
    output_dir: Path,
    removed_duplicates: list[dict[str, str]],
) -> None:
    path = output_dir / "removed_duplicate_pseudo_images.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "removed_pseudo_stem"],
        )
        writer.writeheader()
        writer.writerows(removed_duplicates)


def stratified_splits(
    label_paths: list[Path],
    val_fraction: float,
    rng: random.Random,
) -> dict[Path, str]:
    if len(label_paths) < 2:
        return {label_path: "train" for label_path in label_paths}

    shuffled = label_paths.copy()
    rng.shuffle(shuffled)
    val_count = round(len(shuffled) * val_fraction)
    if val_fraction > 0 and label_paths:
        val_count = max(1, val_count)
    val_paths = set(shuffled[:val_count])
    return {
        label_path: "val" if label_path in val_paths else "train"
        for label_path in label_paths
    }


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
    raw_image_roots: list[Path],
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> tuple[
    int,
    dict[str, int],
    dict[str, dict[str, int]],
    list[Path],
    list[str],
    list[dict[str, str]],
]:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    pseudo_count = copy_existing_dataset(pseudo_dir, output_dir)
    image_index = build_image_index(raw_image_roots)

    rng = random.Random(seed)
    copied_manual: dict[str, int] = {}
    manual_split_counts: dict[str, dict[str, int]] = {}
    missing_images: list[Path] = []
    duplicate_stems: list[str] = []
    seen_manual_stems: set[str] = set()

    for class_name, label_dirs in manual_label_groups:
        class_id = TARGET_CLASSES.index(class_name)
        manual_label_paths = []
        for label_dir in label_dirs:
            manual_label_paths.extend(sorted(label_dir.glob("*.txt")))

        copied_manual.setdefault(class_name, 0)
        manual_split_counts.setdefault(class_name, {"train": 0, "val": 0})
        split_by_label = stratified_splits(manual_label_paths, val_fraction, rng)

        for label_path in manual_label_paths:
            if label_path.stem in seen_manual_stems:
                duplicate_stems.append(label_path.stem)
                continue
            seen_manual_stems.add(label_path.stem)

            source_image_path = image_index.get(label_path.stem)
            if source_image_path is None:
                missing_images.append(label_path)
                continue

            split = split_by_label[label_path]
            output_stem = f"manual_{class_name}__{label_path.stem}"
            target_image_path = (
                output_dir / "images" / split / f"{output_stem}{source_image_path.suffix.lower()}"
            )
            target_label_path = output_dir / "labels" / split / f"{output_stem}.txt"

            shutil.copy2(source_image_path, target_image_path)
            remap_label_file(label_path, target_label_path, class_id=class_id)
            copied_manual[class_name] += 1
            manual_split_counts[class_name][split] += 1

    removed_duplicates = remove_duplicate_pseudo_annotations(
        output_dir=output_dir,
        manual_stems=seen_manual_stems,
    )
    write_removed_duplicates_csv(output_dir, removed_duplicates)
    write_data_yaml(output_dir)
    return (
        pseudo_count,
        copied_manual,
        manual_split_counts,
        missing_images,
        duplicate_stems,
        removed_duplicates,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge manual YOLO annotations with the pseudo YOLO dataset"
    )
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        default=Path("data/processed/yolo_pseudo"),
    )
    parser.add_argument(
        "--manual-beds-label-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--manual-chair-label-dir",
        type=Path,
        action="append",
        default=[],
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
        action="append",
        default=[],
        help=(
            "Image root used to match manual label stems. Can be supplied more "
            "than once."
        ),
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
        ("beds_mattresses", args.manual_beds_label_dir),
        ("chair_seating", args.manual_chair_label_dir),
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

    raw_image_roots = args.raw_image_root or [Path("data/raw/realsense/labelled")]

    (
        pseudo_count,
        manual_counts,
        manual_split_counts,
        missing_images,
        duplicate_stems,
        removed_duplicates,
    ) = merge_manual_annotations(
        pseudo_dir=args.pseudo_dir,
        manual_label_groups=manual_label_groups,
        raw_image_roots=raw_image_roots,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Copied pseudo-labelled images: {pseudo_count}")
    for class_name, count in manual_counts.items():
        print(f"Copied manual {class_name} images: {count}")
        split_counts = manual_split_counts[class_name]
        print(
            f"  manual split: train={split_counts['train']}, "
            f"val={split_counts['val']}"
        )
    print(f"Removed duplicate pseudo-labelled images: {len(removed_duplicates)}")
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

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUGMENTED_SUFFIX = re.compile(r"__aug\d+$")


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path
    source_split: str
    group_key: str
    class_ids: tuple[int, ...]


def class_ids_from_label(label_path: Path) -> tuple[int, ...]:
    class_ids: list[int] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_ids.append(int(float(parts[0])))
    return tuple(sorted(set(class_ids)))


def find_image_for_label(image_dir: Path, label_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{label_path.stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def group_key_from_stem(stem: str) -> str:
    return AUGMENTED_SUFFIX.sub("", stem)


def collect_dataset_items(source_dir: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for split in ("train", "val"):
        image_dir = source_dir / "images" / split
        label_dir = source_dir / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue

        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image_for_label(image_dir, label_path)
            if image_path is None:
                continue
            class_ids = class_ids_from_label(label_path)
            if not class_ids:
                continue
            items.append(
                DatasetItem(
                    image_path=image_path,
                    label_path=label_path,
                    source_split=split,
                    group_key=group_key_from_stem(label_path.stem),
                    class_ids=class_ids,
                )
            )
    return items


def grouped_items(items: list[DatasetItem]) -> dict[str, list[DatasetItem]]:
    groups: dict[str, list[DatasetItem]] = {}
    for item in items:
        groups.setdefault(item.group_key, []).append(item)
    return groups


def primary_class(items: list[DatasetItem]) -> int:
    counts: dict[int, int] = {}
    for item in items:
        for class_id in item.class_ids:
            counts[class_id] = counts.get(class_id, 0) + 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def assign_groups_to_folds(
    groups: dict[str, list[DatasetItem]],
    folds: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    groups_by_class: dict[int, list[str]] = {}
    for group_key, items in groups.items():
        groups_by_class.setdefault(primary_class(items), []).append(group_key)

    assignments: dict[str, int] = {}
    fold_sizes = [0 for _ in range(folds)]
    class_fold_sizes: dict[int, list[int]] = {}

    for class_id, group_keys in sorted(groups_by_class.items()):
        rng.shuffle(group_keys)
        class_fold_sizes[class_id] = [0 for _ in range(folds)]

        for group_key in group_keys:
            target_fold = min(
                range(folds),
                key=lambda fold: (class_fold_sizes[class_id][fold], fold_sizes[fold], fold),
            )
            assignments[group_key] = target_fold
            group_size = len(groups[group_key])
            class_fold_sizes[class_id][target_fold] += group_size
            fold_sizes[target_fold] += group_size

    return assignments


def rewrite_data_yaml(source_dir: Path, fold_dir: Path) -> None:
    source_yaml = source_dir / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"Missing source data.yaml: {source_yaml}")

    lines = source_yaml.read_text().splitlines()
    rewritten = []
    saw_train = False
    saw_val = False
    for line in lines:
        if line.startswith("path:"):
            rewritten.append(f"path: {fold_dir.as_posix()}")
        elif line.startswith("train:"):
            rewritten.append("train: images/train")
            saw_train = True
        elif line.startswith("val:"):
            rewritten.append("val: images/val")
            saw_val = True
        else:
            rewritten.append(line)

    if not saw_train:
        rewritten.insert(1, "train: images/train")
    if not saw_val:
        rewritten.insert(2, "val: images/val")
    (fold_dir / "data.yaml").write_text("\n".join(rewritten) + "\n")


def copy_item(item: DatasetItem, target_split: str, fold_dir: Path) -> tuple[Path, Path]:
    image_dir = fold_dir / "images" / target_split
    label_dir = fold_dir / "labels" / target_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    output_image = image_dir / item.image_path.name
    output_label = label_dir / item.label_path.name
    shutil.copy2(item.image_path, output_image)
    shutil.copy2(item.label_path, output_label)
    return output_image, output_label


def write_fold_summary(
    output_dir: Path,
    rows: list[dict[str, str | int]],
) -> None:
    with (output_dir / "cross_validation_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fold",
                "target_split",
                "source_split",
                "group_key",
                "class_ids",
                "source_image",
                "source_label",
                "output_image",
                "output_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_fold_counts(output_dir: Path, rows: list[dict[str, str | int]]) -> None:
    counts: dict[tuple[int, str, int], int] = {}
    image_counts: dict[tuple[int, str], int] = {}

    for row in rows:
        fold = int(row["fold"])
        target_split = str(row["target_split"])
        image_counts[(fold, target_split)] = image_counts.get((fold, target_split), 0) + 1
        for class_id in str(row["class_ids"]).split():
            key = (fold, target_split, int(class_id))
            counts[key] = counts.get(key, 0) + 1

    with (output_dir / "cross_validation_counts.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["fold", "split", "class_id", "images", "instances"],
        )
        writer.writeheader()
        for fold, split, class_id in sorted(counts):
            writer.writerow(
                {
                    "fold": fold,
                    "split": split,
                    "class_id": class_id,
                    "images": image_counts[(fold, split)],
                    "instances": counts[(fold, split, class_id)],
                }
            )


def make_yolo_kfolds(
    source_dir: Path,
    output_dir: Path,
    folds: int,
    seed: int,
    overwrite: bool = False,
) -> list[dict[str, str | int]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Choose a new output directory or pass --overwrite."
            )
        shutil.rmtree(output_dir)

    items = collect_dataset_items(source_dir)
    groups = grouped_items(items)
    if len(groups) < folds:
        raise ValueError(f"Cannot create {folds} folds from only {len(groups)} grouped items.")

    assignments = assign_groups_to_folds(groups=groups, folds=folds, seed=seed)
    rows: list[dict[str, str | int]] = []

    for fold in range(folds):
        fold_dir = output_dir / f"fold_{fold + 1}"
        for split in ("train", "val"):
            (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        rewrite_data_yaml(source_dir=source_dir, fold_dir=fold_dir)

        for item in items:
            target_split = "val" if assignments[item.group_key] == fold else "train"
            output_image, output_label = copy_item(
                item=item,
                target_split=target_split,
                fold_dir=fold_dir,
            )
            rows.append(
                {
                    "fold": fold + 1,
                    "target_split": target_split,
                    "source_split": item.source_split,
                    "group_key": item.group_key,
                    "class_ids": " ".join(str(class_id) for class_id in item.class_ids),
                    "source_image": item.image_path,
                    "source_label": item.label_path,
                    "output_image": output_image,
                    "output_label": output_label,
                }
            )

    write_fold_summary(output_dir=output_dir, rows=rows)
    write_fold_counts(output_dir=output_dir, rows=rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create grouped, stratified k-fold YOLO datasets for cross-validation."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows = make_yolo_kfolds(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        folds=args.folds,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    val_count = sum(1 for row in rows if row["target_split"] == "val")
    print(f"Created {args.folds} folds under: {args.output_dir}")
    print(f"Total validation assignments across folds: {val_count}")
    print(f"Manifest: {args.output_dir / 'cross_validation_manifest.csv'}")
    print(f"Counts: {args.output_dir / 'cross_validation_counts.csv'}")


if __name__ == "__main__":
    main()

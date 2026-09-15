from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class YoloClass:
    class_id: int
    name: str


def parse_yolo_names(data_yaml: Path) -> list[YoloClass]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")

    names: list[YoloClass] = []
    in_names = False
    for raw_line in data_yaml.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key.isdigit():
                names.append(YoloClass(class_id=int(key), name=value.strip().strip("'\"")))
                continue
        if in_names and not raw_line.startswith((" ", "\t")):
            in_names = False

    if not names:
        raise ValueError(f"No class names found in {data_yaml}")
    return sorted(names, key=lambda item: item.class_id)


def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def read_label_lines(label_path: Path | None) -> list[str]:
    if label_path is None or not label_path.exists():
        return []
    return [
        line.strip()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]


def remap_target_lines(label_lines: list[str], target_class_id: int) -> list[str]:
    remapped: list[str] = []
    for line in label_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id == target_class_id:
            remapped.append(" ".join(["0", *parts[1:5]]))
    return remapped


def write_data_yaml(dataset_dir: Path, class_name: str) -> None:
    lines = [
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if (dataset_dir / "images" / "test").exists():
        lines.append("test: images/test")
    lines.extend(["names:", f"  0: {class_name}"])
    (dataset_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def copy_split_for_class(
    source_dir: Path,
    output_dir: Path,
    split: str,
    target_class: YoloClass,
) -> list[dict[str, str | int]]:
    source_image_dir = source_dir / "images" / split
    source_label_dir = source_dir / "labels" / split
    target_image_dir = output_dir / target_class.name / "images" / split
    target_label_dir = output_dir / target_class.name / "labels" / split

    rows: list[dict[str, str | int]] = []
    if not source_image_dir.exists():
        return rows

    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        image_path
        for image_path in sorted(source_image_dir.iterdir(), key=lambda path: path.name.lower())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image_path in image_paths:
        label_path = source_label_dir / f"{image_path.stem}.txt"
        label_lines = read_label_lines(label_path)
        target_lines = remap_target_lines(label_lines, target_class.class_id)

        output_image = target_image_dir / image_path.name
        output_label = target_label_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, output_image)
        output_label.write_text("\n".join(target_lines) + ("\n" if target_lines else ""))

        rows.append(
            {
                "target_class_id": target_class.class_id,
                "target_class_name": target_class.name,
                "split": split,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "output_image": str(output_image),
                "output_label": str(output_label),
                "target_instances": len(target_lines),
                "is_positive": int(bool(target_lines)),
            }
        )
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "target_class_id",
        "target_class_name",
        "split",
        "source_image",
        "source_label",
        "output_image",
        "output_label",
        "target_instances",
        "is_positive",
    ]
    with (output_dir / "one_vs_rest_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_counts(output_dir: Path, rows: list[dict[str, str | int]]) -> None:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row["target_class_name"]), str(row["split"]))
        bucket = counts.setdefault(key, {"images": 0, "positive_images": 0, "instances": 0})
        bucket["images"] += 1
        bucket["positive_images"] += int(row["is_positive"])
        bucket["instances"] += int(row["target_instances"])

    with (output_dir / "one_vs_rest_counts.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target_class_name",
                "split",
                "images",
                "positive_images",
                "negative_images",
                "instances",
            ],
        )
        writer.writeheader()
        for class_name, split in sorted(counts):
            bucket = counts[(class_name, split)]
            writer.writerow(
                {
                    "target_class_name": class_name,
                    "split": split,
                    "images": bucket["images"],
                    "positive_images": bucket["positive_images"],
                    "negative_images": bucket["images"] - bucket["positive_images"],
                    "instances": bucket["instances"],
                }
            )


def make_yolo_one_vs_rest_datasets(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, str | int]]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Choose a new output directory or pass --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    classes = parse_yolo_names(source_dir / "data.yaml")
    splits = [
        split
        for split in ("train", "val", "test")
        if (source_dir / "images" / split).exists()
    ]
    if not splits:
        raise ValueError(f"No YOLO image splits found under {source_dir / 'images'}")

    rows: list[dict[str, str | int]] = []
    for target_class in classes:
        for split in splits:
            rows.extend(
                copy_split_for_class(
                    source_dir=source_dir,
                    output_dir=output_dir,
                    split=split,
                    target_class=target_class,
                )
            )
        write_data_yaml(output_dir / target_class.name, target_class.name)

    write_manifest(output_dir, rows)
    write_counts(output_dir, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one-vs-rest YOLO datasets. Each class gets a single-class "
            "dataset with positives for that class and empty-label negatives "
            "from the other classes, preserving train/val/test splits."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = make_yolo_one_vs_rest_datasets(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Created one-vs-rest datasets under: {args.output_dir}")
    print(f"Manifest rows: {len(rows)}")
    print(f"Counts: {args.output_dir / 'one_vs_rest_counts.csv'}")


if __name__ == "__main__":
    main()

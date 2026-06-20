from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_image_for_label(image_dir: Path, label_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{label_path.stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def copy_split_to_train(source_dir: Path, output_dir: Path, split: str) -> int:
    source_image_dir = source_dir / "images" / split
    source_label_dir = source_dir / "labels" / split
    target_image_dir = output_dir / "images" / "train"
    target_label_dir = output_dir / "labels" / "train"
    copied = 0

    if not source_image_dir.exists() or not source_label_dir.exists():
        return copied

    for label_path in sorted(source_label_dir.glob("*.txt")):
        image_path = find_image_for_label(source_image_dir, label_path)
        if image_path is None:
            continue
        output_stem = f"{split}__{label_path.stem}"
        shutil.copy2(image_path, target_image_dir / f"{output_stem}{image_path.suffix.lower()}")
        shutil.copy2(label_path, target_label_dir / f"{output_stem}.txt")
        copied += 1

    return copied


def write_data_yaml(source_dir: Path, output_dir: Path) -> None:
    source_yaml = source_dir / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"Missing source data.yaml: {source_yaml}")

    lines = source_yaml.read_text().splitlines()
    rewritten: list[str] = []
    saw_train = False
    saw_val = False
    for line in lines:
        if line.startswith("path:"):
            rewritten.append(f"path: {output_dir.as_posix()}")
        elif line.startswith("train:"):
            rewritten.append("train: images/train")
            saw_train = True
        elif line.startswith("val:"):
            rewritten.append("val: images/train")
            saw_val = True
        else:
            rewritten.append(line)

    if not saw_train:
        rewritten.insert(1, "train: images/train")
    if not saw_val:
        rewritten.insert(2, "val: images/train")
    (output_dir / "data.yaml").write_text("\n".join(rewritten) + "\n")


def make_yolo_final_dataset(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Choose a new output directory or pass --overwrite."
            )
        shutil.rmtree(output_dir)

    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "labels" / "train").mkdir(parents=True)

    copied = 0
    copied += copy_split_to_train(source_dir=source_dir, output_dir=output_dir, split="train")
    copied += copy_split_to_train(source_dir=source_dir, output_dir=output_dir, split="val")
    write_data_yaml(source_dir=source_dir, output_dir=output_dir)
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a final YOLO training dataset using all labelled train/val images."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    copied = make_yolo_final_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Copied labelled images into final train split: {copied}")
    print(f"Output directory: {args.output_dir}")
    print("Note: val points to images/train for training sanity checks only.")


if __name__ == "__main__":
    main()

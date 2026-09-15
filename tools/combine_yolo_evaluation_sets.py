"""Combine YOLO test splits into a separate, collision-checked evaluation set."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def combine(sources: list[Path], output_dir: Path, class_name: str) -> int:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    image_output = output_dir / "images" / "test"
    label_output = output_dir / "labels" / "test"
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)

    count = 0
    for source in sources:
        for image in sorted((source / "images" / "test").iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = source / "labels" / "test" / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(label)
            destination_image = image_output / image.name
            destination_label = label_output / label.name
            if destination_image.exists() or destination_label.exists():
                raise FileExistsError(f"Duplicate evaluation filename: {image.name}")
            shutil.copy2(image, destination_image)
            shutil.copy2(label, destination_label)
            count += 1

    data_yaml = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/test",
            "val: images/test",
            "test: images/test",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", default="tables_desks")
    args = parser.parse_args()
    count = combine(args.source, args.output_dir, args.class_name)
    print(f"Combined {count} evaluation images in {args.output_dir}")


if __name__ == "__main__":
    main()

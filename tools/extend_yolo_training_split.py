"""Copy a YOLO dataset and add reviewed examples to its training split only."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def extend_dataset(
    base_dir: Path,
    reviewed_dir: Path,
    output_dir: Path,
    reviewed_class_id: int,
    target_class_id: int,
) -> int:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    shutil.copytree(base_dir, output_dir)

    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    protected_names = {
        path.name.lower()
        for split in ("val", "test")
        for path in (output_dir / "images" / split).glob("*")
        if path.is_file()
    }
    manifest_rows = []
    added = 0
    for image in sorted((reviewed_dir / "images" / "train").iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image.name.lower() in protected_names:
            raise ValueError(f"Reviewed image overlaps validation/test: {image.name}")
        output_image = train_images / image.name
        output_label = train_labels / f"{image.stem}.txt"
        if output_image.exists() or output_label.exists():
            raise FileExistsError(f"Training filename collision: {image.name}")

        source_label = reviewed_dir / "labels" / "train" / f"{image.stem}.txt"
        remapped = []
        for line in source_label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != reviewed_class_id:
                raise ValueError(f"Unexpected label in {source_label}: {line}")
            remapped.append(" ".join([str(target_class_id), *parts[1:]]))
        if not remapped:
            raise ValueError(f"Reviewed image has no boxes: {image}")

        shutil.copy2(image, output_image)
        output_label.write_text("\n".join(remapped) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "source_image": str(image),
                "source_label": str(source_label),
                "output_image": str(output_image),
                "output_label": str(output_label),
                "box_count": len(remapped),
            }
        )
        added += 1

    yaml_path = output_dir / "data.yaml"
    yaml_lines = yaml_path.read_text(encoding="utf-8").splitlines()
    yaml_lines = [
        f"path: {output_dir.as_posix()}" if line.startswith("path:") else line
        for line in yaml_lines
    ]
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    with (output_dir / "reviewed_additions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--reviewed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed-class-id", type=int, required=True)
    parser.add_argument("--target-class-id", type=int, default=0)
    args = parser.parse_args()
    added = extend_dataset(
        args.base_dir,
        args.reviewed_dir,
        args.output_dir,
        args.reviewed_class_id,
        args.target_class_id,
    )
    print(f"Added {added} reviewed training images to {args.output_dir}")


if __name__ == "__main__":
    main()

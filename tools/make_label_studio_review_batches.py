"""Create manageable Label Studio task batches for one grouped image class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_batches(
    input_dir: Path,
    output_dir: Path,
    document_root_name: str,
    batch_size: int,
) -> list[Path]:
    images = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for start in range(0, len(images), batch_size):
        tasks = []
        for sequence, image_path in enumerate(images[start:start + batch_size], start=start + 1):
            relative_path = image_path.relative_to(input_dir.parent).as_posix()
            local_file = quote(f"{document_root_name}/{relative_path}", safe="/")
            tasks.append(
                {
                    "data": {
                        "image": f"/data/local-files/?d={local_file}",
                        "file_name": image_path.name,
                        "review_number": sequence,
                        "source_group": input_dir.name,
                    }
                }
            )
        batch_number = start // batch_size + 1
        output_path = output_dir / f"batch_{batch_number:02d}.json"
        output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        outputs.append(output_path)

    index = {
        "image_count": len(images),
        "batch_size": batch_size,
        "batch_count": len(outputs),
        "batches": [path.name for path in outputs],
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--document-root-name",
        default="realsense_for_annotation_grouped_20260807",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    outputs = make_batches(
        args.input_dir,
        args.output_dir,
        args.document_root_name,
        args.batch_size,
    )
    print(f"Created {len(outputs)} batches in {args.output_dir}")


if __name__ == "__main__":
    main()

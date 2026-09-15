"""Render a folder of images as numbered contact sheets for easy local viewing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def render_sheets(input_dir: Path, output_dir: Path, batch_size: int = 50) -> int:
    images = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    columns, cell_width, cell_height = 5, 280, 230
    thumb_size = (250, 175)
    font = ImageFont.load_default()

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        rows = math.ceil(len(batch) / columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, image_path in enumerate(batch):
            sequence = start + offset + 1
            column, row = offset % columns, offset // columns
            x, y = column * cell_width, row * cell_height
            with Image.open(image_path) as source:
                preview = ImageOps.contain(source.convert("RGB"), thumb_size)
            image_x = x + (cell_width - preview.width) // 2
            sheet.paste(preview, (image_x, y + 8))
            draw.text((x + 10, y + 188), f"#{sequence}", fill="black", font=font)
            short_name = image_path.stem[:38]
            draw.text((x + 10, y + 204), short_name, fill="black", font=font)
        batch_number = start // batch_size + 1
        sheet.save(output_dir / f"tables_desks_batch_{batch_number:02d}.jpg", quality=90)
    return math.ceil(len(images) / batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    count = render_sheets(args.input_dir, args.output_dir, args.batch_size)
    print(f"Created {count} review sheets in {args.output_dir}")


if __name__ == "__main__":
    main()

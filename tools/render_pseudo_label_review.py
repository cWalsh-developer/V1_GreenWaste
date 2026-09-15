from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(
    manifest: Path,
    output: Path,
    samples_per_status: int = 2,
    statuses: list[str] | None = None,
    sample_seed: int | None = None,
) -> None:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        item_class = row.get("expected_class") or "tables_desks"
        grouped[item_class][row["status"]].append(row)

    tiles = []
    for item_class in sorted(grouped):
        selected = []
        sampler = random.Random(sample_seed)
        for status in statuses or ["accepted", "review"]:
            candidates = grouped[item_class][status]
            if sample_seed is not None:
                selected.extend(sampler.sample(candidates, min(samples_per_status, len(candidates))))
            else:
                selected.extend(candidates[:samples_per_status])
        for row in selected:
            source_path = row.get("source_path") or row["source_image"]
            image = Image.open(source_path).convert("RGB")
            image.thumbnail((280, 210))
            canvas = Image.new("RGB", (300, 270), "white")
            x_offset = (300 - image.width) // 2
            y_offset = 35 + (210 - image.height) // 2
            canvas.paste(image, (x_offset, y_offset))
            draw = ImageDraw.Draw(canvas)

            original = Image.open(source_path)
            scale_x = image.width / original.width
            scale_y = image.height / original.height
            original.close()

            box = None
            if row["bbox_xyxy"]:
                raw = row["bbox_xyxy"].strip("()[]").split(",")
                if len(raw) == 4:
                    box = tuple(float(value) for value in raw)
            if box is None:
                votes = json.loads(row.get("votes_json") or "[]")
                candidates = [vote for vote in votes if vote.get("bbox_xyxy")]
                if candidates:
                    box = tuple(max(candidates, key=lambda vote: vote["confidence"])["bbox_xyxy"])

            colour = "#16803a" if row["status"] == "accepted" else "#b42318"
            if box is not None:
                x1, y1, x2, y2 = box
                draw.rectangle(
                    (
                        x_offset + x1 * scale_x,
                        y_offset + y1 * scale_y,
                        x_offset + x2 * scale_x,
                        y_offset + y2 * scale_y,
                    ),
                    outline=colour,
                    width=3,
                )

            confidence = row["confidence"] or "n/a"
            draw.text((8, 8), f"{item_class} | {row['status']} | {confidence}", fill=colour)
            reason = row.get("reason") or "calibrated confidence threshold"
            draw.text((8, 248), reason[:44], fill="black")
            tiles.append(canvas)

    columns = 4
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 300, rows_count * 270), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 300, (index // columns) * 270))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-status", type=int, default=2)
    parser.add_argument("--status", action="append", choices=("accepted", "review"))
    parser.add_argument("--sample-seed", type=int)
    args = parser.parse_args()
    render(
        args.manifest,
        args.output,
        args.samples_per_status,
        args.status,
        args.sample_seed,
    )
    print(args.output)

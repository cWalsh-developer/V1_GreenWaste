"""Export pseudo-label manifest rows as Label Studio tasks with visible boxes."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image


def choose_box(row: dict[str, str]) -> tuple[list[float] | None, float]:
    if row.get("bbox_xyxy"):
        return list(ast.literal_eval(row["bbox_xyxy"])), float(row.get("confidence") or 0)

    votes = json.loads(row["votes_json"])
    expected = row["expected_class"]
    matching = [vote for vote in votes if vote["predicted_class"] == expected]
    candidates = matching or [vote for vote in votes if vote.get("bbox_xyxy")]
    if not candidates:
        return None, 0.0
    best = max(candidates, key=lambda vote: float(vote["confidence"]))
    return best["bbox_xyxy"], float(best["confidence"])


def export_tasks(
    manifest: Path,
    output: Path,
    document_root: Path,
    document_root_name: str,
    expected_class: str | None,
) -> int:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if expected_class:
        rows = [row for row in rows if row["expected_class"] == expected_class]

    tasks = []
    for number, row in enumerate(rows, start=1):
        source = Path(row["source_path"]).resolve()
        with Image.open(source) as image:
            width, height = image.size
        relative = source.relative_to(document_root.resolve()).as_posix()
        image_url = quote(f"{document_root_name}/{relative}", safe="/")
        box, score = choose_box(row)
        results = []
        if box:
            x1, y1, x2, y2 = box
            results.append(
                {
                    "id": f"proposal-{number}",
                    "from_name": "objects",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": 100 * x1 / width,
                        "y": 100 * y1 / height,
                        "width": 100 * (x2 - x1) / width,
                        "height": 100 * (y2 - y1) / height,
                        "rotation": 0,
                        "rectanglelabels": [row["expected_class"]],
                    },
                    "score": score,
                }
            )
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d={image_url}",
                    "review_number": number,
                    "pseudo_status": row["status"],
                    "pseudo_reason": row["reason"],
                },
                "predictions": [
                    {
                        "model_version": "five-fold-cv-ensemble",
                        "score": score,
                        "result": results,
                    }
                ],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--document-root",
        type=Path,
        default=Path("data/raw/realsense_for_annotation_grouped_20260807"),
    )
    parser.add_argument(
        "--document-root-name",
        default="realsense_for_annotation_grouped_20260807",
    )
    parser.add_argument("--expected-class")
    args = parser.parse_args()
    count = export_tasks(
        args.manifest,
        args.output,
        args.document_root,
        args.document_root_name,
        args.expected_class,
    )
    print(f"Exported {count} tasks with predictions to {args.output}")


if __name__ == "__main__":
    main()

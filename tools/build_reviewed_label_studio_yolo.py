"""Build a clean YOLO training package from audited Label Studio JSON tasks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from urllib.parse import unquote


ACCEPTED_DECISIONS = {"Correct table/desk", "Table/desk - box corrected"}


def build_dataset(
    export_json: Path,
    source_root: Path,
    output_dir: Path,
    class_id: int,
    split: str = "train",
    include_rejected_as_background: bool = False,
) -> dict[str, int]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(export_json.read_text(encoding="utf-8"))
    audit_rows = []
    accepted = rejected = boxes_written = 0
    for task in tasks:
        annotations = task.get("annotations") or []
        if not annotations:
            raise ValueError(f"Task {task.get('id')} has no submitted annotation")
        result = annotations[-1].get("result", [])
        choices = [
            item["value"]["choices"][0]
            for item in result
            if item.get("type") == "choices" and item.get("value", {}).get("choices")
        ]
        if len(choices) != 1:
            raise ValueError(f"Task {task.get('id')} does not have one review decision")
        decision = choices[0]

        image_url = unquote(task["data"]["image"])
        marker = "?d="
        if marker not in image_url:
            raise ValueError(f"Unsupported local image URL: {image_url}")
        relative = image_url.split(marker, 1)[1]
        root_name = source_root.name + "/"
        if relative.startswith(root_name):
            relative = relative[len(root_name):]
        source_image = (source_root / Path(relative)).resolve()
        source_image.relative_to(source_root.resolve())
        if not source_image.exists():
            raise FileNotFoundError(source_image)

        kept = decision in ACCEPTED_DECISIONS
        task_boxes = [item for item in result if item.get("type") == "rectanglelabels"]
        if kept:
            if not task_boxes:
                raise ValueError(f"Accepted task {task.get('id')} has no box")
            destination_image = image_dir / source_image.name
            destination_label = label_dir / f"{source_image.stem}.txt"
            lines = []
            for box in task_boxes:
                value = box["value"]
                x_center = (float(value["x"]) + float(value["width"]) / 2) / 100
                y_center = (float(value["y"]) + float(value["height"]) / 2) / 100
                width = float(value["width"]) / 100
                height = float(value["height"]) / 100
                values = (x_center, y_center, width, height)
                if any(number < 0 or number > 1 for number in values):
                    raise ValueError(f"Task {task.get('id')} has an out-of-range box")
                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
            shutil.copy2(source_image, destination_image)
            destination_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            accepted += 1
            boxes_written += len(lines)
        else:
            rejected += 1
            if include_rejected_as_background:
                shutil.copy2(source_image, image_dir / source_image.name)
                (label_dir / f"{source_image.stem}.txt").write_text("", encoding="utf-8")

        audit_rows.append(
            {
                "review_number": task["data"].get("review_number"),
                "source_image": str(source_image),
                "decision": decision,
                "included_in_training": kept,
                "box_count": len(task_boxes) if kept else 0,
            }
        )

    with (output_dir / "review_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "reviewed_tasks": len(tasks),
        "accepted_images": accepted,
        "rejected_images": rejected,
        "boxes_written": boxes_written,
        "yolo_class_id": class_id,
        "split": split,
        "background_images_included": rejected if include_rejected_as_background else 0,
    }
    (output_dir / "review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-json", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--include-rejected-as-background", action="store_true")
    args = parser.parse_args()
    summary = build_dataset(
        args.export_json,
        args.source_root,
        args.output_dir,
        args.class_id,
        args.split,
        args.include_rejected_as_background,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

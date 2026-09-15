"""Create a reproducible Label Studio review batch from YOLO predictions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from urllib.parse import quote, unquote

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def excluded_paths(export_paths: list[Path], source_root: Path) -> set[Path]:
    excluded: set[Path] = set()
    for export_path in export_paths:
        tasks = json.loads(export_path.read_text(encoding="utf-8"))
        for task in tasks:
            image_url = unquote(task["data"]["image"])
            relative = image_url.split("?d=", 1)[1]
            prefix = source_root.name + "/"
            if relative.startswith(prefix):
                relative = relative[len(prefix):]
            excluded.add((source_root / Path(relative)).resolve())
    return excluded


def make_batch(
    input_dir: Path,
    source_root: Path,
    model_path: Path,
    output_path: Path,
    exclude_exports: list[Path],
    sample_size: int,
    sample_seed: int,
    image_size: int,
    inference_confidence: float,
) -> dict[str, int | float]:
    excluded = excluded_paths(exclude_exports, source_root)
    candidates = sorted(
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve() not in excluded
    )
    if len(candidates) < sample_size:
        raise ValueError(f"Only {len(candidates)} unseen images remain")
    selected = random.Random(sample_seed).sample(candidates, sample_size)

    model = YOLO(str(model_path.resolve()))
    tasks = []
    boxes_proposed = 0
    for number, image_path in enumerate(selected, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        prediction = model.predict(
            source=str(image_path),
            conf=inference_confidence,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )[0]
        result_items = []
        score = 0.0
        if len(prediction.boxes):
            best_index = int(prediction.boxes.conf.argmax().item())
            box = prediction.boxes[best_index]
            score = float(box.conf.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            result_items.append(
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
                        "rectanglelabels": ["tables_desks"],
                    },
                    "score": score,
                }
            )
            boxes_proposed += 1

        relative = image_path.relative_to(source_root.resolve()).as_posix()
        image_url = quote(f"{source_root.name}/{relative}", safe="/")
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d={image_url}",
                    "review_number": number,
                    "source_file": image_path.name,
                },
                "predictions": [
                    {
                        "model_version": model_path.stem,
                        "score": score,
                        "result": result_items,
                    }
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    summary = {
        "sample_size": sample_size,
        "excluded_previous_images": len(excluded),
        "remaining_pool_before_sampling": len(candidates),
        "boxes_proposed": boxes_proposed,
        "images_without_prediction": sample_size - boxes_proposed,
        "sample_seed": sample_seed,
        "inference_confidence": inference_confidence,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-export", action="append", type=Path, default=[])
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20260829)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--inference-confidence", type=float, default=0.01)
    args = parser.parse_args()
    summary = make_batch(
        args.input_dir,
        args.source_root,
        args.model,
        args.output,
        args.exclude_export,
        args.sample_size,
        args.sample_seed,
        args.image_size,
        args.inference_confidence,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

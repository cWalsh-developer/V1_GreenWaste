from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .size_estimation import (
    estimate_size_from_roi,
    load_capture_images,
    parse_intrinsics,
    select_roi,
)


def load_metadata(capture_dir: Path) -> dict[str, Any]:
    metadata_path = capture_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text())


def build_manifest_row(
    capture_dir: Path,
    category: str,
    materials: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    metadata = load_metadata(capture_dir)
    rgb, depth = load_capture_images(capture_dir)
    intrinsics = parse_intrinsics(metadata)
    depth_scale = float(metadata.get("depth_scale", 0.001))

    roi = select_roi(rgb)
    estimate = estimate_size_from_roi(depth, roi, intrinsics, depth_scale)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        roi_path = output_dir / f"{capture_dir.name}_roi.json"
        roi_payload = {"capture_id": metadata.get("capture_id"), **asdict(estimate)}
        roi_path.write_text(json.dumps(roi_payload, indent=2))

    return {
        "retailer": "realsense",
        "product_name": metadata.get("capture_id"),
        "category": category,
        "materials": materials,
        "width_cm": estimate.width_cm,
        "depth_cm": estimate.depth_cm,
        "height_cm": estimate.height_cm,
        "distance_cm": estimate.distance_cm,
        "weight_kg": None,
        "sku": metadata.get("capture_id"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a capture manifest row")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--materials", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/realsense_manifest.csv"),
    )
    parser.add_argument(
        "--roi-output-dir",
        type=Path,
        default=Path("data/interim/realsense_roi"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    row = build_manifest_row(
        capture_dir=args.capture_dir,
        category=args.category,
        materials=args.materials,
        output_dir=args.roi_output_dir,
    )

    manifest_path = args.manifest
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)


if __name__ == "__main__":
    main()

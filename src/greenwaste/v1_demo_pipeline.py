from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .lca import DEFAULT_LCA_FACTOR_PATH, DEFAULT_MATERIAL_PROFILE_PATH
from .lca_estimation import run_lca_estimation
from .reference_matching import run_reference_matching
from .yolo_depth_size import estimate_capture_items, write_outputs


def run_v1_demo_pipeline(
    capture_dir: Path,
    model_path: Path,
    reference_path: Path,
    output_root: Path,
    confidence: float = 0.25,
    image_size: int = 960,
    max_detections: int = 1,
    top_n: int = 10,
    label_file: Path | None = None,
    label_item_class: str = "manual_roi",
    factor_path: Path = DEFAULT_LCA_FACTOR_PATH,
    material_profile_path: Path = DEFAULT_MATERIAL_PROFILE_PATH,
    use_material_profiles: bool = True,
    condition_status: str = "unknown",
) -> dict[str, Any]:
    capture_id = capture_dir.name
    size_output_dir = output_root / "yolo_size_estimates"
    reference_output_dir = output_root / "reference_matches"
    lca_output_dir = output_root / "lca_estimates"
    demo_output_dir = output_root / "v1_demo"

    size_rows = estimate_capture_items(
        capture_dir=capture_dir,
        model_path=model_path,
        confidence=confidence,
        image_size=image_size,
        max_detections=max_detections,
        label_file=label_file,
        label_item_class=label_item_class,
    )
    write_outputs(size_rows, size_output_dir, capture_id)
    size_csv = size_output_dir / f"{capture_id}_yolo_size.csv"

    reference_payloads = run_reference_matching(
        size_estimates_path=size_csv,
        reference_path=reference_path,
        output_dir=reference_output_dir,
        top_n=top_n,
    )
    match_summary_csv = (
        reference_output_dir / f"{capture_id}_reference_match_summary.csv"
    )

    lca_payloads = run_lca_estimation(
        match_summary_path=match_summary_csv,
        output_dir=lca_output_dir,
        factor_path=factor_path,
        material_profile_path=material_profile_path,
        use_material_profiles=use_material_profiles,
        condition_status=condition_status,
    )

    summary = {
        "capture_id": capture_id,
        "paths": {
            "size_csv": str(size_csv),
            "reference_match_summary_csv": str(match_summary_csv),
            "lca_csv": str(lca_output_dir / f"{capture_id}_lca_estimates.csv"),
            "lca_json": str(lca_output_dir / f"{capture_id}_lca_estimates.json"),
        },
        "size_estimates": size_rows,
        "reference_matches": reference_payloads,
        "lca_estimates": lca_payloads,
    }

    demo_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = demo_output_dir / f"{capture_id}_v1_demo_summary.json"
    summary["paths"]["demo_summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full saved-capture V1 pipeline on one capture folder."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/train-2/weights/best.pt"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/interim/ikea_reference_cleaned.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--max-detections", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--label-item-class", type=str, default="manual_roi")
    parser.add_argument("--factor-table", type=Path, default=DEFAULT_LCA_FACTOR_PATH)
    parser.add_argument(
        "--material-profiles",
        type=Path,
        default=DEFAULT_MATERIAL_PROFILE_PATH,
    )
    parser.add_argument(
        "--no-material-profiles",
        action="store_true",
        help="Use direct material_family factors instead of class proxy profiles.",
    )
    parser.add_argument(
        "--condition",
        choices=["not_reusable", "reusable", "unknown"],
        default="unknown",
        help="Manual visible-condition assessment for route recommendation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_v1_demo_pipeline(
        capture_dir=args.capture_dir,
        model_path=args.model,
        reference_path=args.reference,
        output_root=args.output_root,
        confidence=args.confidence,
        image_size=args.image_size,
        max_detections=args.max_detections,
        top_n=args.top_n,
        label_file=args.label_file,
        label_item_class=args.label_item_class,
        factor_path=args.factor_table,
        material_profile_path=args.material_profiles,
        use_material_profiles=not args.no_material_profiles,
        condition_status=args.condition,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

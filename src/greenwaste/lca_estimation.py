from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .lca import default_scenario_intensities, scenario_co2e_range


def build_lca_payload(
    match_row: pd.Series,
    scenario_intensities: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    scenario_intensities = scenario_intensities or default_scenario_intensities()
    weight_low = match_row.get("weight_low_kg")
    weight_high = match_row.get("weight_high_kg")

    if pd.isna(weight_low) or pd.isna(weight_high):
        scenarios = []
    else:
        weight_range = (float(weight_low), float(weight_high))
        scenarios = [
            {
                "scenario": scenario,
                "co2e_low_kg": co2e_low,
                "co2e_high_kg": co2e_high,
                "intensity_low_kgco2e_per_kg": intensity[0],
                "intensity_high_kgco2e_per_kg": intensity[1],
            }
            for scenario, intensity in scenario_intensities.items()
            for co2e_low, co2e_high in [scenario_co2e_range(weight_range, intensity)]
        ]

    return {
        "capture_id": match_row.get("capture_id"),
        "item_class": match_row.get("item_class"),
        "material_family": match_row.get("material_family", "unknown"),
        "weight_range_kg": [
            None if pd.isna(weight_low) else float(weight_low),
            None if pd.isna(weight_high) else float(weight_high),
        ],
        "scenarios": scenarios,
    }


def run_lca_estimation(
    match_summary_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    match_summary = pd.read_csv(match_summary_path)
    payloads = [
        build_lca_payload(match_row) for _, match_row in match_summary.iterrows()
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = match_summary_path.stem.replace("_reference_match_summary", "")
    json_path = output_dir / f"{stem}_lca_estimates.json"
    csv_path = output_dir / f"{stem}_lca_estimates.csv"

    json_path.write_text(json.dumps(payloads, indent=2))
    flat_rows = []
    for payload in payloads:
        for scenario in payload["scenarios"]:
            flat_rows.append(
                {
                    "capture_id": payload["capture_id"],
                    "item_class": payload["item_class"],
                    "material_family": payload["material_family"],
                    "weight_low_kg": payload["weight_range_kg"][0],
                    "weight_high_kg": payload["weight_range_kg"][1],
                    **scenario,
                }
            )
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate scenario CO2e ranges from reference match summaries"
    )
    parser.add_argument("--match-summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/lca_estimates"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payloads = run_lca_estimation(
        match_summary_path=args.match_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps(payloads, indent=2))


if __name__ == "__main__":
    main()

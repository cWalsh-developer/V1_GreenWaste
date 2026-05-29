from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .lca import (
    DEFAULT_LCA_FACTOR_PATH,
    default_scenario_intensities,
    load_lca_factor_table,
    scenario_co2e_range,
    select_lca_factors,
)


def build_lca_payload(
    match_row: pd.Series,
    scenario_intensities: dict[str, tuple[float, float]] | None = None,
    factor_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    weight_low = match_row.get("weight_low_kg")
    weight_high = match_row.get("weight_high_kg")

    if pd.isna(weight_low) or pd.isna(weight_high):
        scenarios = []
    elif factor_rows is not None:
        weight_range = (float(weight_low), float(weight_high))
        scenarios = []
        for _, factor in factor_rows.iterrows():
            intensity_range = (
                float(factor["factor_low_kgco2e_per_kg"]),
                float(factor["factor_high_kgco2e_per_kg"]),
            )
            co2e_low, co2e_high = scenario_co2e_range(weight_range, intensity_range)
            scenarios.append(
                {
                    "scenario": factor["scenario"],
                    "co2e_low_kg": co2e_low,
                    "co2e_high_kg": co2e_high,
                    "intensity_low_kgco2e_per_kg": intensity_range[0],
                    "intensity_high_kgco2e_per_kg": intensity_range[1],
                    "boundary": factor.get("boundary"),
                    "source_name": factor.get("source_name"),
                    "source_year": factor.get("source_year"),
                    "source_url": factor.get("source_url"),
                    "source_detail": factor.get("source_detail"),
                    "assumption_quality": factor.get("assumption_quality"),
                    "notes": factor.get("notes"),
                }
            )
    else:
        scenario_intensities = scenario_intensities or default_scenario_intensities()
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
    factor_path: Path = DEFAULT_LCA_FACTOR_PATH,
) -> list[dict[str, Any]]:
    match_summary = pd.read_csv(match_summary_path)
    factors = load_lca_factor_table(factor_path)
    payloads = [
        build_lca_payload(
            match_row,
            factor_rows=select_lca_factors(
                factors,
                str(match_row.get("material_family", "unknown")),
            ),
        )
        for _, match_row in match_summary.iterrows()
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
    parser.add_argument(
        "--factor-table",
        type=Path,
        default=DEFAULT_LCA_FACTOR_PATH,
        help="CSV table containing scenario factors, sources, and assumptions.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payloads = run_lca_estimation(
        match_summary_path=args.match_summary,
        output_dir=args.output_dir,
        factor_path=args.factor_table,
    )
    print(json.dumps(payloads, indent=2))


if __name__ == "__main__":
    main()

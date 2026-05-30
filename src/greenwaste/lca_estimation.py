from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .lca import (
    DEFAULT_LCA_FACTOR_PATH,
    DEFAULT_MATERIAL_PROFILE_PATH,
    build_composition_factor_rows,
    default_scenario_intensities,
    load_lca_factor_table,
    load_material_profile_table,
    scenario_co2e_range,
    select_material_profiles,
    select_lca_factors,
)


SCENARIO_LABELS = {
    "reuse_avoided_production": "Reuse",
    "closed_loop_recycling": "Recycle",
    "incineration_energy_recovery": "Incineration",
    "landfill": "Landfill",
}


REUSE_SCENARIOS = {"reuse_avoided_production"}
CONDITION_STATUSES = {"reusable", "not_reusable", "unknown"}


def _rank_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        scenarios,
        key=lambda scenario: (
            float(scenario["co2e_high_kg"]),
            float(scenario["co2e_low_kg"]),
        ),
    )


def _recommendation_payload(
    scenario: dict[str, Any] | None,
    rationale: str,
) -> dict[str, Any]:
    if scenario is None:
        return {
            "recommended_scenario": None,
            "recommended_route": None,
            "co2e_range_kg": [None, None],
            "rationale": rationale,
        }

    scenario_name = str(scenario["scenario"])
    route = SCENARIO_LABELS.get(
        scenario_name,
        scenario_name.replace("_", " ").title(),
    )
    return {
        "recommended_scenario": scenario_name,
        "recommended_route": route,
        "co2e_range_kg": [scenario["co2e_low_kg"], scenario["co2e_high_kg"]],
        "rationale": rationale,
    }


def build_scenario_recommendation(
    scenarios: list[dict[str, Any]],
    condition_status: str = "unknown",
) -> dict[str, Any]:
    condition_status = str(condition_status or "unknown").strip().lower()
    if condition_status not in CONDITION_STATUSES:
        raise ValueError(
            "condition_status must be one of: "
            + ", ".join(sorted(CONDITION_STATUSES))
        )

    if not scenarios:
        return {
            "condition_status": condition_status,
            "recommended_scenario": None,
            "recommended_route": None,
            "preferred_if_reusable": None,
            "recommended_if_not_reusable": None,
            "rationale": "No scenario result was available because the weight range could not be estimated.",
        }

    reuse_candidates = [
        scenario for scenario in scenarios if scenario["scenario"] in REUSE_SCENARIOS
    ]
    disposal_candidates = [
        scenario for scenario in scenarios if scenario["scenario"] not in REUSE_SCENARIOS
    ]
    best_reuse = _rank_scenarios(reuse_candidates)[0] if reuse_candidates else None
    best_disposal = (
        _rank_scenarios(disposal_candidates)[0] if disposal_candidates else None
    )

    preferred_if_reusable = _recommendation_payload(
        best_reuse,
        "Reuse is preferred when the item is manually assessed as reusable.",
    )
    recommended_if_not_reusable = _recommendation_payload(
        best_disposal,
        "If the item is not reusable, this is the lowest upper-bound CO2e route among the modelled disposal routes.",
    )

    if condition_status == "reusable" and best_reuse is not None:
        selected = preferred_if_reusable
        rationale = (
            "The item condition was marked reusable, so reuse is recommended. "
            "Reuse is modelled as an avoided-production route and should only "
            "be selected when the collector judges the item suitable for reuse."
        )
    elif condition_status == "not_reusable" and best_disposal is not None:
        selected = recommended_if_not_reusable
        rationale = (
            "The item condition was marked not reusable, so reuse is excluded. "
            f"{selected['recommended_route']} has the lowest upper-bound CO2e "
            "estimate among the remaining modelled disposal routes."
        )
    elif best_disposal is not None:
        selected = recommended_if_not_reusable
        rationale = (
            "Item condition was not provided. Reuse should be selected only if "
            "a collector manually assesses the item as reusable. Until then, "
            f"{selected['recommended_route']} is reported as the non-reuse "
            "disposal recommendation."
        )
    else:
        selected = preferred_if_reusable
        rationale = (
            "Only a reuse scenario was available. Manual condition assessment "
            "is required before selecting reuse."
        )

    return {
        "condition_status": condition_status,
        "recommended_scenario": selected["recommended_scenario"],
        "recommended_route": selected["recommended_route"],
        "co2e_range_kg": selected["co2e_range_kg"],
        "preferred_if_reusable": preferred_if_reusable,
        "recommended_if_not_reusable": recommended_if_not_reusable,
        "rationale": rationale,
    }


def build_lca_payload(
    match_row: pd.Series,
    scenario_intensities: dict[str, tuple[float, float]] | None = None,
    factor_rows: pd.DataFrame | None = None,
    composition_factor_rows: pd.DataFrame | None = None,
    condition_status: str = "unknown",
) -> dict[str, Any]:
    weight_low = match_row.get("weight_low_kg")
    weight_high = match_row.get("weight_high_kg")
    weight_source_summary = {
        "reference_count": int(match_row.get("weight_reference_count", 0) or 0),
        "imputed_count": int(match_row.get("weight_imputed_count", 0) or 0),
        "missing_count": int(match_row.get("weight_missing_count", 0) or 0),
        "note": match_row.get(
            "weight_source_note",
            "Weight source summary was not provided.",
        ),
    }

    if pd.isna(weight_low) or pd.isna(weight_high):
        scenarios = []
    elif composition_factor_rows is not None and not composition_factor_rows.empty:
        weight_range = (float(weight_low), float(weight_high))
        scenarios = []
        for scenario, scenario_rows in composition_factor_rows.groupby("scenario"):
            case_results = []
            all_lows = []
            all_highs = []
            for _, factor in scenario_rows.iterrows():
                intensity_range = (
                    float(factor["factor_low_kgco2e_per_kg"]),
                    float(factor["factor_high_kgco2e_per_kg"]),
                )
                co2e_low, co2e_high = scenario_co2e_range(
                    weight_range,
                    intensity_range,
                )
                all_lows.append(co2e_low)
                all_highs.append(co2e_high)
                case_results.append(
                    {
                        "profile_case": factor["profile_case"],
                        "material_fractions": factor["material_fractions"],
                        "intensity_low_kgco2e_per_kg": intensity_range[0],
                        "intensity_high_kgco2e_per_kg": intensity_range[1],
                        "co2e_low_kg": co2e_low,
                        "co2e_high_kg": co2e_high,
                    }
                )

            scenarios.append(
                {
                    "scenario": scenario,
                    "co2e_low_kg": min(all_lows),
                    "co2e_high_kg": max(all_highs),
                    "intensity_low_kgco2e_per_kg": min(
                        case["intensity_low_kgco2e_per_kg"]
                        for case in case_results
                    ),
                    "intensity_high_kgco2e_per_kg": max(
                        case["intensity_high_kgco2e_per_kg"]
                        for case in case_results
                    ),
                    "boundary": "class-based material composition proxy",
                    "source_name": " + ".join(
                        sorted(scenario_rows["source_name"].dropna().astype(str).unique())
                    ),
                    "source_year": " + ".join(
                        sorted(scenario_rows["source_year"].dropna().astype(str).unique())
                    ),
                    "source_url": " + ".join(
                        sorted(scenario_rows["source_url"].dropna().astype(str).unique())
                    ),
                    "source_detail": (
                        "Low/high result spans class-specific material composition "
                        "profile cases."
                    ),
                    "assumption_quality": "low",
                    "notes": (
                        "Material composition is not measured; proxy profile cases "
                        "are used to carry composition uncertainty."
                    ),
                    "composition_cases": case_results,
                }
            )
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
        "composition_profile": match_row.get(
            "composition_profile",
            match_row.get("item_class", "unknown"),
        ),
        "material_family": match_row.get("material_family", "unknown"),
        "weight_range_kg": [
            None if pd.isna(weight_low) else float(weight_low),
            None if pd.isna(weight_high) else float(weight_high),
        ],
        "weight_source_summary": weight_source_summary,
        "scenarios": scenarios,
        "recommendation": build_scenario_recommendation(
            scenarios,
            condition_status=condition_status,
        ),
    }


def run_lca_estimation(
    match_summary_path: Path,
    output_dir: Path,
    factor_path: Path = DEFAULT_LCA_FACTOR_PATH,
    material_profile_path: Path = DEFAULT_MATERIAL_PROFILE_PATH,
    use_material_profiles: bool = True,
    condition_status: str = "unknown",
) -> list[dict[str, Any]]:
    match_summary = pd.read_csv(match_summary_path)
    factors = load_lca_factor_table(factor_path)
    material_profiles = (
        load_material_profile_table(material_profile_path)
        if use_material_profiles
        else pd.DataFrame()
    )
    payloads = []
    for _, match_row in match_summary.iterrows():
        if use_material_profiles:
            composition_profile = match_row.get(
                "composition_profile",
                match_row.get("item_class", "unknown"),
            )
            selected_profiles = select_material_profiles(
                material_profiles,
                str(composition_profile),
            )
            composition_factors = build_composition_factor_rows(
                factors,
                selected_profiles,
            )
            payloads.append(
                build_lca_payload(
                    match_row,
                    composition_factor_rows=composition_factors,
                    condition_status=condition_status,
                )
            )
        else:
            payloads.append(
                build_lca_payload(
                    match_row,
                    factor_rows=select_lca_factors(
                        factors,
                        str(match_row.get("material_family", "unknown")),
                    ),
                    condition_status=condition_status,
                )
            )

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
                    "composition_profile": payload["composition_profile"],
                    "material_family": payload["material_family"],
                    "recommended_scenario": payload["recommendation"][
                        "recommended_scenario"
                    ],
                    "recommended_route": payload["recommendation"][
                        "recommended_route"
                    ],
                    "condition_status": payload["recommendation"][
                        "condition_status"
                    ],
                    "weight_low_kg": payload["weight_range_kg"][0],
                    "weight_high_kg": payload["weight_range_kg"][1],
                    "weight_reference_count": payload["weight_source_summary"][
                        "reference_count"
                    ],
                    "weight_imputed_count": payload["weight_source_summary"][
                        "imputed_count"
                    ],
                    "weight_missing_count": payload["weight_source_summary"][
                        "missing_count"
                    ],
                    "weight_source_note": payload["weight_source_summary"]["note"],
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
    parser.add_argument(
        "--material-profiles",
        type=Path,
        default=DEFAULT_MATERIAL_PROFILE_PATH,
        help="CSV table containing low/high material composition profiles.",
    )
    parser.add_argument(
        "--no-material-profiles",
        action="store_true",
        help="Use direct material_family factors instead of class proxy profiles.",
    )
    parser.add_argument(
        "--condition",
        choices=sorted(CONDITION_STATUSES),
        default="unknown",
        help="Manual condition assessment for recommendation logic.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payloads = run_lca_estimation(
        match_summary_path=args.match_summary,
        output_dir=args.output_dir,
        factor_path=args.factor_table,
        material_profile_path=args.material_profiles,
        use_material_profiles=not args.no_material_profiles,
        condition_status=args.condition,
    )
    print(json.dumps(payloads, indent=2))


if __name__ == "__main__":
    main()

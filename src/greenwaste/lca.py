from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_LCA_FACTOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "interim"
    / "mappings"
    / "lca_scenario_factors.csv"
)
DEFAULT_MATERIAL_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "interim"
    / "mappings"
    / "material_composition_profiles.csv"
)


def scenario_co2e_range(
    weight_range_kg: Tuple[float, float],
    intensity_range_kgco2e_per_kg: Tuple[float, float],
) -> Tuple[float, float]:
    low_weight, high_weight = weight_range_kg
    low_intensity, high_intensity = intensity_range_kgco2e_per_kg

    values = (
        low_weight * low_intensity,
        low_weight * high_intensity,
        high_weight * low_intensity,
        high_weight * high_intensity,
    )
    return (min(values), max(values))


def default_scenario_intensities() -> Dict[str, Tuple[float, float]]:
    # Legacy fallback used by older tests/callers. The main pipeline now reads
    # auditable factors from data/interim/mappings/lca_scenario_factors.csv.
    return {
        "reuse": (-0.4, -0.1),
        "recycling": (0.1, 0.5),
        "incineration": (0.4, 1.0),
        "landfill": (0.6, 1.4),
    }


def load_lca_factor_table(path: Path = DEFAULT_LCA_FACTOR_PATH) -> pd.DataFrame:
    factors = pd.read_csv(path)
    required_columns = {
        "material_family",
        "scenario",
        "factor_low_kgco2e_per_kg",
        "factor_high_kgco2e_per_kg",
        "boundary",
        "source_name",
        "source_year",
    }
    missing = required_columns.difference(factors.columns)
    if missing:
        raise ValueError(f"LCA factor table missing columns: {sorted(missing)}")

    factors["material_family"] = factors["material_family"].str.strip().str.lower()
    factors["scenario"] = factors["scenario"].str.strip()
    factors["factor_low_kgco2e_per_kg"] = pd.to_numeric(
        factors["factor_low_kgco2e_per_kg"], errors="coerce"
    )
    factors["factor_high_kgco2e_per_kg"] = pd.to_numeric(
        factors["factor_high_kgco2e_per_kg"], errors="coerce"
    )
    return factors.dropna(
        subset=["factor_low_kgco2e_per_kg", "factor_high_kgco2e_per_kg"]
    )


def select_lca_factors(
    factors: pd.DataFrame,
    material_family: str,
    fallback_material_family: str = "unknown",
) -> pd.DataFrame:
    material_key = str(material_family or "unknown").strip().lower()
    selected = factors[factors["material_family"] == material_key]
    if selected.empty and material_key != fallback_material_family:
        selected = factors[factors["material_family"] == fallback_material_family]
    return selected.copy()


def load_material_profile_table(
    path: Path = DEFAULT_MATERIAL_PROFILE_PATH,
) -> pd.DataFrame:
    profiles = pd.read_csv(path)
    required_columns = {
        "item_class",
        "profile_case",
        "material_family",
        "fraction",
    }
    missing = required_columns.difference(profiles.columns)
    if missing:
        raise ValueError(f"Material profile table missing columns: {sorted(missing)}")

    profiles["item_class"] = profiles["item_class"].str.strip().str.lower()
    profiles["profile_case"] = profiles["profile_case"].str.strip()
    profiles["material_family"] = profiles["material_family"].str.strip().str.lower()
    profiles["fraction"] = pd.to_numeric(profiles["fraction"], errors="coerce")
    profiles = profiles.dropna(subset=["fraction"])

    totals = profiles.groupby(["item_class", "profile_case"])["fraction"].sum()
    invalid = totals[(totals - 1.0).abs() > 0.001]
    if not invalid.empty:
        bad_profiles = [f"{item}/{case}" for item, case in invalid.index]
        raise ValueError(
            "Material profile fractions must sum to 1.0 for: "
            + ", ".join(bad_profiles)
        )
    return profiles


def select_material_profiles(
    profiles: pd.DataFrame,
    item_class: str,
    fallback_item_class: str = "unknown",
) -> pd.DataFrame:
    item_key = str(item_class or "unknown").strip().lower()
    selected = profiles[profiles["item_class"] == item_key]
    if selected.empty and item_key != fallback_item_class:
        selected = profiles[profiles["item_class"] == fallback_item_class]
    return selected.copy()


def build_composition_factor_rows(
    factors: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    scenario_names = sorted(factors["scenario"].unique())
    factor_lookup = factors.set_index(["material_family", "scenario"])

    for (item_class, profile_case), profile_group in profiles.groupby(
        ["item_class", "profile_case"]
    ):
        material_fractions = {
            row["material_family"]: float(row["fraction"])
            for _, row in profile_group.iterrows()
        }
        for scenario in scenario_names:
            weighted_low = 0.0
            weighted_high = 0.0
            factor_rows = []
            missing_materials = []

            for material, fraction in material_fractions.items():
                key = (material, scenario)
                if key not in factor_lookup.index:
                    missing_materials.append(material)
                    continue
                factor = factor_lookup.loc[key]
                weighted_low += fraction * float(factor["factor_low_kgco2e_per_kg"])
                weighted_high += fraction * float(factor["factor_high_kgco2e_per_kg"])
                factor_rows.append(factor)

            if missing_materials:
                raise ValueError(
                    f"Missing LCA factor(s) for scenario {scenario}: "
                    + ", ".join(missing_materials)
                )

            factor_df = pd.DataFrame(factor_rows)
            rows.append(
                {
                    "item_class": item_class,
                    "profile_case": profile_case,
                    "scenario": scenario,
                    "factor_low_kgco2e_per_kg": weighted_low,
                    "factor_high_kgco2e_per_kg": weighted_high,
                    "material_fractions": material_fractions,
                    "boundary": "class-based material composition proxy",
                    "source_name": " + ".join(
                        sorted(factor_df["source_name"].dropna().astype(str).unique())
                    ),
                    "source_year": " + ".join(
                        sorted(
                            factor_df["source_year"].dropna().astype(str).unique()
                        )
                    ),
                    "source_url": " + ".join(
                        sorted(factor_df["source_url"].dropna().astype(str).unique())
                    ),
                    "source_detail": (
                        f"{item_class}/{profile_case} weighted material profile"
                    ),
                    "assumption_quality": "low",
                    "notes": (
                        "Material composition is not measured; this is a proxy "
                        "scenario profile used for uncertainty analysis."
                    ),
                }
            )

    return pd.DataFrame(rows)

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

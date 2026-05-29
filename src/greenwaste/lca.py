from __future__ import annotations

from typing import Dict, Tuple


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
    # Placeholder ranges (kg CO2e per kg item). Update with real LCA factors.
    return {
        "reuse": (-0.4, -0.1),
        "recycling": (0.1, 0.5),
        "incineration": (0.4, 1.0),
        "landfill": (0.6, 1.4),
    }

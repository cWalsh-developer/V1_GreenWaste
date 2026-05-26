from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .lca import scenario_co2e_range
from .models import ItemEstimate, ScenarioResult


def build_item_estimate(
    item_label: str,
    size_category: str,
    material_family: str,
    weight_range_kg: Tuple[float, float],
) -> ItemEstimate:
    return ItemEstimate(
        item_label=item_label,
        size_category=size_category,
        material_family=material_family,
        weight_range_kg=weight_range_kg,
    )


def build_scenario_results(
    weight_range_kg: Tuple[float, float],
    scenario_intensities: Dict[str, Tuple[float, float]],
) -> List[ScenarioResult]:
    return [
        ScenarioResult(
            scenario_name=name,
            co2e_range_kg=scenario_co2e_range(weight_range_kg, intensity),
        )
        for name, intensity in scenario_intensities.items()
    ]

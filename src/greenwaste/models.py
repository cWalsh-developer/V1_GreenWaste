from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ItemEstimate:
    item_label: str
    size_category: str
    material_family: str
    weight_range_kg: Tuple[float, float]


@dataclass(frozen=True)
class ScenarioResult:
    scenario_name: str
    co2e_range_kg: Tuple[float, float]

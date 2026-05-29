import pandas as pd

from greenwaste.lca_estimation import (
    build_lca_payload,
    build_scenario_recommendation,
)


def test_build_scenario_recommendation_selects_lowest_upper_bound():
    recommendation = build_scenario_recommendation(
        [
            {"scenario": "landfill", "co2e_low_kg": 5.0, "co2e_high_kg": 12.0},
            {
                "scenario": "reuse_avoided_production",
                "co2e_low_kg": -20.0,
                "co2e_high_kg": -2.0,
            },
            {
                "scenario": "closed_loop_recycling",
                "co2e_low_kg": 0.1,
                "co2e_high_kg": 1.0,
            },
        ]
    )

    assert recommendation["recommended_scenario"] == "reuse_avoided_production"
    assert recommendation["recommended_route"] == "Reuse"


def test_build_lca_payload_uses_weight_range_for_each_scenario():
    row = pd.Series(
        {
            "capture_id": "capture_001",
            "item_class": "chair_seating",
            "material_family": "mixed",
            "weight_low_kg": 5.0,
            "weight_high_kg": 10.0,
        }
    )

    payload = build_lca_payload(
        row,
        scenario_intensities={"reuse": (-0.4, -0.1), "landfill": (0.6, 1.4)},
    )

    assert payload["weight_range_kg"] == [5.0, 10.0]
    assert payload["scenarios"][0]["scenario"] == "reuse"
    assert payload["scenarios"][0]["co2e_low_kg"] == -4.0
    assert payload["scenarios"][0]["co2e_high_kg"] == -0.5
    assert payload["scenarios"][1]["co2e_low_kg"] == 3.0
    assert payload["scenarios"][1]["co2e_high_kg"] == 14.0
    assert payload["recommendation"]["recommended_scenario"] == "reuse"


def test_build_lca_payload_includes_factor_metadata():
    row = pd.Series(
        {
            "capture_id": "capture_001",
            "item_class": "chair_seating",
            "material_family": "wood",
            "weight_low_kg": 10.0,
            "weight_high_kg": 20.0,
        }
    )
    factors = pd.DataFrame(
        {
            "material_family": ["wood"],
            "scenario": ["landfill"],
            "factor_low_kgco2e_per_kg": [0.9],
            "factor_high_kgco2e_per_kg": [1.0],
            "boundary": ["end-of-life waste disposal"],
            "source_name": ["UK Government GHG Conversion Factors"],
            "source_year": [2025],
            "source_url": ["https://example.com"],
            "source_detail": ["Wood landfill"],
            "assumption_quality": ["high"],
            "notes": ["test note"],
        }
    )

    payload = build_lca_payload(row, factor_rows=factors)

    scenario = payload["scenarios"][0]
    assert scenario["scenario"] == "landfill"
    assert scenario["co2e_low_kg"] == 9.0
    assert scenario["co2e_high_kg"] == 20.0
    assert scenario["source_year"] == 2025
    assert scenario["assumption_quality"] == "high"


def test_build_lca_payload_uses_composition_profile_cases():
    row = pd.Series(
        {
            "capture_id": "capture_001",
            "item_class": "chair_seating",
            "composition_profile": "office_chair",
            "material_family": "mixed",
            "weight_low_kg": 10.0,
            "weight_high_kg": 20.0,
        }
    )
    composition_factors = pd.DataFrame(
        {
            "item_class": ["chair_seating", "chair_seating"],
            "profile_case": ["low_impact", "high_impact"],
            "scenario": ["landfill", "landfill"],
            "factor_low_kgco2e_per_kg": [0.2, 0.6],
            "factor_high_kgco2e_per_kg": [0.3, 0.8],
            "material_fractions": [{"wood": 1.0}, {"metal": 1.0}],
            "source_name": ["source", "source"],
            "source_year": [2025, 2025],
            "source_url": ["url", "url"],
        }
    )

    payload = build_lca_payload(
        row,
        composition_factor_rows=composition_factors,
    )

    scenario = payload["scenarios"][0]
    assert payload["composition_profile"] == "office_chair"
    assert scenario["co2e_low_kg"] == 2.0
    assert scenario["co2e_high_kg"] == 16.0
    assert len(scenario["composition_cases"]) == 2

import pandas as pd

from greenwaste.lca_estimation import build_lca_payload


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

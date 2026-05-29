import pandas as pd

from greenwaste.lca import (
    scenario_co2e_range,
    select_lca_factors,
)


def test_scenario_co2e_range_multiplies_bounds():
    result = scenario_co2e_range((2.0, 4.0), (0.5, 1.5))
    assert result == (1.0, 6.0)


def test_scenario_co2e_range_handles_negative_intensities():
    result = scenario_co2e_range((2.0, 4.0), (-0.4, -0.1))
    assert result == (-1.6, -0.2)


def test_select_lca_factors_falls_back_to_unknown():
    factors = pd.DataFrame(
        {
            "material_family": ["unknown"],
            "scenario": ["landfill"],
            "factor_low_kgco2e_per_kg": [0.1],
            "factor_high_kgco2e_per_kg": [0.2],
        }
    )

    selected = select_lca_factors(factors, "not_mapped")

    assert selected["material_family"].tolist() == ["unknown"]

import pandas as pd

from greenwaste.lca import (
    build_composition_factor_rows,
    load_material_profile_table,
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


def test_build_composition_factor_rows_weights_material_fractions():
    factors = pd.DataFrame(
        {
            "material_family": ["wood", "metal"],
            "scenario": ["landfill", "landfill"],
            "factor_low_kgco2e_per_kg": [1.0, 3.0],
            "factor_high_kgco2e_per_kg": [1.0, 3.0],
            "source_name": ["source", "source"],
            "source_year": [2025, 2025],
            "source_url": ["url", "url"],
        }
    )
    profiles = pd.DataFrame(
        {
            "item_class": ["chair_seating", "chair_seating"],
            "profile_case": ["low_impact", "low_impact"],
            "material_family": ["wood", "metal"],
            "fraction": [0.75, 0.25],
        }
    )

    rows = build_composition_factor_rows(factors, profiles)

    assert rows.loc[0, "factor_low_kgco2e_per_kg"] == 1.5
    assert rows.loc[0, "material_fractions"] == {"wood": 0.75, "metal": 0.25}


def test_load_material_profile_table_validates_fraction_totals(tmp_path):
    path = tmp_path / "profiles.csv"
    path.write_text(
        "\n".join(
            [
                "item_class,profile_case,material_family,fraction",
                "chair_seating,low_impact,wood,0.6",
                "chair_seating,low_impact,metal,0.4",
            ]
        )
    )

    profiles = load_material_profile_table(path)

    assert profiles["fraction"].sum() == 1.0

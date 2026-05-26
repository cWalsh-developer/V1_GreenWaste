from greenwaste.lca import scenario_co2e_range


def test_scenario_co2e_range_multiplies_bounds():
    result = scenario_co2e_range((2.0, 4.0), (0.5, 1.5))
    assert result == (1.0, 6.0)

import pandas as pd

from greenwaste.reference_cleaning import (
    assign_bin,
    assign_item_label,
    assign_material_family,
    assign_weight_bin,
    build_missing_summary,
    estimate_weight_kg,
    extract_keywords,
    normalize_text,
)


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  Solid  Wood  ") == "solid wood"


def test_extract_keywords_splits_materials():
    assert extract_keywords("Solid wood, steel") == ["solid", "wood", "steel"]


def test_assign_material_family_single_match():
    mapping = {"wood": "wood", "steel": "metal"}
    assert assign_material_family(["wood"], mapping) == "wood"


def test_assign_material_family_mixed():
    mapping = {"wood": "wood", "steel": "metal"}
    assert assign_material_family(["wood", "steel"], mapping) == "mixed"


def test_assign_material_family_unknown():
    mapping = {"wood": "wood"}
    assert assign_material_family(["glass"], mapping) == "unknown"


def test_assign_bin_uses_upper_bound():
    bins = pd.DataFrame(
        {
            "size_bin": ["small", "large"],
            "min_cm": [0, 40],
            "max_cm": [40, None],
        }
    )
    assert assign_bin(10, bins, "size_bin") == "small"
    assert assign_bin(50, bins, "size_bin") == "large"


def test_assign_weight_bin_uses_upper_bound():
    bins = pd.DataFrame(
        {
            "weight_bin": ["light", "heavy"],
            "min_kg": [0, 15],
            "max_kg": [15, None],
        }
    )
    assert assign_weight_bin(4, bins, "weight_bin") == "light"
    assert assign_weight_bin(20, bins, "weight_bin") == "heavy"


def test_assign_item_label_fallback():
    mapping = {"chair": "chair"}
    assert assign_item_label("Office Chair", mapping) == "office_chair"


def test_build_missing_summary_counts():
    cleaned = pd.DataFrame(
        {
            "missing_materials": [True, False, True],
            "missing_weight": [False, False, True],
            "missing_dimensions": [True, True, False],
            "material_family": ["unknown", "wood", "unknown"],
            "item_label": ["unknown", "chair", "table"],
        }
    )

    summary = build_missing_summary(cleaned).set_index("field")
    assert summary.loc["missing_materials", "missing_count"] == 2
    assert summary.loc["missing_weight", "missing_count"] == 1
    assert summary.loc["missing_dimensions", "missing_count"] == 2
    assert summary.loc["unknown_material_family", "missing_count"] == 2
    assert summary.loc["unknown_item_label", "missing_count"] == 1


def test_estimate_weight_uses_existing_value():
    value, imputed = estimate_weight_kg(12.0, 50.0, "wood", "medium")
    assert value == 12.0
    assert imputed is False


def test_estimate_weight_uses_volume_and_material():
    value, imputed = estimate_weight_kg(None, 10.0, "wood", "medium")
    assert value == 2.0
    assert imputed is True


def test_estimate_weight_fallback_size_bin():
    value, imputed = estimate_weight_kg(None, None, "unknown", "large")
    assert value == 20.0
    assert imputed is True

import pandas as pd

from greenwaste.reference_matching import (
    dimension_similarity_score,
    filter_reference_by_item_class,
    match_reference_items,
    summarize_reference_matches,
)


def test_filter_reference_by_item_class_uses_broad_keywords():
    reference = pd.DataFrame(
        {
            "item_label": ["chair", "sofa", "storage_cabinet"],
            "category": ["dining chair", "three seat sofa", "wall cabinet"],
        }
    )

    filtered = filter_reference_by_item_class(reference, "chair_seating")

    assert filtered["item_label"].tolist() == ["chair"]


def test_dimension_similarity_prefers_closer_dimensions():
    close = dimension_similarity_score(80, 100, 30, 82, 98, 32)
    far = dimension_similarity_score(80, 100, 30, 200, 90, 90)

    assert close < far


def test_match_reference_items_sorts_by_similarity():
    reference = pd.DataFrame(
        {
            "record_id": ["near", "far"],
            "source": ["IKEA", "IKEA"],
            "item_label": ["chair_seating", "chair_seating"],
            "category": ["dining chair", "dining chair"],
            "material_family": ["wood", "metal"],
            "weight_kg_filled": [6.0, 20.0],
            "weight_imputed": [False, False],
            "width_cm": [82.0, 200.0],
            "depth_cm": [32.0, 90.0],
            "height_cm": [98.0, 90.0],
            "size_bin": ["large", "extra_large"],
            "weight_bin": ["medium", "heavy"],
            "url": ["", ""],
            "product_url": ["", ""],
        }
    )

    matches = match_reference_items(reference, "chair_seating", 80, 100, 30, top_n=2)

    assert matches.loc[0, "record_id"] == "near"


def test_summarize_reference_matches_returns_weight_range_and_material_mode():
    matches = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "material_family": ["wood", "wood", "metal"],
            "weight_kg_filled": [5.0, 10.0, 20.0],
            "size_bin": ["medium", "medium", "large"],
        }
    )

    summary = summarize_reference_matches(matches)

    assert summary["candidate_count"] == 3
    assert summary["material_family"] == "wood"
    assert summary["size_bin"] == "medium"
    assert summary["weight_range_kg"][0] > 5.0
    assert summary["weight_range_kg"][1] < 20.0

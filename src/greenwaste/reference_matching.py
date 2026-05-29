from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reference_cleaning import normalize_text


BROAD_CLASS_KEYWORDS = {
    "beds_mattresses": ["bed", "mattress", "crib", "daybed", "sleeper"],
    "chair_seating": ["chair", "armchair", "stool", "bench"],
    "sofa": ["sofa", "loveseat", "sectional", "chaise", "futon", "couch"],
    "storage": [
        "storage",
        "cabinet",
        "shelf",
        "shelving",
        "wardrobe",
        "dresser",
        "drawer",
        "bookcase",
        "sideboard",
        "cart",
        "shoe",
    ],
    "tables_desks": ["table", "desk", "dining", "coffee", "side", "console"],
}


BROAD_CLASS_EXCLUDE_KEYWORDS = {
    "beds_mattresses": ["cover", "pad", "protector", "sheet", "pillow", "blanket"],
    "chair_seating": ["cover", "cushion", "pad", "set", "storage"],
    "sofa": ["cover", "cushion", "pad"],
    "storage": ["box", "basket", "bin", "insert", "organizer"],
    "tables_desks": ["chair", "set", "cover", "runner"],
}


BROAD_CLASS_ITEM_LABEL_PATTERNS = {
    "beds_mattresses": [
        r"beds_mattresses_.*(?:_|^)beds?(?:_|$)",
        r"beds_mattresses_.*mattresses?",
        r"crib_mattresses",
    ],
    "chair_seating": [
        r"dining_furniture_dining_chairs",
        r"chairs_armchairs",
        r"bar_furniture_bar_stools_chairs",
        r"(?:^|_)stools?(?:_|$)",
        r"(?:^|_)benches?(?:_|$)",
        r"high_chairs",
        r"children_s_chairs",
        r"desk_desk_chairs_desk_chairs",
        r"desk_desk_chairs_conference_chairs",
        r"desk_desk_chairs_gaming_furniture_gaming_chairs",
        r"conference_chairs",
    ],
    "sofa": [
        r"sofas_armchairs_sofas",
        r"sleeper_sofas_sofa_beds",
        r"(?:^|_)chaise_lounges(?:_|$)",
    ],
    "storage": [
        r"storage_organization",
        r"storage_containers",
        r"cabinet",
        r"shelving",
        r"wardrobe",
        r"dressers",
        r"bookcases",
        r"tv_stands",
        r"sideboards",
        r"shoe_cabinets",
    ],
    "tables_desks": [
        r"accent_tables",
        r"dining_furniture_dining_tables",
        r"computer_desks",
        r"gaming_furniture_gaming_desks",
        r"(?:^|_)tables?(?:_|$)",
        r"(?:^|_)desks?(?:_|$)",
    ],
}


MATCH_COLUMNS = [
    "record_id",
    "source",
    "item_label",
    "category",
    "material_family",
    "weight_kg_filled",
    "weight_imputed",
    "width_cm",
    "depth_cm",
    "height_cm",
    "size_bin",
    "weight_bin",
    "similarity_score",
    "url",
    "product_url",
]


COMPOSITION_PROFILE_RULES = {
    "chair_seating": [
        (
            "office_chair",
            [
                r"\bdesk chair",
                r"\boffice chair",
                r"\bgaming chair",
                r"\bcomputer chair",
                r"\bswivel chair",
                r"desk_chairs",
                r"gaming_furniture_gaming_chairs",
            ],
        )
    ]
}


def _search_text(reference_df: pd.DataFrame) -> pd.Series:
    return (
        reference_df["item_label"].fillna("").astype(str)
        + " "
        + reference_df["category"].fillna("").astype(str)
    ).apply(normalize_text)


def _keyword_pattern(keywords: list[str]) -> str:
    return "|".join(rf"\b{re.escape(keyword)}s?\b" for keyword in keywords)


def infer_composition_profile(item_class: str, matches: pd.DataFrame) -> str:
    item_class = str(item_class or "unknown")
    rules = COMPOSITION_PROFILE_RULES.get(item_class, [])
    if matches.empty or not rules:
        return item_class

    search_text = _search_text(matches).str.cat(sep=" ")
    for profile, patterns in rules:
        if any(re.search(pattern, search_text) for pattern in patterns):
            return profile
    return item_class


def filter_reference_by_item_class(
    reference_df: pd.DataFrame,
    item_class: str,
) -> pd.DataFrame:
    item_labels = reference_df["item_label"].fillna("").astype(str).str.lower()
    label_patterns = BROAD_CLASS_ITEM_LABEL_PATTERNS.get(item_class, [])
    filtered = pd.DataFrame()
    if label_patterns:
        label_pattern = "|".join(label_patterns)
        filtered = reference_df[item_labels.str.contains(label_pattern, regex=True)]

    if filtered.empty:
        keywords = BROAD_CLASS_KEYWORDS.get(item_class, [item_class])
        search_text = _search_text(reference_df)
        pattern = _keyword_pattern(keywords)
        filtered = reference_df[search_text.str.contains(pattern, regex=True)]

    exclude_keywords = BROAD_CLASS_EXCLUDE_KEYWORDS.get(item_class, [])
    if exclude_keywords and not filtered.empty:
        exclude_pattern = _keyword_pattern(exclude_keywords)
        keep_mask = ~_search_text(filtered).str.contains(exclude_pattern, regex=True)
        filtered = filtered[keep_mask]
    if filtered.empty:
        return reference_df.copy()
    return filtered.copy()


def _dimension_vector(width_cm: float, height_cm: float, depth_cm: float) -> np.ndarray:
    values = np.array([width_cm, height_cm, depth_cm], dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("At least one positive dimension is required")
    return np.sort(values)[::-1]


def dimension_similarity_score(
    detected_width_cm: float,
    detected_height_cm: float,
    detected_depth_cm: float,
    reference_width_cm: float,
    reference_height_cm: float,
    reference_depth_cm: float,
) -> float:
    detected = _dimension_vector(
        detected_width_cm,
        detected_height_cm,
        detected_depth_cm,
    )
    reference = _dimension_vector(
        reference_width_cm,
        reference_height_cm,
        reference_depth_cm,
    )
    pair_count = min(len(detected), len(reference))
    detected = detected[:pair_count]
    reference = reference[:pair_count]

    return float(np.mean(np.abs(np.log((reference + 1.0) / (detected + 1.0)))))


def match_reference_items(
    reference_df: pd.DataFrame,
    item_class: str,
    width_cm: float,
    height_cm: float,
    depth_cm: float,
    top_n: int = 10,
) -> pd.DataFrame:
    candidates = filter_reference_by_item_class(reference_df, item_class)
    candidates = candidates.dropna(
        subset=["width_cm", "depth_cm", "height_cm", "weight_kg_filled"]
    ).copy()
    if candidates.empty:
        return candidates

    candidates["similarity_score"] = candidates.apply(
        lambda row: dimension_similarity_score(
            detected_width_cm=width_cm,
            detected_height_cm=height_cm,
            detected_depth_cm=depth_cm,
            reference_width_cm=row["width_cm"],
            reference_height_cm=row["height_cm"],
            reference_depth_cm=row["depth_cm"],
        ),
        axis=1,
    )

    return (
        candidates.sort_values(["similarity_score", "weight_imputed"])
        .head(top_n)
        .loc[:, [column for column in MATCH_COLUMNS if column in candidates.columns]]
        .reset_index(drop=True)
    )


def summarize_reference_matches(
    matches: pd.DataFrame,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
) -> dict[str, Any]:
    if matches.empty:
        return {
            "candidate_count": 0,
            "material_family": "unknown",
            "weight_range_kg": [None, None],
            "size_bin": "unknown",
        }

    weights = matches["weight_kg_filled"].dropna().astype(float)
    material_counts = matches.loc[
        matches["material_family"].fillna("unknown") != "unknown",
        "material_family",
    ].value_counts()
    size_counts = matches["size_bin"].fillna("unknown").value_counts()

    material_family = (
        str(material_counts.index[0]) if not material_counts.empty else "unknown"
    )
    size_bin = str(size_counts.index[0]) if not size_counts.empty else "unknown"
    low_weight = float(weights.quantile(low_quantile)) if not weights.empty else None
    high_weight = float(weights.quantile(high_quantile)) if not weights.empty else None

    return {
        "candidate_count": int(len(matches)),
        "material_family": material_family,
        "weight_range_kg": [low_weight, high_weight],
        "size_bin": size_bin,
        "top_record_ids": matches["record_id"].head(5).astype(str).tolist(),
    }


def build_match_payload(
    size_row: pd.Series,
    matches: pd.DataFrame,
) -> dict[str, Any]:
    summary = summarize_reference_matches(matches)
    item_class = size_row.get("item_class")
    return {
        "capture_id": size_row.get("capture_id"),
        "item_class": item_class,
        "composition_profile": infer_composition_profile(str(item_class), matches),
        "confidence": (
            None
            if pd.isna(size_row.get("confidence"))
            else float(size_row.get("confidence"))
        ),
        "size_cues": {
            "width_cm": float(size_row["width_cm"]),
            "height_cm": float(size_row["height_cm"]),
            "depth_cm": float(size_row["depth_cm"]),
            "distance_cm": float(size_row["distance_cm"]),
            "size_category": size_row.get("size_category"),
        },
        "reference_summary": summary,
        "matches": matches.head(10).to_dict(orient="records"),
    }


def run_reference_matching(
    size_estimates_path: Path,
    reference_path: Path,
    output_dir: Path,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    size_estimates = pd.read_csv(size_estimates_path)
    reference = pd.read_csv(reference_path)

    payloads = []
    for _, size_row in size_estimates.iterrows():
        matches = match_reference_items(
            reference_df=reference,
            item_class=str(size_row["item_class"]),
            width_cm=float(size_row["width_cm"]),
            height_cm=float(size_row["height_cm"]),
            depth_cm=float(size_row["depth_cm"]),
            top_n=top_n,
        )
        payloads.append(build_match_payload(size_row, matches))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = size_estimates_path.stem.replace("_yolo_size", "")
    json_path = output_dir / f"{stem}_reference_matches.json"
    csv_path = output_dir / f"{stem}_reference_match_summary.csv"

    json_path.write_text(json.dumps(payloads, indent=2))
    summary_rows = [
        {
            "capture_id": payload["capture_id"],
            "item_class": payload["item_class"],
            "composition_profile": payload["composition_profile"],
            "confidence": payload["confidence"],
            "width_cm": payload["size_cues"]["width_cm"],
            "height_cm": payload["size_cues"]["height_cm"],
            "depth_cm": payload["size_cues"]["depth_cm"],
            "size_category": payload["size_cues"]["size_category"],
            "candidate_count": payload["reference_summary"]["candidate_count"],
            "material_family": payload["reference_summary"]["material_family"],
            "weight_low_kg": payload["reference_summary"]["weight_range_kg"][0],
            "weight_high_kg": payload["reference_summary"]["weight_range_kg"][1],
            "reference_size_bin": payload["reference_summary"]["size_bin"],
        }
        for payload in payloads
    ]
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match YOLO/depth size estimates to reference products"
    )
    parser.add_argument("--size-estimates", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/interim/ikea_reference_cleaned.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/reference_matches"),
    )
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payloads = run_reference_matching(
        size_estimates_path=args.size_estimates,
        reference_path=args.reference,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    print(json.dumps(payloads, indent=2))


if __name__ == "__main__":
    main()

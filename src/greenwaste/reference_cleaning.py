from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

DENSITY_KG_PER_L = {
    "wood": 0.2,
    "metal": 0.5,
    "plastic": 0.1,
    "glass": 0.4,
    "textile": 0.05,
    "foam": 0.03,
    "paper": 0.1,
    "electronic": 0.3,
    "mixed": 0.2,
}

SIZE_BIN_WEIGHT_KG = {
    "small": 2.5,
    "medium": 8.0,
    "large": 20.0,
    "extra_large": 45.0,
}


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def load_mapping_csv(path: Path, key_col: str, value_col: str) -> dict[str, str]:
    table = pd.read_csv(path)
    mapping = {}
    for _, row in table.iterrows():
        key = normalize_text(str(row[key_col]))
        value = str(row[value_col]).strip().lower()
        if key:
            mapping[key] = value
    return mapping


def extract_keywords(text: str) -> list[str]:
    if not text:
        return []
    normalized = normalize_text(text)
    return re.findall(r"[a-z0-9]+", normalized)


def assign_material_family(
    keywords: Iterable[str],
    keyword_map: dict[str, str],
) -> str:
    families = {keyword_map[k] for k in keywords if k in keyword_map}
    if not families:
        return "unknown"
    if len(families) == 1:
        return next(iter(families))
    return "mixed"


def assign_bin(value: float | None, bins: pd.DataFrame, label_col: str) -> str:
    if value is None or pd.isna(value):
        return "unknown"

    for _, row in bins.iterrows():
        min_val = row.get("min_cm")
        max_val = row.get("max_cm")
        if pd.isna(min_val):
            continue
        if pd.isna(max_val) and value >= min_val:
            return str(row[label_col])
        if value >= min_val and value < max_val:
            return str(row[label_col])
    return "unknown"


def assign_weight_bin(value: float | None, bins: pd.DataFrame, label_col: str) -> str:
    if value is None or pd.isna(value):
        return "unknown"

    for _, row in bins.iterrows():
        min_val = row.get("min_kg")
        max_val = row.get("max_kg")
        if pd.isna(min_val):
            continue
        if pd.isna(max_val) and value >= min_val:
            return str(row[label_col])
        if value >= min_val and value < max_val:
            return str(row[label_col])
    return "unknown"


def assign_item_label(category: str, label_map: dict[str, str]) -> str:
    if not category:
        return "unknown"
    normalized = normalize_text(category)
    return label_map.get(normalized, normalized.replace(" ", "_"))


def estimate_weight_kg(
    weight_kg: float | None,
    volume_l: float | None,
    material_family: str,
    size_bin: str,
) -> tuple[float | None, bool]:
    if weight_kg is not None and not pd.isna(weight_kg):
        return float(weight_kg), False

    density = DENSITY_KG_PER_L.get(material_family)
    if volume_l is not None and not pd.isna(volume_l) and density is not None:
        return max(volume_l * density, 0.0), True

    fallback = SIZE_BIN_WEIGHT_KG.get(size_bin)
    if fallback is not None:
        return fallback, True

    return None, False


def clean_reference_df(
    raw_df: pd.DataFrame,
    mapping_dir: Path,
) -> pd.DataFrame:
    material_map = load_mapping_csv(
        mapping_dir / "material_family_map.csv", "keyword", "material_family"
    )
    label_map = load_mapping_csv(
        mapping_dir / "item_label_map.csv", "raw_category", "item_label"
    )
    size_bins = pd.read_csv(mapping_dir / "size_bins.csv")
    weight_bins = pd.read_csv(mapping_dir / "weight_bins.csv")

    df = raw_df.copy()

    for column in ("width_cm", "depth_cm", "height_cm", "weight_kg"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    dimension_max = df[["width_cm", "depth_cm", "height_cm"]].max(axis=1, skipna=True)
    materials_raw = df["materials"].fillna("")
    material_keywords = materials_raw.apply(extract_keywords)
    material_family = material_keywords.apply(
        lambda keys: assign_material_family(keys, material_map)
    )

    record_id = df["sku"].fillna("").astype(str)
    record_id = record_id.where(
        record_id.str.len() > 0, other=df.index.map(lambda idx: f"row-{idx}")
    )

    missing_dimensions = df[["width_cm", "depth_cm", "height_cm"]].isna().all(axis=1)
    missing_weight = df["weight_kg"].isna()
    missing_materials = materials_raw.str.strip().eq("")

    volume_l = df["width_cm"] * df["depth_cm"] * df["height_cm"] / 1000.0

    size_bin = dimension_max.apply(
        lambda value: assign_bin(value, size_bins, "size_bin")
    )

    weight_estimates = [
        estimate_weight_kg(weight, volume, family, size)
        for weight, volume, family, size in zip(
            df.get("weight_kg"),
            volume_l,
            material_family,
            size_bin,
            strict=False,
        )
    ]
    weight_kg_filled = pd.Series([item[0] for item in weight_estimates])
    weight_imputed = pd.Series([item[1] for item in weight_estimates])

    cleaned = pd.DataFrame(
        {
            "record_id": record_id,
            "source": df.get("retailer", "IKEA"),
            "item_label": df["category"]
            .fillna("")
            .apply(lambda c: assign_item_label(c, label_map)),
            "category": df["category"].fillna("").apply(normalize_text),
            "material_family": material_family,
            "material_keywords": material_keywords.apply(lambda items: ",".join(items)),
            "weight_kg": df.get("weight_kg"),
            "weight_kg_filled": weight_kg_filled,
            "weight_imputed": weight_imputed,
            "width_cm": df.get("width_cm"),
            "depth_cm": df.get("depth_cm"),
            "height_cm": df.get("height_cm"),
            "volume_l": volume_l,
            "size_bin": size_bin,
            "weight_bin": weight_kg_filled.apply(
                lambda value: assign_weight_bin(value, weight_bins, "weight_bin")
            ),
            "missing_dimensions": missing_dimensions,
            "missing_weight": missing_weight,
            "missing_materials": missing_materials,
            "url": df.get("url"),
            "product_url": df.get("product_url"),
        }
    )

    return cleaned


def build_missing_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    total = len(cleaned)
    weight_imputed = cleaned.get("weight_imputed", pd.Series(False, index=cleaned.index))
    weight_kg_filled = cleaned.get("weight_kg_filled", pd.Series(pd.NA, index=cleaned.index))
    summary = {
        "missing_materials": cleaned["missing_materials"].sum(),
        "missing_weight": cleaned["missing_weight"].sum(),
        "missing_dimensions": cleaned["missing_dimensions"].sum(),
        "unknown_material_family": (cleaned["material_family"] == "unknown").sum(),
        "unknown_item_label": (cleaned["item_label"] == "unknown").sum(),
        "imputed_weight": weight_imputed.sum(),
        "missing_weight_after_impute": weight_kg_filled.isna().sum(),
    }
    rows = []
    for field, count in summary.items():
        percent = 0.0 if total == 0 else (count / total) * 100
        rows.append(
            {"field": field, "missing_count": int(count), "missing_pct": percent}
        )
    return pd.DataFrame(rows)


def run_cleaning(
    input_path: Path, mapping_dir: Path, output_path: Path
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_excel(input_path)
    cleaned = clean_reference_df(raw_df, mapping_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    summary_path = output_path.with_name(f"{output_path.stem}_missing_summary.csv")
    summary = build_missing_summary(cleaned)
    summary.to_csv(summary_path, index=False)
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean IKEA reference dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    LOGGER.info("Cleaning reference data from %s", args.input)
    cleaned = run_cleaning(args.input, args.mapping_dir, args.output)
    LOGGER.info("Cleaned rows: %s", len(cleaned))
    LOGGER.info("Output written to %s", args.output)


if __name__ == "__main__":
    main()

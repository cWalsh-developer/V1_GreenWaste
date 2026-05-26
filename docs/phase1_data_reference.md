# Phase 1 - Data Cleaning and Reference Mapping

## Goal

Create a clean, consistent reference table from the IKEA dataset so the pipeline can map a detected item to:

- item label
- size bin
- material family
- weight range

This phase is focused on data quality and stable mappings, not model training.

## Inputs

- IKEA dataset (Excel)
- Controlled item label list (small and practical)
- Mapping tables for materials, size bins, and weight bins

## Outputs

- Cleaned reference table (CSV or Parquet)
- Mapping tables in data/interim/mappings/
- Data dictionary and cleaning spec (docs/)

## Reference schema (cleaned)

Required columns:

- record_id
- source
- item_label
- category
- subcategory
- material_family
- material_keywords
- weight_kg
- length_cm
- width_cm
- height_cm
- size_bin
- weight_bin

Optional columns:

- volume_l
- price
- url
- series
- notes

## Quality checks

- No negative weights or dimensions
- Dimensions and weight are numeric if present
- Material family must be in mapping list
- Size bin must be assigned when any dimension exists
- Weight bin must be assigned when weight exists

## Mapping sources

- data/interim/mappings/material_family_map.csv
- data/interim/mappings/size_bins.csv
- data/interim/mappings/weight_bins.csv
- data/interim/mappings/item_label_map.csv

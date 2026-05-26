# Cleaning and Mapping Spec

## 1) Normalize text

- Lowercase
- Strip punctuation
- Collapse multiple spaces
- Remove marketing adjectives

## 2) Standardize units

- Convert g to kg
- Convert mm to cm
- Convert combined dimension strings to separate L/W/H

## 3) Dimension parsing rules

- Support formats like "W x D x H" and "L x W x H"
- If only two dimensions exist, leave missing dimension as null
- If multiple dimension strings exist, keep the largest set

## 4) Weight parsing rules

- Use numeric value only
- If range is provided, keep low/high and use midpoint for weight_kg
- If missing, leave null

## 5) Category normalization

- Map raw categories to controlled item labels
- Keep an explicit map in item_label_map.csv

## 6) Material extraction

- Split material strings into keywords
- Map keywords to a material_family
- If multiple families appear, assign "mixed"

## 7) Derived fields

- volume*l = (length_cm * width*cm * height_cm) / 1000
- size_bin from size_bins.csv using largest dimension
- weight_bin from weight_bins.csv using weight_kg

## 7a) Missing value handling

- If weight_kg is missing and dimensions exist, estimate weight using volume and a
  conservative material density factor.
- If weight_kg is still missing, use a size_bin fallback weight.
- Keep original weight_kg and store filled values in weight_kg_filled with a
  weight_imputed flag.

## 8) Validation

- Fail records with negative values
- Flag outliers for review
- Ensure material_family in allowed list
- Ensure item_label in allowed list

## 9) Missing data reporting

The cleaning step produces a summary file alongside the cleaned output:

- `{output_stem}_missing_summary.csv`

Fields reported:

- missing_materials
- missing_weight
- missing_dimensions
- unknown_material_family
- unknown_item_label
- imputed_weight
- missing_weight_after_impute

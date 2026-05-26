# Data Dictionary (IKEA Reference)

## Raw columns (expected)

These are common IKEA-style fields. Actual columns may vary and will be aligned during cleaning.

- product_name
- category
- subcategory
- dimensions
- weight
- material
- series
- price
- url
- country

## Cleaned columns

- record_id: stable unique id (string)
- source: data source name, e.g. IKEA (string)
- item_label: canonical label for model output (string)
- category: normalized category (string)
- subcategory: normalized subcategory (string, optional)
- material_family: mapped family (string)
- material_keywords: extracted keywords (string list or comma-delimited)
- weight_kg: item weight in kg (float, optional)
- length_cm: length in cm (float, optional)
- width_cm: width in cm (float, optional)
- height_cm: height in cm (float, optional)
- volume_l: derived volume in liters (float, optional)
- size_bin: size category (string)
- weight_bin: weight category (string)
- price: numeric price (float, optional)
- url: product url (string, optional)
- series: product series name (string, optional)
- notes: freeform notes (string, optional)

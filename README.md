# Green Waste AI System

Decision-support prototype for green waste and bulky items. The system combines RGB/depth sensing with lightweight classification, rough size cues, and reference product data (IKEA-style materials and weight ranges) to generate scenario-based CO2e guidance. Outputs are indicative ranges intended for routing and reuse decisions rather than precise LCA.

## Scope and intent

The goal is a practical, field-ready pipeline that works with limited hardware and controlled item categories:

- Capture RGB and depth data for bulky items.
- Classify items into a small, controlled label set.
- Derive rough size cues from depth (not full 3D reconstruction).
- Match items to reference materials and weight ranges.
- Produce scenario comparisons (reuse, recycling, incineration, landfill) with CO2e ranges.

## Example output

- Detected item: chair
- Size category: medium
- Material family: wood-dominant / mixed
- Estimated weight: 5-8 kg
- Scenario comparison: reuse, recycling, incineration, landfill
- CO2e output: indicative range, not exact value

## Hardware context

- Intel RealSense D415 depth camera (RGB + depth, scale cues)
- Stereolabs ZED 2i depth camera (higher fidelity depth, wider scene capture)
- NVIDIA Jetson Orin Nano Developer Kit (edge inference)
- Portable power banks (field testing)

## Project layout

- `data/raw/`: original inputs (xlsx, images, sensor dumps)
- `data/interim/`: cleaned and merged outputs
- `data/processed/`: features ready for modeling
- `notebooks/`: exploratory notebooks (created after initial scaffold)
- `src/greenwaste/`: reusable Python modules
- `tests/`: unit tests
- `docs/`: project notes

## Full V1 saved-capture demo

Run the complete V1 chain on one saved RealSense capture:

```powershell
$env:PYTHONPATH = "src"
python -m greenwaste.v1_demo_pipeline `
  --capture-dir data/raw/realsense/capture_20260527_022745 `
  --model runs/detect/train-2/weights/best.pt `
  --condition unknown
```

For a manually labelled ROI, add:

```powershell
  --label-file labels_realsense_chair_2026-05-28-04-41-30/rgb.txt `
  --label-item-class chair_seating
```

This writes size estimates, reference matches, LCA scenario ranges, and the
recommended route under `data/interim/`.

Use `--condition reusable` when the collector manually judges the item suitable
for reuse. Use `--condition not_reusable` when it is visibly damaged,
contaminated, unsafe, or otherwise unsuitable for reuse.

## Data sources (local)

Holds csv reference data for comparison of an IKEA dataset

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
- NVIDIA Jetson TX2 (edge inference)
- Portable power banks (field testing)

## Project layout

- `data/raw/`: original inputs (xlsx, images, sensor dumps)
- `data/interim/`: cleaned and merged outputs
- `data/processed/`: features ready for modeling
- `notebooks/`: exploratory notebooks (created after initial scaffold)
- `src/greenwaste/`: reusable Python modules
- `tests/`: unit tests
- `docs/`: project notes

## Data sources (local)

Holds csv reference data for comparison of an IKEA dataset

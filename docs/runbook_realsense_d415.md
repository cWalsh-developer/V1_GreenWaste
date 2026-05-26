# RealSense D415 Desktop Runbook (Phase 1)

## Goal

Capture a single RGB + depth frame per item and record metadata so the data can feed the reference mapping pipeline.

## Dependencies (desktop)

- Intel RealSense SDK
- pyrealsense2
- opencv-python (for visualization and ROI checks)

## Capture outputs

Store one folder per capture:

- rgb.png
- depth.png
- metadata.json

## Metadata fields

- capture_id
- timestamp_utc
- camera_model
- resolution_rgb
- resolution_depth
- depth_scale
- intrinsics
- notes

## Notes

- Keep a fixed camera distance per session when possible.
- Capture a neutral background to reduce segmentation noise.
- Use a simple bounding box or manual ROI for size estimation in early tests.

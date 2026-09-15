# GreenWaste Chosen Detector Weights

This folder contains the chosen YOLO detector weights for the GreenWaste proof-of-concept pipeline.

## Files

- `best.pt`: chosen YOLO detector checkpoint.
- `model_metadata.json`: source run, training context, metrics, and checksum.

## Intended Use

Use this model for the saved-capture and live RealSense prototype demos:

```powershell
$env:PYTHONPATH = "src"
.\myenv\Scripts\python.exe -m greenwaste.v1_demo_pipeline `
  --capture-dir data\raw\realsense\capture_20260527_022745 `
  --model models\greenwaste_detector_chosen\best.pt `
  --condition unknown
```

The model was selected as the final fitted prototype checkpoint. The most defensible detector evaluation remains the original stratified five-fold cross-validation result reported in the dissertation, because this final model was trained using all labelled/augmented data and its internal validation metrics are training diagnostics rather than an unbiased test result.

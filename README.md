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

## Live RealSense demo

Run the detector directly on a live Intel RealSense feed and display the
detected item, approximate size, reference weight range, recommended route, and
CO2e range on screen:

```powershell
$env:PYTHONPATH = "src"
.\myenv\Scripts\python.exe -m greenwaste.live_realsense_demo `
  --model "runs\detect\train_final_all_labelled_augmented_70ep_20260620\weights\best.pt" `
  --confidence 0.25 `
  --image-size 960 `
  --condition unknown `
  --color-format bgr
```

The live window supports:

- `q` or `Esc`: quit
- `s`: save the current annotated screen and route JSON to `data/interim/live_realsense_demo`
- `u`: set item condition to unknown
- `r`: set item condition to reusable
- `n`: set item condition to not reusable

The route recommendation is refreshed once per second by default. Increase
`--route-update-seconds` if the display feels sluggish, or lower it if you want
route outputs to update more frequently. The displayed CO2e values are
indicative decision-support ranges, not product-specific carbon accounting.

## Domain augmentation experiment

The stratified YOLO dataset can be copied into an augmented training dataset
without changing the validation split. This is useful for testing whether the
model is overfitting to clean product images with plain backgrounds.

```powershell
$env:PYTHONPATH = "src"
.\myenv\Scripts\python.exe -m greenwaste.augment_yolo_dataset `
  --source-dir data\processed\yolo_combined_stratified_20260616 `
  --output-dir data\processed\yolo_combined_stratified_augmented_20260620 `
  --variants-per-image 2 `
  --include-class-ids 0,3
```

Class `0` is beds/mattresses and class `3` is storage. Omit
`--include-class-ids` to augment every training class.

Train a comparison model against the augmented dataset:

```powershell
.\myenv\Scripts\yolo.exe detect train `
  model=yolo26n.pt `
  data=data\processed\yolo_combined_stratified_augmented_20260620\data.yaml `
  imgsz=960 `
  epochs=80 `
  batch=8 `
  name=train_stratified_augmented_20260620
```

Then test the same RealSense capture with the new model:

```powershell
.\myenv\Scripts\python.exe -m greenwaste.yolo_depth_size `
  --capture-dir data\raw\realsense\capture_20260618_222719 `
  --rgb-image-name rgb_bgr.png `
  --model runs\detect\train_stratified_augmented_20260620\weights\best.pt `
  --confidence 0.25
```

## YOLO cross-validation experiment

Create grouped, stratified folds from the non-augmented stratified dataset:

```powershell
$env:PYTHONPATH = "src"
.\myenv\Scripts\python.exe -m greenwaste.make_yolo_kfolds `
  --source-dir data\processed\yolo_combined_stratified_20260616 `
  --output-dir data\processed\yolo_combined_stratified_5fold_20260620 `
  --folds 5 `
  --seed 42
```

Train one model per fold:

```powershell
.\myenv\Scripts\yolo.exe detect train `
  model=yolo26n.pt `
  data=data\processed\yolo_combined_stratified_5fold_20260620\fold_1\data.yaml `
  imgsz=960 `
  epochs=80 `
  batch=8 `
  name=cv_stratified_fold1
```

Repeat the command for `fold_2` to `fold_5`, changing both the `data=...`
folder and `name=...`.

Summarise the cross-validation metrics:

```powershell
.\myenv\Scripts\python.exe -m greenwaste.summarize_yolo_cv_results `
  --results-glob "runs/detect/cv_stratified_fold*/results.csv" `
  --output-dir data\processed\yolo_cv_summary_20260620
```

The summary reports mean and standard deviation across folds for precision,
recall, mAP50, and mAP50-95.

## Final all-labelled YOLO model

After cross-validation has been used for evaluation, train one final deployment
model using all labelled images:

```powershell
$env:PYTHONPATH = "src"
.\myenv\Scripts\python.exe -m greenwaste.make_yolo_final_dataset `
  --source-dir data\processed\yolo_combined_stratified_20260616 `
  --output-dir data\processed\yolo_combined_all_labelled_20260620
```

Train the final detector:

```powershell
.\myenv\Scripts\yolo.exe detect train `
  model=yolo26n.pt `
  data=data\processed\yolo_combined_all_labelled_20260620\data.yaml `
  imgsz=960 `
  epochs=80 `
  batch=8 `
  name=train_final_all_labelled_20260620
```

The validation numbers from this final run are only training sanity checks,
because the same labelled images are used for training. Report the 5-fold
cross-validation metrics as the unbiased internal evaluation.

## Data sources (local)

Holds csv reference data for comparison of an IKEA dataset

## Label Studio local files

Start Label Studio with local-file serving enabled and restricted to `data/raw`:

```powershell
.\scripts\start-label-studio.ps1
```

In **Settings > Cloud Storage > Add Source Storage > Local files**, configure:

- **Absolute local path**:
  `D:\Green Waste\V1_GreenWaste\data\raw\realsense_for_annotation_grouped_20260807`
- **Import method**: `Tasks`
- **Scan all sub-folders** (recursive scan): enabled
- **File filter regex**: `.*\.(jpg|jpeg|png)$`

The `Tasks` import method is required for images. The alternative import method
expects task definitions in JSON, JSONL, or Parquet format and will reject JPG
files. Recursive scanning is required because the images are inside category
subdirectories.

The resulting image task values use local-file URLs relative to the configured
document root, for example:

```text
/data/local-files/?d=realsense_for_annotation_grouped_20260807/beds_mattresses/example.jpg
```

The launcher sets `LOCAL_FILES_SERVING_ENABLED=true` only for the Label Studio
process and sets `LOCAL_FILES_DOCUMENT_ROOT` to the resolved `data/raw` path.

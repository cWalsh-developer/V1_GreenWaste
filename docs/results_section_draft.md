# 4. Results

This section presents the outputs of the prototype pipeline, the item recognition and size estimation results, and the downstream reference matching and CO2e scenario estimates. The results are separated into internal detector evaluation, which uses the labelled dataset and cross-validation, and external RealSense/manual testing, which is treated as a qualitative domain-shift assessment.

Suggested figures:

- Figure 4.1: `docs/results_assets/capture_20260527_022745_annotated_detection.png`
- Figure 4.2: `docs/results_assets/cv_metric_summary.png`
- Figure 4.3: `docs/results_assets/capture_20260618_222719_annotated_detection.png`
- Figure 4.4: `docs/results_assets/capture_20260618_222719_lca_scenarios.png`
- Figure 4.5: `docs/results_assets/capture_20260618_232017_lca_scenarios.png`

## 4.1. Prototype Pipeline Output

The prototype was tested as an end-to-end RGB-D decision-support pipeline. A saved Intel RealSense capture was used as input, containing an RGB image, aligned depth image, and metadata file with camera intrinsics and depth scale. The pipeline then applied object detection, converted the detected bounding box into a depth region of interest, estimated physical dimensions, matched the detected item against the reference product dataset, and generated CO2e scenario outputs.

Figure 4.1 shows an example RealSense chair capture. The detector classified the object as `chair_seating` with a confidence score of 0.837. The associated depth frame was used to estimate an approximate size of 149.3 cm width, 112.5 cm height, and 59.0 cm depth, placing the item in the large size category. The result demonstrates that the prototype can connect RGB object recognition with depth-based size estimation and downstream decision-support outputs.

For the storage example shown in Figure 4.3, the detector classified the item as `storage` with a confidence score of 0.341. The estimated dimensions were 83.6 cm width, 73.2 cm height, and 12.8 cm depth, placing the object in the medium size category. Although the confidence was lower than the chair example, the pipeline still produced a complete output, including reference matching, estimated weight range, and LCA scenario values.

The full pipeline output consists of the following artefacts:

| Output stage | Example output | Purpose |
|---|---:|---|
| RGB-D capture | `rgb.png`, `depth.png`, `metadata.json` | Stores colour, depth, and camera calibration data |
| YOLO detection and size estimation | `*_yolo_size.csv` / `.json` | Stores predicted class, confidence, bounding box, and estimated dimensions |
| Reference matching | `*_reference_match_summary.csv` / `.json` | Stores matched reference products and estimated weight range |
| CO2e scenario estimation | `*_lca_estimates.csv` / `.json` | Stores scenario-level CO2e ranges and route recommendation |
| V1 summary | `*_v1_demo_summary.json` | Combines pipeline outputs for a single capture |

This confirms that the prototype is operational as an integrated proof-of-concept. However, the quality of later stages depends strongly on the correctness of the object detector. If the item class is misidentified, the reference matching and LCA estimate can be routed through the wrong product category.

## 4.2. Item Recognition and Size Estimation Results

The final labelled dataset contained 420 manually or pseudo-labelled images across five broad furniture categories. The class distribution was:

| Class | Labelled instances |
|---|---:|
| Beds and mattresses | 79 |
| Chair seating | 74 |
| Sofa | 114 |
| Storage | 93 |
| Tables and desks | 61 |

To provide a more defensible estimate of internal detector performance, five-fold cross-validation was used. Each fold contained 336 training images and 84 validation images, with stratification used to preserve class balance. Across the five folds, every labelled image was used for validation exactly once.

The cross-validation results are summarised in Figure 4.2. Using the best checkpoint from each fold, the detector achieved:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Precision | 0.908 | 0.014 |
| Recall | 0.868 | 0.070 |
| mAP50 | 0.943 | 0.030 |
| mAP50-95 | 0.839 | 0.033 |

The main metric used for reporting detector performance was mAP50-95, as it evaluates both object classification and bounding-box localisation across stricter IoU thresholds. The mean cross-validation mAP50-95 of 0.839 indicates good internal performance on the available labelled dataset. However, the fold-level results also showed variability, with best mAP50-95 values ranging from 0.804 to 0.879. This suggests that model performance is still sensitive to the relatively small dataset size.

After cross-validation, a final deployment model was trained using all 420 labelled images. A further final experiment used synthetic augmentation, producing 840 augmented copies in addition to the 420 original images, for a total of 1,260 training images. This final augmented run achieved high training-set-as-validation metrics, with mAP50-95 reaching 0.949. This value should not be interpreted as an unbiased evaluation score because the final model used all labelled images for training; the cross-validation score remains the more defensible internal performance estimate.

External RealSense tests showed weaker and less stable performance than the internal cross-validation results. Manual testing indicated that storage was recognised under fold 1 with approximately 67% confidence, beds/mattresses were recognised under fold 1 with approximately 47% confidence, and chair seating was only recognised under fold 2 at approximately 27% confidence. These results suggest partial transfer to natural capture conditions, but also show that the model remains sensitive to domain shift between clean product-style training images and RealSense scenes with clutter, lighting variation, colour shift, and natural backgrounds.

The saved RealSense size-estimation outputs are summarised below:

| Capture | Predicted class | Confidence | Estimated size (cm) | Size category |
|---|---|---:|---|---|
| `capture_20260527_022745` | chair_seating | 0.837 | 149.3 x 112.5 x 59.0 | large |
| `capture_20260618_222719` | storage | 0.341 | 83.6 x 73.2 x 12.8 | medium |
| `capture_20260618_232017` | chair_seating | 0.637 | 91.7 x 72.5 x 31.6 | medium |
| `capture_20260620_164635` | sofa | 0.713 | 172.7 x 111.0 x 88.2 | large |
| `capture_20260620_165001` | tables_desks | 0.219 | 125.9 x 95.4 x 16.1 | large |
| `capture_20260620_165406` | sofa | 0.773 | 181.2 x 107.1 x 50.9 | large |
| `capture_20260620_165713` | sofa | 0.152 | 199.1 x 91.3 x 3.7 | large |

The lower-confidence detections, particularly values near or below 0.25, should be interpreted cautiously. They are useful for diagnostic analysis but are not strong evidence of reliable recognition. These external tests support the argument that the prototype is functional, but that further natural-background training data would be required before deployment.

## 4.3. Reference Matching and CO2e Scenario Outputs

The reference matching stage used the detected item class and estimated size category to retrieve comparable products from the reference dataset. These matches were used to estimate a plausible product weight range and material-family profile. The LCA stage then applied scenario factors to estimate indicative CO2e ranges for reuse, closed-loop recycling, incineration with energy recovery, and landfill.

For the storage capture (`capture_20260618_222719`), the reference matching stage selected 10 candidate products. The item was assigned a mixed material family and a weight range of 22.28-22.71 kg. Nine matched products used source/reference weights and one used an imputed weight. The resulting CO2e scenario estimates were:

| Scenario | CO2e range (kg CO2e) | Decision score |
|---|---:|---:|
| Reuse avoided production | -30.22 to -14.45 | -14.45 |
| Closed-loop recycling | 0.09 to 0.10 | 0.10 |
| Incineration with energy recovery | 0.09 to 0.10 | 1.23 |
| Landfill | 11.39 to 17.87 | 20.14 |

For the chair-seating capture (`capture_20260618_232017`), the reference matching stage selected 10 candidate products. The item was assigned a mixed material family and a weight range of 8.00-11.81 kg. Eight matched products used source/reference weights and two used imputed weights. The resulting CO2e scenario estimates were:

| Scenario | CO2e range (kg CO2e) | Decision score |
|---|---:|---:|
| Reuse avoided production | -78.44 to -24.62 | -24.62 |
| Closed-loop recycling | 0.03 to 0.05 | 0.05 |
| Incineration with energy recovery | 0.03 to 0.05 | 0.64 |
| Landfill | 2.29 to 8.25 | 9.43 |

In both examples, reuse showed the largest potential avoided emissions where the item is actually suitable for reuse. However, because item condition was entered as unknown, the system recommended the best non-reuse route, which was recycling. This prevents the system from automatically recommending reuse without a manual suitability judgement.

The closed-loop recycling and incineration scenarios produced very similar raw CO2e ranges for the storage and chair examples. To avoid presenting these as equivalent routing choices, the final recommendation used a decision score that combines the upper-bound CO2e estimate with a waste-hierarchy adjustment. This preserved the raw CO2e ranges for transparency while ranking recycling ahead of incineration when the emissions ranges were otherwise similar.

These results demonstrate the intended decision-support behaviour: the model produces indicative scenario ranges rather than exact LCA values, and it makes the uncertainty visible through weight ranges, material composition profiles, and scenario intervals. The results should therefore be interpreted as suitable for proof-of-concept triage and comparison, not as product-specific carbon accounting.

## Suggested Figure Captions

Figure 4.1. RealSense chair capture with YOLO detection and aligned depth frame. The object was detected as `chair_seating` with 0.837 confidence and passed to depth-based size estimation.

Figure 4.2. Five-fold cross-validation detector metrics. Error bars show standard deviation across folds. mAP50-95 was used as the main detector performance metric.

Figure 4.3. Storage capture with YOLO detection and aligned depth frame. The storage item was detected with lower confidence, illustrating the challenge of natural-scene RealSense inputs.

Figure 4.4. CO2e scenario ranges for the storage capture. Reuse indicates potential avoided production emissions, while recycling is recommended as the best non-reuse route under unknown condition status.

Figure 4.5. CO2e scenario ranges for the chair-seating capture. The chart shows the larger avoided-emissions potential of reuse and the higher impact of landfill relative to recycling/incineration.

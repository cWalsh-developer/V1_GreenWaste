# 4. Results

This section presents the outputs of the GreenWaste proof-of-concept pipeline. The results are separated into three parts: the end-to-end prototype output, the object recognition and size-estimation results, and the downstream reference matching and CO2e scenario estimates.

A distinction is made between internal detector evaluation and external RealSense testing. The internal evaluation uses labelled image splits and five-fold cross-validation, and is therefore the most defensible quantitative evidence for detector performance. The RealSense examples are treated as qualitative stress tests because they contain natural backgrounds, camera noise, lighting variation, and a much smaller number of manually tested scenes.

## 4.1. Prototype Pipeline Output

The prototype was tested as an end-to-end RGB-D decision-support pipeline. A saved Intel RealSense capture was used as the input, containing an RGB image, an aligned depth image, and metadata describing camera intrinsics and depth scale. The pipeline then performed object detection, extracted the predicted bounding-box region from the depth frame, estimated approximate object dimensions, matched the detected item against a reference product dataset, and generated scenario-level CO2e outputs.

The pipeline produced the following artefacts for each processed capture.

| Output stage | Example output | Purpose |
|---|---|---|
| RGB-D capture | `rgb.png`, `depth.png`, `metadata.json` | Stores colour, depth, and camera calibration data |
| YOLO detection and size estimation | `*_yolo_size.csv` / `*_yolo_size.json` | Stores predicted class, confidence, bounding box, and estimated dimensions |
| Reference matching | `*_reference_match_summary.csv` / `.json` | Stores matched reference products and estimated weight range |
| CO2e scenario estimation | `*_lca_estimates.csv` / `.json` | Stores scenario-level CO2e ranges and recommendation scores |
| Combined summary | `*_v1_demo_summary.json` | Combines the full pipeline output for a single capture |

This confirms that the main research objective was met at prototype level: the system can connect computer vision, depth-based size estimation, reference matching, and indicative environmental decision support into a single workflow. However, the reliability of the later stages depends strongly on the correctness of the item recognition stage. If the detector assigns the wrong furniture category, the reference matching and CO2e calculation will be routed through the wrong product group.

Suggested figures for this section:

| Figure | File | Use |
|---|---|---|
| Figure 4.1 | `docs/results_assets/capture_20260527_022745_annotated_detection.png` | Example successful RealSense detection and size-estimation input |
| Figure 4.2 | `docs/results_assets/capture_20260618_222719_annotated_detection.png` | Lower-confidence RealSense example showing natural-scene difficulty |
| Figure 4.3 | `docs/results_assets/capture_20260618_222719_lca_scenarios.png` | Example scenario output chart |

## 4.2. Item Recognition and Size Estimation Results

The object detector was evaluated across several dataset versions. The most defensible quantitative result came from the original stratified five-fold cross-validation experiment. This dataset contained 420 labelled images across five furniture categories: `beds_mattresses`, `chair_seating`, `sofa`, `storage`, and `tables_desks`. Cross-validation was used because the dataset was relatively small; across the five folds, each labelled image was used for validation once while maintaining class balance.

The original stratified cross-validation achieved the strongest internal performance:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Precision | 0.908 | 0.014 |
| Recall | 0.868 | 0.070 |
| mAP50 | 0.943 | 0.030 |
| mAP50-95 | 0.839 | 0.033 |

mAP50-95 was treated as the main detector metric because it evaluates both classification and localisation across stricter overlap thresholds than mAP50 alone. The mean mAP50-95 of 0.839 indicates that the detector learned the labelled product-image dataset well. The fold-level spread, from 0.804 to 0.879 mAP50-95, also shows that performance remained sensitive to the limited dataset size.

A later dataset version blended the original IKEA-derived labelled images with curated natural-background images. This increased the labelled dataset to 560 images, with 394 training images, 83 validation images, and 83 test images in the stratified split. However, the five-fold cross-validation result dropped substantially:

| Dataset version | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Original stratified CV | 0.908 | 0.868 | 0.943 | 0.839 |
| IKEA + curated CV | 0.800 | 0.813 | 0.849 | 0.378 |

This result is important because it shows that simply adding more images did not improve the model. The larger dataset introduced more natural-background variation, but it also introduced greater visual inconsistency and harder class boundaries. In particular, `chair_seating` and `sofa` overlap visually, while some storage units and tables/desks can share similar rectangular shapes. The sharp drop in mAP50-95 suggests that localisation quality and label consistency were affected, not only class prediction.

A further experiment trained on all available images using a mixture of trusted labels and weak labels. This produced 2,022 training images, but the weak labels included generated foreground boxes and full-image boxes rather than consistently hand-drawn bounding boxes. Although the internal training diagnostics appeared high, manual testing showed poor transfer to RealSense scenes. This experiment was therefore treated as a negative result: weakly labelled scale did not compensate for inconsistent annotation quality.

The main experiment comparison is summarised below.

| Experiment | Evaluation type | mAP50-95 | Interpretation |
|---|---|---:|---|
| Original stratified five-fold CV | Unbiased internal CV | 0.839 | Best defensible quantitative detector result |
| Final augmented all-labelled model | Train-as-validation diagnostic | 0.949 | Useful final fitting run, but not an unbiased score |
| IKEA + curated stratified split | Held-out validation split | 0.386 | Natural-background addition reduced localisation performance |
| IKEA + curated five-fold CV | Unbiased internal CV | 0.378 | Confirms the curated blend did not improve generalisation |
| All-available weak-label model | Train-as-validation diagnostic | 0.427 | Weak labels increased scale but reduced practical reliability |

Suggested figures for this section:

| Figure | File | Use |
|---|---|---|
| Figure 4.4 | `docs/results_assets/cv_original_vs_curated_comparison.png` | Best single chart for defending why the original CV result is stronger |
| Figure 4.5 | `docs/results_assets/cv_fold_map50_95_comparison.png` | Shows fold-level stability and the drop after adding curated data |
| Figure 4.6 | `docs/results_assets/experiment_map50_95_summary.png` | Summarises the main modelling experiments |
| Figure 4.7 | `docs/results_assets/original_cv_fold1_confusion_matrix_normalized.png` | Example original CV confusion matrix |
| Figure 4.8 | `docs/results_assets/curated_stratified_confusion_matrix_normalized.png` | Shows class confusion in the later curated model |
| Figure 4.9 | `docs/results_assets/final_augmented_training_curves.png` | Final model training curves, clearly labelled as train-as-validation |

The RealSense tests were less stable than the internal validation results. In manual testing, an obvious chair was predicted as `sofa` with higher confidence than `chair_seating`, and a storage unit was also predicted as `sofa`. A bed image was not reliably recognised; with no confidence threshold it was predicted as a table/desk, and with a manual confidence threshold it was predicted as chair/seating. These cases were not used as formal accuracy statistics because the RealSense sample size was too small, but they are useful evidence of domain shift.

The key explanation is that most training data consisted of clean product-style images, while the RealSense captures contained natural backgrounds, different colour response, perspective effects, clutter, and sensor noise. The model could learn the dataset classes under internal validation, but it did not yet have enough representative real-world, manually annotated RGB-D examples to transfer reliably. This supports treating the system as a proof of concept rather than a deployment-ready detector.

The saved RealSense size-estimation outputs are shown below.

| Capture | Predicted class | Confidence | Estimated size (cm) | Size category |
|---|---|---:|---|---|
| `capture_20260527_022745` | chair_seating | 0.837 | 149.3 x 112.5 x 59.0 | large |
| `capture_20260618_222719` | storage | 0.341 | 83.6 x 73.2 x 12.8 | medium |
| `capture_20260618_232017` | chair_seating | 0.637 | 91.7 x 72.5 x 31.6 | medium |
| `capture_20260620_164635` | sofa | 0.713 | 172.7 x 111.0 x 88.2 | large |
| `capture_20260620_165001` | tables_desks | 0.219 | 125.9 x 95.4 x 16.1 | large |
| `capture_20260620_165406` | sofa | 0.773 | 181.2 x 107.1 x 50.9 | large |
| `capture_20260620_165713` | sofa | 0.152 | 199.1 x 91.3 x 3.7 | large |

Lower-confidence detections, especially those near or below 0.25, should be interpreted cautiously. They demonstrate that the pipeline can produce outputs, but they should not be treated as strong evidence of reliable recognition.

## 4.3. Reference Matching and CO2e Scenario Outputs

The reference matching stage used the detected item class and estimated size category to retrieve comparable products from the reference dataset. These matches were used to estimate a plausible product weight range and material-family profile. The LCA stage then applied scenario factors to estimate indicative CO2e ranges for reuse, closed-loop recycling, incineration with energy recovery, and landfill.

For the storage capture (`capture_20260618_222719`), the reference matching stage selected ten candidate products. The item was assigned a mixed material family and a weight range of 22.28-22.71 kg. Nine matched products used source/reference weights and one used an imputed weight. The resulting scenario estimates were:

| Scenario | CO2e range (kg CO2e) | Decision score |
|---|---:|---:|
| Reuse avoided production | -30.22 to -14.45 | -14.45 |
| Closed-loop recycling | 0.09 to 0.10 | 0.10 |
| Incineration with energy recovery | 0.09 to 0.10 | 1.23 |
| Landfill | 11.39 to 17.87 | 20.14 |

For the chair-seating capture (`capture_20260618_232017`), the reference matching stage selected ten candidate products. The item was assigned a mixed material family and a weight range of 8.00-11.81 kg. Eight matched products used source/reference weights and two used imputed weights. The resulting scenario estimates were:

| Scenario | CO2e range (kg CO2e) | Decision score |
|---|---:|---:|
| Reuse avoided production | -78.44 to -24.62 | -24.62 |
| Closed-loop recycling | 0.03 to 0.05 | 0.05 |
| Incineration with energy recovery | 0.03 to 0.05 | 0.64 |
| Landfill | 2.29 to 8.25 | 9.43 |

In both examples, reuse produced the largest potential avoided-emissions benefit where the item is suitable for reuse. However, because item condition was entered as unknown, the system recommended the best non-reuse route rather than automatically recommending reuse. This is appropriate for a waste-management decision-support tool because reuse depends on condition, safety, and practical suitability, not only estimated emissions.

The closed-loop recycling and incineration scenarios produced similar raw CO2e ranges in these examples. To avoid presenting these as equivalent decisions, the final recommendation used a decision score that combines the upper-bound CO2e estimate with a waste-hierarchy adjustment. This preserves the raw CO2e ranges for transparency while ranking recycling ahead of incineration when the emissions estimates are otherwise similar.

These outputs should be interpreted as indicative scenario estimates rather than product-specific carbon accounting. The pipeline is most useful as a proof-of-concept triage tool: it can show how item class, approximate size, reference matching, and disposal route assumptions influence the likely environmental recommendation. The results also make the main limitation visible: classification errors at the vision stage can propagate into the reference matching and CO2e estimate.

## Suggested Captions

Figure 4.1. Example RealSense chair capture with YOLO detection and aligned depth-based size estimation.

Figure 4.2. Lower-confidence RealSense storage capture, illustrating the difficulty of natural-scene inputs compared with clean product images.

Figure 4.3. Example CO2e scenario output for the storage capture. Negative reuse values represent avoided production emissions.

Figure 4.4. Original stratified cross-validation compared with the later IKEA + curated dataset. Error bars show standard deviation across folds.

Figure 4.5. Fold-level mAP50-95 results. The original stratified dataset produced consistently stronger localisation performance than the later curated blend.

Figure 4.6. Main modelling experiments by mAP50-95. Train-as-validation runs are included for transparency but should not be treated as unbiased model evaluation.

Figure 4.7. Normalised confusion matrix from an original cross-validation fold.

Figure 4.8. Normalised confusion matrix from the IKEA + curated stratified experiment, showing increased class confusion.

Figure 4.9. Training curves for the final augmented all-labelled model. This figure supports final fitting behaviour, while the cross-validation results remain the defensible evaluation.

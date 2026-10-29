# Independent review of Ferret MRI candidate changes

This package is ready for review, not a confirmed-injury dataset. Open `index.html` locally. Nothing has been sent to a reviewer.

## What was completed

The existing pipeline decoded T2 scans, generated provisional atlas masks, registered paired scans, trained ordinary and first-order meta-learning models with simulated changes across three seeds, and generated exploratory scores for 15 pairs. Four pairs failed registration QC and remain withheld; one baseline has no unambiguous match. This package adds all-slice native and aligned comparisons, component IDs, and blank review forms. It does not retrain models, change the threshold, or establish diagnostic validity.

## Suggested review procedure

1. Verify the animal ID, baseline timing, intervention, control status, and acquisition dates from laboratory records. Scanner names alone do not verify exposure or health.
2. Review native baseline and native follow-up images before looking at the outline column. These are decoded source reconstructions, not masked images. Each has its own percentile display window; brightness is not quantitative T2. A matching stored slice index is not necessarily matching anatomy.
3. Check aligned baseline/follow-up anatomy and the existing processing QC. Four withheld pairs are included for image-quality review only. Do not interpret their absent contours as absence of abnormalities.
4. Examine every slice, including regions outside model contours and outside the provisional mask. Note changes overlooked by the model in `subject_review.csv` with the slice number and description.
5. For a model-highlighted region, record its candidate ID in `candidate_review.csv`. Assess native-image visibility, neighboring-slice consistency, registration plausibility, and supporting sequences if available. Single-slice findings are not automatically rejected: slices are thick.
6. Use reviewer assessments such as `indeterminate`, `likely artifact`, `normal anatomical structure`, or `possible biological abnormality`. A possible abnormality is not necessarily traumatic injury. A reviewer should document the evidence and qualifications behind any stronger conclusion.
7. Retain original forms and save completed forms under new names. All reviewer fields are deliberately blank. No model prediction should be recycled as independent ground truth.

## How highlights were generated

- Frozen ensemble change-score maps from the three existing meta-learning runs; scores are not calibrated injury probabilities.
- Existing score threshold strictly greater than 0.5, restricted to the provisional brain mask and follow-up coverage. No tuning against reviewer findings.
- All 26-connected 3D components retained, minimum size one voxel. Neighboring voxels can connect through edges/corners; components can merge distinct phenomena or split a single finding. Candidate counts are not lesion counts.
- Candidate IDs are stable for these frozen inputs. NIfTI integer values correspond to the C-number in the CSV. Coordinates are zero-based array indices; displayed slices are one-based. Anatomical left/right remains unverified.
- Orange outlines include every component on a slice; cyan text identifies only the three largest on that slice to reduce overlap. Full IDs are in the label volumes.
- Edge fraction describes the fraction of a component within one in-plane voxel of the common-mask/coverage boundary. It is an artifact-review cue, not a validated classifier. Component voxel counts are not validated lesion volumes.
- Mean seed SD measures variation between three models, not calibrated uncertainty. Slice persistence and seed agreement cannot confirm injury.
- Full native frames are shown. Aligned images use the prior 160 × 160 grid and independent per-visit normalization within the common mask. The native follow-up column is intentionally unregistered and must not be interpreted as voxelwise aligned.

## Scope and unresolved work

20 baseline animals, 19 follow-ups, 15 scored pairs, four withheld pairs, one unmatched baseline. All 18 stored T2 slices per baseline are displayed, including slices outside the central training selection. Only T2 is reviewed here; other sequences have not been matched or checked in this package. No new robustness experiments, sham calibration, or biological validation have been performed. Source binary hashes are checked against the existing manifest before rendering. Frozen model-map hashes are in `manifest.json`.

Training/validation/development-test membership is retained. Real maps for training animals are in-sample. Inspected development holdouts are not a locked publication test set. Independent validation and verified controls remain necessary to measure injury-detection performance or false alarms.

## Draft request to a reviewer (not sent)

Please review these longitudinal Ferret T2 MRI comparisons for image quality and possible biological abnormalities. Automated contours were generated by a model trained on synthetic intensity changes and are not injury labels. Please first inspect the images without contours, assess registration, and record findings both inside and outside highlighted areas. We would appreciate candidate-level assessments, confidence expressed in your own terms, and recommendations for supporting sequences or records. Please distinguish imaging abnormalities from evidence of trauma. Exposure and control records still require verification.

## Reproduce

From the project root: `MPLCONFIGDIR=/tmp/ferret-mpl ferret-study/.venv/bin/python ferret-study/scripts/build_candidate_review.py`.

Rebuilding overwrites generated CSV templates. Save reviewer-completed copies separately before rebuilding.

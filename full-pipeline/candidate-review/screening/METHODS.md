# Provisional candidate screening: engineering rules v1

These are automated screening labels, not human visual assessments, diagnoses, or validated lesion labels. Thresholds below are transparent engineering choices, not clinically established cutoffs. This exploratory analysis uses previously inspected development data. It must not be presented as a preregistered confirmatory experiment.

## Inputs and calculations

Use the frozen candidate-ID volumes and score maps from the existing 15 accepted T2 pairs. Four registration-QC failures remain excluded from model interpretation; the unmatched animal remains unpaired. All 210 connected components are retained, including tiny ones. No new training or adjustment of the original >0.5 candidate threshold is performed.

Normalize each visit using its own 1st and 99.5th percentiles within the common provisional mask and coverage region. Compute signed follow-up minus baseline intensity; units are normalized display intensity, not quantitative T2. A global contrast change can produce an apparent regional difference.

Calculate the fraction of each component within two processing pixels of the in-plane common-mask boundary. This is mask/coverage proximity, not verified anatomical tissue-boundary proximity. Calculate the fraction of original component voxels retaining a score >0.6. This measures sensitivity of the frozen score map to a second threshold; it is not independent validation.

Shift the already normalized follow-up image and coverage mask by one processing pixel in each of four in-plane directions (+x, -x, +y, -y). Within each fixed candidate, calculate the mean signed difference using only remaining valid coverage. Contrast retention is the smallest sign-consistent shifted mean divided by the absolute unshifted mean (denominator floored at 1e-8). Coverage retention is the smallest valid fraction across shifts. These shifts probe local sensitivity, not realistic registration uncertainty. Models and registration are not rerun. No robustness to intensity normalization, alternative masks, acquisition noise, or large registration errors is established.

## Labels, in decision order

1. **Likely processing artifact:** at least 80% of the component lies within the two-pixel boundary band, AND at least one shift retains <90% coverage or <50% signed contrast. The wording denotes a provisional artifact concern. True peripheral abnormalities can satisfy these rules; do not discard them as confirmed false positives.
2. **Possible change:** all of the following hold: at least 30 processing voxels; at least two stored slices; at least 50% of voxels outside the boundary band; at least 50% retained above score 0.6; absolute mean normalized difference at least 0.1; all shifts retain at least 90% coverage and 50% signed contrast; global baseline/follow-up correlation at least 0.7. These rules establish only technical screening support, not biological origin or injury specificity.
3. **Uncertain:** everything else. The table lists failed support rules. Failing a rule does not establish normal tissue, artifact, or absence of injury. Small or single-slice abnormalities can be real; whole-pair correlation cannot validate local alignment.

Component continuity is defined by the previous 26-connected 3D labeling. Thick slices and interpolation constrain spatial interpretation. Neither the model nor these rules can exclude abnormalities outside the provisional mask, below the threshold, or outside available T2 coverage.

## Shortlist and images

Select up to two largest components per screening label per animal for display. This keeps examples from multiple animals and categories visible; it does not rank injury severity or select all possible biological changes. All 210 components and reasons remain in `all_candidate_screening.csv` and JSON, including candidates not shown on the shortlist.

Each shortlist image shows the slice with the largest candidate cross-section and its immediate neighbors, on the existing aligned processing grid, cropped to the full mask extent with surrounding context. Comparison columns show baseline, follow-up, follow-up with only that candidate outlined, and signed difference (fixed range -0.4 to +0.4). Display windows are separately normalized per visit. Links lead to the full unmasked native scan sheets. Native visibility and other-sequence support are deliberately marked **not assessed**; these have not been automatically verified or individually reviewed by a radiologist.

## Independence and reproduction

Review forms in the parent directory are not modified. Screening labels are saved in separate files with their automated origin. Never train or evaluate an injury model using these labels as independent ground truth.

Run from the project root: `MPLCONFIGDIR=/tmp/ferret-mpl ferret-study/.venv/bin/python ferret-study/scripts/screen_candidates.py`.

The machine-readable configuration and counts are in `summary.json`; parent `manifest.json` records the original score-map hashes. Publication requires sensitivity analysis of these screening choices and independent evidence appropriate to the claim.

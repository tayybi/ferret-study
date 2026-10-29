# Ferret end-to-end development pipeline

**Status: computational development workflow completed. Injury detection and anatomical masks are not clinically or experimentally validated.**

## Completed stages

- Decoded scanner T2 data with stored intensity scaling and checked geometry.
- Preserved scanner physical geometry on a 160 × 160 processing grid. Anatomical RAS orientation remains unverified.
- Generated provisional public-atlas brain masks for 20 baseline-labeled animals.
- Matched and rigidly registered 19 labeled follow-ups; ambiguous BCF94/BCR94 naming was not silently resolved.
- Created extraction/registration QC sheets and withheld inference for automated registration failures.
- Trained ordinary and first-order meta-learning models across multiple random seeds, using synthetic changes restricted to eroded provisional masks.
- Selected checkpoints using validation animals; scored synthetic examples on animal-separated development holdouts.
- Adapted each meta-trained model using each animal baseline only and generated exploratory real follow-up score maps.
- Saved transforms, images, masks, score maps, seed disagreement, configuration, source hashes, and summary tables.

## Synthetic development results

| Run | Method | Synthetic Dice at 0.5 | Synthetic AP |
|---|---|---:|---:|
| atlas-development-seed42 | erm_no_adaptation | 0.948 | 0.994 |
| atlas-development-seed42 | erm_baseline_adaptation | 0.948 | 0.994 |
| atlas-development-seed42 | fomaml_baseline_adaptation | 0.951 | 0.994 |
| atlas-development-seed43 | erm_no_adaptation | 0.939 | 0.993 |
| atlas-development-seed43 | erm_baseline_adaptation | 0.939 | 0.993 |
| atlas-development-seed43 | fomaml_baseline_adaptation | 0.942 | 0.993 |
| atlas-development-seed44 | erm_no_adaptation | 0.932 | 0.979 |
| atlas-development-seed44 | erm_baseline_adaptation | 0.932 | 0.979 |
| atlas-development-seed44 | fomaml_baseline_adaptation | 0.938 | 0.983 |

## Real follow-up outputs

Exploratory maps generated for 15 of 19 matched pairs. Any withheld pairs are listed in inference_summary.csv.

The score is learned from synthetic changes and is not a calibrated probability of injury. The 0.5 score threshold has no false-alarm guarantee. Seed disagreement is model variability, not a validated uncertainty interval. Outputs for training animals are in-sample exploratory inference; development-holdout animals are distinguished in the summary.

Visual review of selected maps showed boundary-heavy and broad intensity-related scores. These patterns may arise from registration, residual contrast differences, masking, or normal longitudinal changes; they are not evidence of injury. Conservative atlas masks omit parts of the brain, so whole-brain lesion-volume claims are not supported.

## Review before scientific claims

All atlas masks require expert anatomical review. Registration metrics are engineering QC and are not proof of alignment accuracy; inspect processing_qc.png. Exposure, sham status, pre-injury timing, stable animal IDs, and real lesion references remain unverified. Normal growth, acquisition differences, masking, and registration errors can produce change scores. No real injury AUROC, lesion Dice, sensitivity, specificity, or conformal control is claimed.

This pipeline uses rigid longitudinal registration to reduce the risk of deforming away biological changes. Baseline masking and per-visit percentile normalization can still suppress or distort genuine changes and need sensitivity analysis.

The pilot holdouts have already been inspected during development. They are not a locked confirmatory publication test set. Plan independent evaluation or a properly nested animal-level study before publication.

## Files

- subjects/<animal>/processing_qc.png: baseline-mask, follow-up, checkerboard QC.
- subjects/<animal>/change_maps.png: baseline, follow-up, subtraction, score overlay.
- subjects/<animal>/*.nii.gz: intermediate volumes, provisional masks, score maps.
- pair_inventory.json and inference_summary.csv: pairing, processing, QC, inference.
- data/manifest.json: frozen animal partitions and source provenance.
- ../runs/<run>/: checkpoints, learning curves, synthetic examples, metrics.
- reference_provenance.json: adult-ferret atlas attribution and hashes.

Atlas: Hutchinson et al. (2017), https://doi.org/10.1016/j.neuroimage.2017.03.009; downloaded from https://scalablebrainatlas.incf.org/ferret/HSRetal17.

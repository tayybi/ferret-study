# Initial technical training run

**This is a synthetic-change feasibility experiment, not injury validation or evidence of novelty.**

Scanner baseline labels are provisional. Provisional public-atlas masks were used; no expert-verified masks, longitudinal sham calibration, or real lesion references were available. No real follow-up images were used to train or evaluate this pilot.

| Method | Synthetic AUROC | Synthetic AP | Positive-slice Dice at 0.5 | Negative foreground fraction flagged at 0.5 |
|---|---:|---:|---:|---:|
| erm_no_adaptation | 0.998 | 0.979 | 0.932 | 0.448% |
| erm_baseline_adaptation | 0.998 | 0.979 | 0.932 | 0.448% |
| fomaml_baseline_adaptation | 0.998 | 0.983 | 0.938 | 0.171% |

Values are unweighted means across held-out animals. Pixel AUROC/AP are first computed within each animal; they are not independent animal sample sizes. Threshold 0.5 was fixed, not calibrated to a false-alarm target. The evaluation region is a provisional atlas-derived mask, not expert anatomical ground truth.

Test simulated abnormalities use a two-lobe shape; training uses ellipses. Both share a simple synthetic intensity model, so this is only a limited morphology holdout, not proof of transfer to real injury.

The ERM and FOMAML models start with identical weights and see identical synthetic training query episodes. Both train the reconstruction head. The additional ERM + adaptation result tests ordinary baseline adaptation without meta-training. Best checkpoints are selected on validation loss only. Final test metrics are not used for checkpoint selection.

Required before real study: confirmed animal/health/exposure records, reviewed ferret brain extraction, registration quality control, healthy follow-up controls, independent expert evaluation, and broader validation across animals/seeds.

# Initial technical training run

**This is a synthetic-change feasibility experiment, not injury validation or evidence of novelty.**

Training and evaluation completed on the Mac's MPS GPU in 28.15 seconds (excluding report plotting and process startup). Both methods ran 200 training episodes. Four focused checks passed for episode generation, disjoint animal partitions, restricted adaptation, and the first-order outer update. Saved checkpoint tensors were checked for finite values.

The synthetic Dice values are 0.878 for ordinary training and 0.886 for first-order meta-learning with adaptation. This single-seed difference is exploratory and does not establish that meta-learning or personalization is beneficial. Ordinary baseline adaptation alone made essentially no difference in this run.

Visual QA confirmed that the heuristic placement region can include tissue outside the anatomical brain. This is therefore a **head-image change-detection engineering test**, not a brain-lesion benchmark. A reviewed brain mask is required before using synthetic abnormalities as an injury-oriented training exercise. The four prototype test animals should be treated as development holdouts in subsequent iterations; this pilot does not establish a frozen confirmatory study test set.

Scanner baseline labels are provisional. No verified brain masks, longitudinal sham calibration, or real lesion references were used. No real follow-up images were used to train or evaluate this pilot.

| Method | Synthetic AUROC | Synthetic AP | Positive-slice Dice at 0.5 | Negative foreground fraction flagged at 0.5 |
|---|---:|---:|---:|---:|
| erm_no_adaptation | 0.997 | 0.951 | 0.878 | 0.152% |
| erm_baseline_adaptation | 0.997 | 0.951 | 0.878 | 0.151% |
| fomaml_baseline_adaptation | 0.998 | 0.959 | 0.886 | 0.045% |

Values are unweighted means across held-out animals. Pixel AUROC/AP are first computed within each animal; they are not independent animal sample sizes. Threshold 0.5 was fixed, not calibrated to a false-alarm target. The foreground region is a heuristic central signal region, not an anatomical brain mask.

Test simulated abnormalities use a two-lobe shape; training uses ellipses. Both share a simple synthetic intensity model, so this is only a limited morphology holdout, not proof of transfer to real injury.

The ERM and FOMAML models start with identical weights and see identical synthetic training query episodes. Both train the reconstruction head. The additional ERM + adaptation result tests ordinary baseline adaptation without meta-training. Best checkpoints are selected on validation loss only. Final test metrics are not used for checkpoint selection.

Required before real study: confirmed animal/health/exposure records, reviewed ferret brain extraction, registration quality control, healthy follow-up controls, independent expert evaluation, and broader validation across animals/seeds.

# Locked analysis plan: sham-calibrated normative anomaly mapping

Written on 28 October 2026 before any group-level score, burden or endpoint was computed. Its SHA-256 is stored in `analysis_plan.sha256`, and the statistics script refuses to run if this file has changed. This is a local lock, not an OSF registration. Phase 0 record verification (exposure, sex, animal IDs) has **not** been done, so all results are provisional.

## Data
- Blast cohort, `MRI groups/3d` and `7d`: sham, 10 psi and 20 psi. The `30d` folder (2 sessions) is excluded because its internal labels say 7d.
- Fiskum 2023–24 baselines (n=20) are used only in the secondary variant "B" and as an external normal-cohort check. Their exposure status is unverified.
- Channels in template space at 0.5 mm: T2, FA, MD, AD, RD, and log-Jacobian of the T2-to-template warp.
- Analysis mask: template brain mask eroded by 1 voxel, intersected with the voxels covered by T2 and DTI in **every** included session.
- Sessions that fail preprocessing or show template correlation < 0.35 are excluded and listed. No imputation.

## Cross-fitting
- 16 shams are split into 4 folds, stratified by time point, with a fixed seed of 2026.
- Each fold's model trains on the shams outside that fold. Variant B also trains on the Fiskum baselines.
- Each blast animal is assigned to one fold model, round-robin within each group × time cell. Blast and sham residuals therefore come from single held-out models with the same null distribution.

## Model (primary)
- 2.5D masked autoencoder (U-Net with a FiLM bottleneck).
- Input: 3 adjacent coronal slices × 6 channels, plus a patch-mask indicator, the template slice and the AP coordinate. 50% of 8×8 patches are hidden across all three slices.
- Loss: L1 on hidden brain voxels. Primary score uses the mean residual over seeds 0, 1 and 2.
- Inference: 4 random masks plus their complements, so each voxel is hidden 4 times.

## Score and calibration
- Per-channel z of the absolute residual against sham residual mean and SD per voxel. The SD is smoothed with σ = 1.5 voxels and floored at its 10th percentile. Sham statistics are leave-one-out when scoring a sham.
- s(v) = maximum over channels of z, then Gaussian smoothing with σ = 1 voxel.
- Conformal p-value per voxel against the pooled sham s values, excluding the animal being scored. Benjamini–Hochberg at q = 0.05 within each animal.
- **Burden** = fraction of analysis-mask voxels rejected.
- Animal-level conformal p-value = rank of the animal's burden among the 16 sham burdens.

## Endpoints
1. **Primary:** burden, 7d 20 psi vs 7d sham. Exact permutation test (two-sided, difference in means), Hedges g, and a 10,000-sample bootstrap 95% CI.
2. Dose trend: OLS of burden on dose (0, 1, 2) + time (3d/7d) + scan year. The dose coefficient is tested by permuting dose within time strata (10,000 permutations).
3. Time: dose × time interaction, tested the same way.
4. MRS: in animals passing spectroscopy QC (FWHM ≤ 0.08 ppm, S/N ≥ 5, metabolite %SD ≤ 20), Spearman ρ between the mean s(v) inside the PRESS voxel and NAA+NAAG/tCr, Glu/tCr, Ins/tCr and Lac/tCr. Permutation p-values, Holm-corrected across the 4 metabolites. The PRESS position convention is chosen by cohort-wide fraction-in-brain before any correlation is computed.
5. False-alarm check: in leave-one-out shams, report the voxel-level false-positive fraction and the share of shams with any rejected voxel.
6. Synthetic lesions: one 2 mm-radius sphere per sham, with amplitude a ∈ {1, 2, 3, 4} × between-sham SD (T2, MD, AD, RD up; FA down). Report voxel sensitivity at BH q = 0.05 for the model and for baseline B1.

## Comparators and ablations (reported, not used to select the primary)
- B1: per-voxel z of raw channel values against shams, no model.
- B2: group voxelwise max-T permutation test on FA and MD (sham vs 20 psi at each time point).
- Ablations: Fiskum added to the training pool, T2 only, DTI only, without log-Jacobian, FiLM test-time adaptation on vs off, and single-seed spread.

## Not available, so not claimed
- Sex and weight: weights are recorded as 0.001 or 1.2 (placeholders) and sex as UNKNOWN, so neither can be a covariate.
- No histology, no expert masks, no topup/eddy. Distortion correction is a reverse-PE halfway SyN along the phase-encode axis, with rigid motion correction and no bvec rotation.

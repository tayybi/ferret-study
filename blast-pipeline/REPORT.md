# Sham-calibrated normative anomaly mapping: first full run

Run date: 28 October 2026. Status: **computational run complete; all locked endpoints null; results provisional** (Phase 0 record verification not done).

## What was run
| Stage | Outcome |
|---|---|
| Inventory | 54 blast-cohort sessions found; T2, DTI (30 dir × b1000/2000, 5 b0, 2 reps) and reverse-PE b0 present in all; PFC PRESS in 49. Sex is UNKNOWN in every session and weights are placeholders (0.001 / 1.2). |
| Preprocessing | 72/72 sessions OK: 52 blast + 20 Fiskum baselines; 2 "30d" sessions excluded per plan. |
| Model | 2.5D masked autoencoder (U-Net + FiLM), 4 cross-fitted sham folds × 3 seeds; ablations listed below. |
| Statistics | Locked plan (`ANALYSIS_PLAN.md`, SHA-256 checked by the script) plus a separately hashed post-hoc addendum. |

### Preprocessing findings worth reporting
- The custom `neg_DtiEpi` reverse-PE b0 is **stored flipped along the phase-encode axis**. Forward/reverse correlation is about 0.0 as stored and 0.84 after flipping. After half-way SyN correction the median is 0.90, minimum 0.87.
- Motion: each volume is registered to the mean of its own b-shell. Maximum translation is 0.70 mm, with 0 fallbacks to identity.
- PRESS voxel position needs an x/y sign flip relative to image coordinates. With the flip, the mean fraction of the voxel inside the brain is 0.996; as stored it is 0.37.
- Analysis mask: 36,634 voxels (4,579 mm³) at 0.5 mm, covered in every session.

## Locked endpoints (all null)
| Endpoint | Result |
|---|---|
| **Primary:** burden, 7d 20 psi vs 7d sham | 0.00% vs 0.44%. Hedges g = −0.55; exact p = 0.44 |
| Dose trend (Freedman–Lane) | coef = −0.001 per step, p = 0.19 |
| Dose × time | p = 0.22 |
| MRS: anomaly in PRESS voxel vs NAA/tCr | ρ = −0.28, p = 0.086, Holm p = 0.26 (n = 39) |
| MRS vs Glu/tCr, Ins/tCr | ρ = −0.19 and +0.19; Holm p = 0.49 |
| MRS vs Lac/tCr | not testable: no spectrum met %SD ≤ 20 |
| False alarms (LOO shams) | mean voxel FP 0.19%; 1 of 16 shams had any rejected voxel |
| Synthetic lesions, voxel sensitivity at FDR 5% | model: 0 / 0 / 1.5% / 3.7% at 1–4 SD. B1: 0 at all amplitudes |
| B2 voxelwise max-T, FA and MD | 0 significant voxels; peak abs(t) 6.4 (FA, 7d), FWER p = 0.12 |
| Ablations (Fiskum pool, T2-only, DTI-only, no logJ, FiLM adaptation, single seeds) | all null on the primary; p = 0.44 or 1.0 |

The locked burden endpoint has nearly no power. Benjamini–Hochberg over about 36k voxels against a pooled sham null requires a score above almost every sham voxel.

## Exploratory (post hoc, `ANALYSIS_ADDENDUM_EXPLORATORY.md`)
- **E1, top-1% score:** blast animals score *lower* than shams. 7d 20 psi vs sham: g = −0.88, p = 0.066. The dose trend is negative, p = 0.009. The raw-value baseline B1 shows the same direction.
- **Tail diagnostics:**
  - The sham null is dominated by a few animals: 7d-Sham-709611 (7% of voxels above the pooled 99th percentile), 3d-Sham-FY249792 and 7d-Sham-707761.
  - QC shows a misplaced atlas mask in 709611. 709611 and 707761 are the only sessions recorded prone, and their images are upside-down relative to all supine animals. (Corrected 28 Oct 2026: an earlier version wrongly described FY249792 as upside-down; see ERRATA.md.)
  - The extreme voxels are not concentrated at mask edges (48% vs 38% expected) or at end slices.
- **E5:** dropping those two shams (outcome-informed) does not reverse the direction. Top-1% g = −0.72, p = 0.13.
- **Scan-day confound:** several days scanned only one group. 2025-10-30 had only the two highest-scoring 7d shams, and both lack the `7d_post_blast_` study label. 2025-11-06 and 2024-11-07 had only 20 psi animals. Group cannot be separated from scan-session effects with the data as recorded.
- **E2, atlas regions:** directions match the classic injury pattern, but nothing survives FDR (248 tests, minimum q = 0.15).
  - FA −7.8% (g = −1.6) and −10.4% (g = −1.4) in labels 40 and 60 at 3d.
  - MD +7.1% (g = 1.4) in label 30 at 7d.
  - logJ lower in label 28 at 3d (g = −1.9).
- **E3, uncorrected 0.1% operating point:** the model detects 4.3% of 4-SD lesion voxels; B1 detects 0.1%. The model beats raw z-scoring, but both are weak.
- **E4:** Fiskum baselines differ strongly from blast-cohort shams. FA g = −2.6, MD g = +2.9, T2 g = −1.3, all p < 0.001. They are not interchangeable normal controls, and adding them to training changed nothing.

## Record conflicts to resolve with the PI
1. Sessions on 2025-10-30 (7d Sham 707761, 709611) and 2025-11-06 (7d 20 psi 708371, 710161) have bare-ID study names, unlike the rest.
2. Folder `3d/20psi/706801` has study name `3d_post_blast_126594`.
3. Sex and weight are unrecorded in the scanner metadata. They are needed as covariates.
4. Two LCModel analysis variants (`v6` = 19, `ouput` = 23) are mixed. The settings difference must be confirmed.
5. Exposure status of the Fiskum cohort, plus the age and source of both cohorts.

## What this means for the paper
- Detection of blast injury cannot be claimed from this run.
- Defensible contributions as things stand:
  - a validated, open ferret preprocessing pipeline, including the reverse-PE flip and the PRESS sign convention;
  - a conformal anomaly framework with measured sensitivity limits;
  - a quantified demonstration that scan-session heterogeneity in shams exceeds any blast effect detectable with n = 7–9.
- Next steps that could change the outcome:
  - blinded expert registration and mask QC;
  - resolving the scan-day and ID conflicts;
  - a session-level nuisance model, or harmonisation such as ComBat by scan day;
  - region-level rather than voxel-level calibration;
  - more shams per scan day in future acquisitions.

## Files
`results.json` (locked), `results_exploratory.json`, `per_animal.json`, `mrs_table.json`, `registration_qc.json`, `figures/*.png`, `sessions/<id>/qc.png`, `models/<variant>/`. Scripts are in `../scripts/blast_*.py` and `run_blast_training.sh`.

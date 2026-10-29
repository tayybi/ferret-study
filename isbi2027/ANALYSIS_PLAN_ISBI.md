# ISBI 2027 analysis plan: session-aware conformal calibration (SACC)

Written on 28 October 2026, before any SACC result was computed. The blast-cohort data were analysed earlier with a different method (`blast-pipeline/REPORT.md`, all endpoints null). Real-data results here are therefore **exploratory**. The primary evidence is the semi-synthetic benchmark, which has known ground truth.

## Units
- Animal = one session folder. Scan session = scan day: 21 days for the 52 blast-cohort animals; Fiskum is not used here.
- Controls = 16 shams.
- Regions = evDTI atlas labels with ≥ 200 voxels in the analysis mask. Features = region means of 6 channels (T2, FA, MD, AD, RD, logJ).

## Feature sets
- RAW: region means of the channel maps.
- RES-A: region means of masked-autoencoder residuals, with models cross-fitted by animal (existing `main`, 3 seeds).
- RES-S: the same with models cross-fitted by scan session (`main_sess`, 3 seeds).

## Scores and calibration
- For a test animal t with calibration controls K: per region × channel, z = (F − median_K) / (1.4826 · MAD_K). The scale is floored at the 10th percentile of the scales for that channel.
- Region score = max over channels of |z| (RAW) or of z (RES).
- **Naive (animal-level):** K = controls ≠ t. Each calibration control k is scored against K∖{k}. Unweighted pooling over controls × regions.
- **SACC (session-level):** K = controls not scanned in t's session. Each calibration control k is scored against K∖session(k). Weighted pooling: each calibration session carries equal total weight.
- Conformal p-value per region: weighted fraction of calibration scores ≥ the test score, with the +1 correction. BH across regions within each animal at q = 0.10 (primary) and 0.05.

## Experiments
1. **Session effect:** ICC of the session random effect for RAW region features (blast cohort, after removing group × time means), median over region × channel. Null: permute sessions within each group × time cell (2,000 permutations).
2. **New-session null (validity):** hold out each sham session entirely. Test its shams against the remaining shams, with calibration scores computed (a) naively and (b) by SACC. Metrics:
   - P(p ≤ α) at α = 0.01, 0.05, 0.10;
   - per-animal false-alarm rate, the share of test shams with any BH-flagged region.

   Reported for RAW, RES-A and RES-S.
3. **Semi-synthetic power:** each sham, with its session held out, gets a diffuse change in one region (3 random regions per sham, seed 2027). The shift is a × (between-sham SD of the region mean) for a ∈ {0.5, 1, 1.5, 2, 3}, with FA down, MD/AD/RD/T2 up, and logJ unchanged. RAW features are shifted directly. Metrics:
   - sensitivity: the injected region is flagged;
   - realized FDR: flagged non-injected regions among all flagged.

   Comparator: voxel-level pooled BH (the earlier method) applied to the same RAW injection at a ∈ {1, 2, 3}.
4. **Real application (exploratory):**
   - flagged-region counts per blast animal under SACC (RAW and RES-S);
   - group × time summaries;
   - Spearman correlation between the region score of the label containing the PRESS voxel and NAA/tCr (QC as in `ANALYSIS_PLAN.md`).

## Reporting rules
- All configurations are reported, including the ones where SACC does not win.
- No re-tuning of q, the region size threshold or the scale floor after results are seen. Any change is listed as a deviation.

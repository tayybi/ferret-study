# Exploratory addendum (post hoc)

Written on 28 October 2026 **after** seeing the locked results, which were null on every endpoint. The voxel-level BH burden turned out to have near-zero power: 4-SD synthetic lesions gave 3.7% voxel sensitivity. Nothing here replaces the locked endpoints, and every result from this file is labelled exploratory.

- **E1:** animal-level continuous score = mean of the top 1% of s(v) in the analysis mask. Analysed with the same tests as the locked endpoints (exact permutation for 7d 20 psi vs sham, Freedman–Lane dose trend).
- **E2:** atlas-region analysis. For each evDTI label with ≥ 200 voxels in the mask: mean FA, MD, T2 and logJ per animal. Welch t for 20 psi vs sham at each time point, permutation p-values, BH-FDR across regions × metrics × times.
- **E3:** synthetic-lesion sensitivity at an uncorrected voxel threshold equal to the sham-null 99.9th percentile, for the model and B1. This compares detection power without the BH ceiling.
- **E4:** domain shift. Fiskum baselines vs blast shams on E1, and on whole-mask mean FA and MD.

## E5: outcome-informed sensitivity (added after E1–E4)
The E1 tail diagnostics showed the sham null is dominated by a few animals. Visual QC then found a misplaced atlas mask in 7d-Sham-709611 and atypical (upside-down) head positioning in 3d-Sham-FY249792. E5 repeats the locked primary and E1 with these two shams removed from both the reference and the sham group. They were identified **after** seeing scores, so E5 is descriptive and shows only how sensitive the results are to these two animals. A group-blind in-mask registration metric (3 robust SD rule) flagged no sessions; see `registration_qc.json`.

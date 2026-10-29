# Deviations from ANALYSIS_PLAN_ISBI.md (logged before the affected results were computed)

## D1: acquisition-stratified (Mondrian) calibration, added 28 Oct 2026
**Trigger.** First run of experiments 1–4: power was flat at about 8–10% across 0.5–3 SD. Diagnosis: the pooled calibration tail is dominated by two controls, 7d-Sham-707761 and 709611 (session 2025-10-30), whose maximum region scores are 24.8 and 17.6. All other controls are ≤ 10.2.

**Objective cause, from scanner metadata.** These are the only 2 of 52 sessions with `SUBJECT_position = SUBJ_POS_Prone`; the other 50 are Supine. This was found by inspecting metadata **after** seeing which controls were extreme, so the rule is post hoc.

**Change.** Add the variant SACC+M. Calibration controls must share the test animal's recorded acquisition stratum (`SUBJECT_position`, head-first/supine vs prone). A test animal whose stratum has fewer than 2 calibration sessions is reported as *not calibratable* and is not tested. No blast animal is prone, so the prone shams drop out of all calibration sets and out of the test set (2 animals). Everything else is unchanged.

**Reporting.** Results are reported for both SACC (as locked) and SACC+M, with this deviation stated in the paper.

## D2: design simulation for the number of control sessions, added 28 Oct 2026
**Trigger.** SACC+M restored validity, but semi-synthetic power was ≤ 2% at 3 SD. Analysis: with S calibration sessions and m regions, the smallest weighted conformal p-value is 1/(1 + S·m), while BH at level q needs p ≤ q/m for a single rejection. Detection therefore requires S ≥ (1/q − 1/m), and in practice a score above nearly all S·m null scores.

**New experiment 5 (simulation).**
- Generator, fitted only to the 14 supine shams (RAW region features, robust-standardized): Gaussian copula with a Ledoit–Wolf-shrunk correlation across region × channel. A session random effect gives ICC = 0.18 (the measured value from experiment 1). The marginal tails use the empirical distribution of the standardized sham features.
- Vary the number of control sessions S ∈ {5, 10, 20, 40, 80}, with 2 controls per session.
- Inject a one-region shift a ∈ {1, 2, 3} SD into a test animal from a new session. Apply SACC with m = 31 regions and BH at q = 0.10.
- 400 replicates per cell, seed 2028. Report sensitivity and realized FDR.
- This is a design aid. It does not test the blast cohort.

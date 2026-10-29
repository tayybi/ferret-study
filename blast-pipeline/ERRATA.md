# Errata

## 28 Oct 2026: head orientation of 3d-Sham-FY249792
REPORT.md and ANALYSIS_ADDENDUM_EXPLORATORY.md (E5) described FY249792 as having "upside-down head positioning". That was a misreading of the QC sheets. FY249792 is recorded supine and has the normal supine orientation. The sessions that are upside-down relative to the cohort are **7d-Sham-707761 and 7d-Sham-709611**: recorded prone, scanned 2025-10-30, confirmed by viewing native T2 slices. This prone orientation likely explains the misplaced atlas mask in 709611, because the atlas initialisation assumes a supine head.

The E5 sensitivity analysis (dropping 709611 and FY249792) is unaffected numerically, but its stated rationale for FY249792 was wrong. FY249792's remaining atypicality is its slab offset (−1.39 mm vs 0–5 mm in other sessions). The ISBI paper does not rely on this claim. The hashed addendum is left unedited to preserve its lock; this file records the correction.

## Animal identity check
Folder names and study names are unique across all 54 blast-cohort sessions, and the 3d and 7d IDs do not overlap (cross-sectional design). `SUBJECT_id` encodes the operator and scan date, not the animal. Image similarity found no duplicate animals beyond the shared-orientation similarity of the two prone shams (FA z = 4.8, logJ z = 3.3). One-session-per-animal should still be confirmed by the PI.

# Ferret original-record verification — 27 October 2026

The local dataset was inspected directly rather than relying on generated summaries or requesting records before searching. The selected pairs are **internally consistent scanner-record matches**. This is stronger than matching folder names, but not independent verification of biological identity or exposure history.

## Scope and results

- Inventoried 32,980 original files in `Ferret/Fiskum` and `Ferret/MRI groups`, covering 112 session `subject` files.
- Searched 20,594 text/metadata files for BCF/BCR IDs, blast, sham, injury, exposure and pressure labels. Included session/acquisition/reconstruction metadata and text spectroscopy outputs. Generated reports were excluded; image and raw-signal binaries were not searched as text.
- Crosschecked all 39 selected scans against original session `SUBJECT_study_name`, reconstruction `VisuStudyId`, session/reconstruction study UIDs and dates, and acquisition timestamps. All checks passed. Reconstruction metadata may be copied from session metadata, so these are consistency checks, not independent observations of animal identity.
- All 19 selected pairs have exactly matching BCF prefixes. Fifteen `1month` pairs are separated by 40–48 calendar days; four `6month` pairs by 193–194 days. Scan-to-scan intervals do not establish time since injury.
- BCF94 remains unmatched. The February 27, 2024 session and selected image reconstruction both say **BCR94_1month**. No BCF94 correction was found in searched metadata for that session. No automatic renaming was performed.

## What can be verified

The recorded BCF labels, visit labels, acquisition dates, internal consistency of selected scan provenance, and folder-assigned groups in the separate MRI-groups collection can be verified from files. Pairing is supported by explicit study labels, not inferred from scan order or image similarity.

Example: BCF103 is explicitly recorded as `BCF103_baseline` on February 14, 2024 and `BCF103_1month` on March 26, 2024, 41 calendar days apart. Both image headers agree with their session records.

## What the searched records did not establish

- No blast/sham/injury/exposure/10psi/20psi keyword matches were found in searched original Fiskum metadata/text files.
- All 58 Fiskum sessions have blank `SUBJECT_remarks`, `SUBJECT_purpose`, and `SUBJECT_comment` fields.
- No independent animal-identification or intervention record was found for the selected BCF cohort. Generic `SUBJECT_id` and subject UID values are reused for different BCF study labels within a day and change across visits. For example, BCF102 and BCF103 share the same date-based generic subject record on February 14. These fields cannot independently corroborate the BCF animal identity.
- The BCF103 birth field changes with scan date, sex is unknown, and weight is 0.001 at both visits. These fields do not offer reliable independent identity corroboration; the reason for those entries is not established.
- The original collections contain no PDF, XLS/XLSX or DOC/DOCX files in the inventory. TXT files are scanner/spectroscopy-related; the two top-level LCModel TXT lists contain source spectrum paths. This does not prove records never existed elsewhere.
- No BCF/BCR cross-reference was found in the searched original MRI-groups files. Sham/pressure labels there cannot be transferred to BCF animals without a documented mapping.

## Separate MRI-groups evidence

Folder assignments contain 16 Sham sessions, 18 at 10psi and 18 at 20psi. Two additional sessions are in the `30d` folder without a pressure subgroup. Both of those sessions have internal study labels saying `7d_post_blast`, a time-label conflict. Some Sham sessions also use `post_blast` in their scanner study name, so that phrase alone must not override the folder group assignment.

These are group/time labels, not proof of injury. The group collection and longitudinal BCF collection use different recorded identifiers. Do not merge them based on the common species or project context.

## Consequences for this study

Continue using the 19 pairs as **record-label-supported longitudinal pairs**, explicitly documenting the verification level. The 20 baselines are **baseline-labeled scans**, not independently verified healthy or pre-injury scans. No experimental group, intervention date, injury diagnosis or biological identity was upgraded on the basis of this audit. Existing candidate classifications remain automated technical screening labels. The four registration failures and BCF94 exclusion remain unchanged.

## Evidence files

- `all_sessions.csv`: all 112 session labels, dates, generic IDs, group paths, free-text fields, source paths and source SHA-256 hashes.
- `selected_pairs.csv`: all 20 baseline records, selected follow-ups and calendar-day intervals.
- `selected_scan_crosschecks.csv`: explicit fields checked for all 39 selected scans.
- `metadata_search_hits.json`: matching text with source paths and line numbers.
- `summary.json`: search scope, counts, zero exposure hits in Fiskum, and reused generic IDs.

Reproduce from the workspace root with `ferret-study/.venv/bin/python ferret-study/scripts/audit_dataset_records.py`. Original scanner files are read only. No model retraining or label mutation is part of this audit.

# Ferret study: completed development work and limits

Original-record audit completed on 27 October 2026: [audit report](../../record-audit/REPORT.md). All 39 selected scans passed session/image-label, UID and date consistency checks; 19 pairs have matching BCF labels. The searched original Fiskum files did not establish intervention dates or sham/injury assignment. Pairing is now explicitly described as record-label-supported, not independently identity-verified. This audit does not validate injury status.

## Research question

Can adapting a change-detection model to an animal's own baseline improve detection of longitudinal MRI changes while reducing sensitivity to acquisition and processing differences?

The current experiments answer only part of this question. They measure localization of synthetic intensity changes and generate exploratory candidates on real T2 pairs. They do not establish improved detection of real injury or a reduced real false-positive rate.

## Completed technical deliverables

- Frozen animal-level split: 12 training, four validation, four development-test animals. Twenty baseline-labeled sessions and 19 follow-ups; scanner labels and exposure records remain provisional.
- Provisional atlas masks, rigid longitudinal registration, and documented QC. Fifteen accepted pairs; four withheld pairs. Source reconstructions preserved.
- Custom small CNN with ordinary training versus first-order meta-learning, three seeds; ordinary training with baseline adaptation also evaluated. This is not an MAE or a current foundation model.
- Synthetic localization Dice approximately 0.940 ordinary, 0.940 ordinary with adaptation, and 0.944 meta-learning with adaptation. The roughly 0.004 difference is small and is not real-injury validation. Seeds do not increase the animal sample size.
- Frozen real-pair model scores, all-slice native and aligned image sheets, candidate-ID volumes, and blank independent review forms.
- Separate provisional screening labels, transparent rules, threshold-retention and small-shift subtraction checks, and a shortlist with evidence images.

## Defensible description today

An exploratory technical feasibility study of baseline-personalized longitudinal Ferret MRI change detection, with synthetic evaluation and an auditable real-scan candidate-screening workflow.

Do not claim confirmed injury localization, diagnostic accuracy, clinically meaningful recovery, calibrated injury probabilities, guaranteed false-alarm rates, validated lesion volumes, or established novelty. No six-month recovery comparison is supported by the accepted pairs. Avoid counting the 210 model components as injuries or independent biological samples.

## Before making stronger publication claims

1. Verify identity, baseline-before-exposure timing, injury/sham groups, and relevant experimental records. This does not require a radiologist, but it does require reliable lab documentation.
2. Establish an independent reference appropriate to the claim: qualified imaging review for imaging findings and, where available, independent pathology, behavior, or other experimental measurements for biological interpretation. These sources address different questions and are not interchangeable ground truths.
3. Review masks, original images and local alignment. Check matching sequences where available. Current automated screening does not substitute for these checks.
4. Evaluate simple subtraction and other defensible comparison methods on the same animal-level tasks. The present synthetic table compares training/adaptation variants and does not yet constitute a complete real-data benchmark.
5. Quantify sensitivity to preprocessing, synthetic change design, candidate thresholds, and known acquisition nuisances. Current four one-pixel shifts test only normalized subtraction, not end-to-end model robustness.
6. Use independent animals or a properly nested animal-level evaluation for confirmatory results. Already inspected holdouts are development data. Report uncertainty at the animal level; three training seeds are not three cohorts.
7. Perform a focused literature comparison before claiming novelty. Meta-learning, subject adaptation, and synthetic change training are not new by themselves.

If independent injury references remain unavailable, retain a technical/synthetic feasibility scope and clearly report real-data results as unvalidated observations. A completed computational workflow does not guarantee publishability or acceptance. No manuscript has been submitted and no reviewer has been contacted.

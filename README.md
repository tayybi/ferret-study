# Ferret baseline adaptation: first technical training run

## End-to-end development workflow

The extended workflow is in `full-pipeline/`. It processes 20 scanner-labeled T2 baselines and 19 candidate follow-ups, generates provisional atlas masks, performs rigid longitudinal registration with QC, trains ordinary and meta-learning models at three fixed seeds, and produces exploratory follow-up change maps for pairs passing automatic registration checks. Four six-month pairs remain withheld after fixed registration retries. No expert masks, verified experimental groups, real lesion references, or valid sham calibration set have been supplied; biological validation remains incomplete.

Run the complete computational workflow using the project environment:

```sh
MPLCONFIGDIR=/tmp/ferret-mpl ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
ferret-study/.venv/bin/python -u ferret-study/scripts/run_pipeline.py
```

This environment inherits the existing scientific packages and adds `antspyx==0.6.3`, `nibabel==5.4.2`, and `webcolors==25.10.0` locally. GPU execution must have access to MPS. Completed fixed training runs are reused; partial run directories are not overwritten. Preparation is cached by animal. To change processing/model choices, create a separate development run and document the change rather than silently overwriting confirmatory results.

See `full-pipeline/REPORT.md`, `full-pipeline/inference_summary.csv`, and per-animal `processing_qc.png` / `change_maps.png`. Atlas provenance and limitations are in `reference/README.md` and `full-pipeline/reference_provenance.json`. The prototype holdouts are development data, not a locked publication test set.

This project implements a small paired-image CNN and **first-order MAML with baseline-only adaptation**. It is an engineering feasibility pilot using scanner-labeled baseline anatomy and synthetic changes. It does not validate real injury detection, establish novelty, or provide conformal guarantees.

The user authorized provisional use of scanner baseline labels. Health, exposure timing, and animal/group records still require verification.

## Inputs and partitions

- 20 baseline-labeled T2 axial acquisitions, one per provisional BCF animal ID.
- Original geometry: 366 × 366 pixels, 18 slices, 55 × 55 mm field of view, 1.5 mm thickness in these acquisitions.
- Fixed central 12 slices per acquisition, reduced to 128 × 128 for the pilot.
- 12 training, 4 validation, and 4 final-test animals, stratified by collection year with a fixed seed.
- Per-volume percentile normalization. No test-population normalization statistics are estimated.
- No external-drive data, TIFF training masks, spectroscopy, or real post-exposure images are used.
- No brain-extraction model was supplied or validated for these scans. A central signal/ellipse region is used for synthetic placement and technical metrics; it is **not a brain segmentation**.

`data/manifest.json` records animal IDs, sources, candidate follow-ups, image geometry, normalization, original binary hashes, and frozen partitions. Source scanner files remain unchanged.

## Training design

The encoder is shared between baseline and synthetic follow-up. Their features and absolute feature differences feed a small change head. A reconstruction head learns to denoise/reconstruct baseline slices.

Each meta-training episode uses one training animal:

1. Take a few baseline reconstruction gradient steps on 64 channel-scale/channel-offset parameters only.
2. Generate matched-distribution nuisance-only and synthetic-abnormality query examples from the same baseline volume.
3. Compute weighted binary cross-entropy plus positive-example Dice loss, with an auxiliary reconstruction loss.
4. Transfer query gradients from the adapted copy to the shared parameters, using the first-order MAML approximation.

At validation and final testing, adaptation sees only the held-out animal's baseline. Query inputs are generated independently of the support RNG. Validation controls checkpoint selection; final-test scores do not. Training uses elliptical synthetic abnormalities; testing uses a limited held-out two-lobe morphology. Synthetic intensity changes do not establish biological injury realism.

The comparison includes pooled training (ERM), ERM with ordinary baseline adaptation, and first-order MAML with baseline adaptation. Models start from identical weights and see the same synthetic training query episodes. The pilot is not tuned to establish an algorithm ranking.

## Run locally

An existing environment was found at `/opt/anaconda3/bin/python3.12`, with PyTorch 2.3.0 and the needed scientific packages. No packages need to be installed for this pilot.

```sh
/opt/anaconda3/bin/python3.12 ferret-study/scripts/prepare_data.py --size 128
/opt/anaconda3/bin/python3.12 -m unittest discover -s ferret-study/scripts -p 'test_*.py' -v
/opt/anaconda3/bin/python3.12 -u ferret-study/scripts/study.py --device mps --steps 200 --run pilot-001
```

If MPS is not accessible from a restricted execution context, launch from a terminal with GPU access or use `--device cpu`. An explicit GPU forward/backward check passed outside the execution sandbox. The model also supports CUDA by a small future device-selection extension; no cloud resources are configured or charged here.

Each run uses a new directory and refuses to overwrite existing run results. It saves best checkpoints, training/validation history, synthetic test results, learning curves, example maps, configuration, and a report. To change run settings, choose a new `--run` name. Do not repeatedly tune against the same final-test animals.

## First run outcome

`runs/pilot-001` completed on the Mac's MPS GPU: 200 episodes per method, approximately 28 seconds for training and evaluation. All four algorithm/data checks passed and checkpoint values were finite. This establishes local engineering feasibility. The single-seed synthetic comparison does not establish an algorithm advantage. Visual QA found that the heuristic simulation region can include tissue outside the anatomical brain; reviewed brain masks are required before this can become an injury-oriented benchmark. Treat the prototype test animals as development holdouts for future iterations, and define a separate confirmatory evaluation before publication claims.

## Hardware and study limits

The local machine is an Apple M3 Pro MacBook Pro with 18 GB unified memory and a 14-core GPU. It is suitable for this small 2D feasibility model. GPU memory and speed for a large 3D MAE or full second-order meta-learning must be benchmarked separately. Final comparisons should use multiple seeds and animal-level uncertainty estimates, potentially on a larger NVIDIA GPU if necessary.

No real follow-up registration or lesion evaluation is performed. Those steps require verified study records, validated brain extraction, registration quality control, suitable longitudinal healthy/sham controls, and blinded expert reference information. Candidate labels do not establish confirmed health, exposure, or pathology. Technical foreground false positives are not brain-tissue false-positive guarantees.

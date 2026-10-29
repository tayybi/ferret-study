#!/bin/bash
# Train the primary normative model (3 seeds + synthetic lesions), FiLM adaptation and ablations.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { .venv/bin/python -u scripts/blast_model.py "$@"; }
run --variant main --seeds 0 1 2 --synthetic
run --variant main --seeds 0 --adapt-steps 20
for v in fiskum_pool t2_only dti_only no_logj; do run --variant "$v" --seeds 0; done
echo ALL_TRAINING_DONE

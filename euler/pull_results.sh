#!/bin/bash
# Pull results back from Euler to the laptop. Run locally from project root:
#     bash euler/pull_results.sh
set -euo pipefail

EULER_USER="${EULER_USER:-candreozzi}"
EULER_HOST="${EULER_HOST:-euler.ethz.ch}"
EULER_RESULTS="${EULER_RESULTS:-\$HOME/era5-store/results}"

rsync -avz "${EULER_USER}@${EULER_HOST}:${EULER_RESULTS}/" ./results/
echo "pulled into ./results/"

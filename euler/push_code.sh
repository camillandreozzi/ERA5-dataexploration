#!/bin/bash
# Push code (NOT data) from the laptop to Euler. Run locally from project root:
#     bash euler/push_code.sh
set -euo pipefail

EULER_USER="${EULER_USER:-candreozzi}"
EULER_HOST="${EULER_HOST:-euler.ethz.ch}"
EULER_PROJECT="${EULER_PROJECT:-ERA5-dataexploration}"

rsync -avz --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'data/' \
    --exclude 'results/' \
    --exclude 'logs/' \
    --exclude '.DS_Store' \
    ./ "${EULER_USER}@${EULER_HOST}:~/${EULER_PROJECT}/"

echo
echo "pushed to ${EULER_USER}@${EULER_HOST}:~/${EULER_PROJECT}/"
echo "data/ and results/ deliberately excluded -- they live in ERA5_DATA_ROOT on the cluster."

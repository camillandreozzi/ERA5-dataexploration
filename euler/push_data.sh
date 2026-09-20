#!/bin/bash
# Push the CLARA source data from the laptop to Euler. Run LOCALLY, from the
# project root:
#     bash euler/push_data.sh
#
# ERA5 is NOT sent -- Euler downloads that from Copernicus itself. This is only
# for the file that exists nowhere else, the full CLARA record. The per-subset
# CLARA_matched.pkl is rebuilt from it on Euler by euler/preprocess.sbatch.
# Run it once; rerun only when CLARA.pkl changes (rsync skips unchanged files).
set -euo pipefail

EULER_USER="${EULER_USER:-candreozzi}"
EULER_HOST="${EULER_HOST:-euler.ethz.ch}"
EULER_DATA="${EULER_DATA:-era5-store/data}"

ssh "${EULER_USER}@${EULER_HOST}" "mkdir -p ~/${EULER_DATA}"

# No --delete here: never let a laptop cleanup wipe data on the cluster.
rsync -avzP \
    data/CLARA.pkl \
    "${EULER_USER}@${EULER_HOST}:~/${EULER_DATA}/"

echo
echo "CLARA data now on ${EULER_HOST}:~/${EULER_DATA}/"

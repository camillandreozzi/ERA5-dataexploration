#!/bin/bash
#SBATCH --job-name=era5_preprocess_midnight
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/era5_preprocess_midnight_%j.out
#SBATCH --error=logs/era5_preprocess_midnight_%j.err
#SBATCH --mail-type=END,FAIL

# Builds the modelling input for the midnight subset from the fetched GRIBs:
#   1. 103 request GRIBs -> inventory of what arrived
#   2. GRIBs + CLARA     -> midnight_subset/CLARA_ERA5_merged.parquet
#
# CLARA_matched.pkl is not rebuilt here: fetch_era5_midnight.sbatch already
# wrote it, and the request plan must match the GRIBs on disk. Re-running
# read_inclara.py would be harmless today but would silently invalidate the
# downloaded files if the selection rule ever changed.
#
# Shorter wall clock than preprocess.sbatch because this subset keeps only the
# CLARA-matched rows -- about a thousand -- rather than a full lat/lon grid.
# Extra arguments are passed to step 2, e.g. a smoke test:
#     sbatch euler/preprocess_midnight.sbatch --allow-missing-files

set -euo pipefail

# Must be set before sourcing: env.sh defaults ERA5_SUBSET to spatial_subset
# and applies spatial-only blocked-CV overrides when it matches.
export ERA5_SUBSET=midnight_subset
source euler/env.sh
mkdir -p logs

echo "started $(date) on $(hostname)"
echo "ERA5_DATA_ROOT=$ERA5_DATA_ROOT  ERA5_SUBSET=$ERA5_SUBSET"
python -u -c "import paths; print('resolved data root:', paths.DATA_ROOT)"

python -u read_in/midnight_subset/read_inera5.py
python -u read_in/midnight_subset/data_preprocessing.py "$@"

echo "finished $(date)"
ls -lh "$ERA5_DATA_ROOT/midnight_subset/CLARA_ERA5_merged.parquet"

#!/bin/bash
#SBATCH --job-name=era5_preprocess
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/era5_preprocess_%j.out
#SBATCH --error=logs/era5_preprocess_%j.err
#SBATCH --mail-type=END,FAIL
# Euler mails your registered ETH address. To use another, uncomment:
##SBATCH --mail-user=you@ethz.ch

# Builds the modelling input for the spatial subset from the fetched GRIBs:
#   1. CLARA.pkl        -> spatial_subset/CLARA_matched.pkl   (Italy, 2020)
#   2. monthly GRIBs    -> inventory + surface-pressure map (sanity check)
#   3. both             -> spatial_subset/CLARA_ERA5_merged.parquet
# Single-threaded; memory is sized for one month of ERA5 held at a time.
# Extra arguments are passed to step 3, e.g. a smoke test:
#     sbatch euler/preprocess.sbatch --months 01 --limit-hours 48

set -euo pipefail
source euler/env.sh
mkdir -p logs

echo "started $(date) on $(hostname)"
echo "ERA5_DATA_ROOT=$ERA5_DATA_ROOT  ERA5_SUBSET=$ERA5_SUBSET"
python -u read_in/spatial_subset/read_inclara.py
python -u read_in/spatial_subset/read_inera5.py
python -u read_in/spatial_subset/data_preprocessing.py "$@"
echo "finished $(date)"
ls -lh "$ERA5_DATA_ROOT/spatial_subset/"

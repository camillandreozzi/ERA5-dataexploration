#!/bin/bash
#SBATCH --job-name=era5_fetch
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/era5_fetch_%j.out
#SBATCH --error=logs/era5_fetch_%j.err
#SBATCH --mail-type=END,FAIL
# Euler mails your registered ETH address. To use another, uncomment:
##SBATCH --mail-user=you@ethz.ch

# Downloads the six monthly ERA5 requests. Mostly waiting on the CDS queue,
# so this asks for one cheap core and a long wall clock rather than resources.
# The fetch script skips months already present, so requeue it after a timeout
# and it resumes instead of re-downloading.

set -euo pipefail
source euler/env.sh
mkdir -p logs

echo "started $(date) on $(hostname)"
echo "ERA5_DATA_ROOT=$ERA5_DATA_ROOT"
python -u -c "import paths; print('resolved data root:', paths.DATA_ROOT)"
# -u = unbuffered, so progress appears in the log live instead of
# only when the job ends.
python -u read_in/spatial_subset/data_fetch.py
echo "finished $(date)"
df -h "$ERA5_DATA_ROOT" | tail -1

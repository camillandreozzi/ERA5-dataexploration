#!/bin/bash
#SBATCH --job-name=era5_fetch_midnight
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/era5_fetch_midnight_%j.out
#SBATCH --error=logs/era5_fetch_midnight_%j.err
#SBATCH --mail-type=END,FAIL

# The midnight subset's 103 CDS requests (one per month x UTC hour where CLARA
# observed near local midnight). Like fetch_era5.sbatch this is almost entirely
# CDS queue wait, so it asks for one cheap core and a long wall clock. Requests
# already on disk are skipped, so requeue after a timeout and it resumes.
#
# Unlike the spatial subset this needs CLARA_matched.pkl first: the request
# boxes are built from the CLARA selection, not from a fixed area. read_inclara
# builds it from CLARA.pkl and is cheap, so it runs here rather than being a
# separate job.

set -euo pipefail

# env.sh defaults ERA5_SUBSET to spatial_subset; override before sourcing so
# paths.py resolves data/midnight_subset and the blocked-CV overrides that are
# specific to the spatial subset do not apply.
export ERA5_SUBSET=midnight_subset
source euler/env.sh
mkdir -p logs

echo "started $(date) on $(hostname)"
echo "ERA5_SUBSET=$ERA5_SUBSET"
python -u -c "import paths; print('resolved data root:', paths.DATA_ROOT)"

# Selection + request plan. Skipped when CLARA_matched.pkl is already there,
# because it also writes plots and there is no need to redo them on a requeue.
if [ ! -f "$ERA5_DATA_ROOT/midnight_subset/CLARA_matched.pkl" ]; then
    echo "building CLARA_matched.pkl and the request plan"
    python -u read_in/midnight_subset/read_inclara.py
else
    echo "CLARA_matched.pkl already present, skipping read_inclara"
fi

python -u read_in/midnight_subset/data_fetch.py

echo "finished $(date)"
ls "$ERA5_DATA_ROOT/midnight_subset"/*.grib 2>/dev/null | wc -l | xargs echo "GRIB files on disk:"
df -h "$ERA5_DATA_ROOT" | tail -1

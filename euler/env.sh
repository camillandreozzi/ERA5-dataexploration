#!/bin/bash
# Shared Euler environment. Source this, do not execute it:
#     source euler/env.sh
#
# Storage note: this points at /cluster/home, which is backed up but quota'd
# (~45 GB, 450k inodes -- check with `lquota`). The ERA5 GRIB files are large
# and cfgrib writes one .idx per file, so both limits are reachable. To move
# the data without touching any Python, change ERA5_DATA_ROOT below, e.g. to
#     /cluster/work/<group>/$USER/era5/data   (permanent, needs allocation)
#     $SCRATCH/era5/data                      (2.5 TB, PURGED after 15 days)

# Adjust these to whatever `module avail python` / `module avail eccodes`
# report on Euler; the names below are a starting guess.
# eth_proxy is REQUIRED: Euler compute nodes have no direct internet, so
# any job talking to Copernicus fails with "Connection refused" without it.
ERA5_MODULES="${ERA5_MODULES:-stack/2024-06 python/3.11.6 eccodes/2.25.0 eth_proxy}"
for m in $ERA5_MODULES; do
    if ! module load "$m" 2>/dev/null; then
        echo "WARNING: could not load module '$m' -- check \`module avail ${m%%/*}\`" >&2
    fi
done

export ERA5_PROJECT_ROOT="${ERA5_PROJECT_ROOT:-$HOME/ERA5-dataexploration}"
export ERA5_DATA_ROOT="${ERA5_DATA_ROOT:-$HOME/era5-store/data}"
export ERA5_RESULTS_ROOT="${ERA5_RESULTS_ROOT:-$HOME/era5-store/results}"

# Which data subset every script reads and writes (see paths.py). On Euler it
# is the Italy / 2020 spatial subset downloaded by fetch_era5.sbatch.
export ERA5_SUBSET="${ERA5_SUBSET:-spatial_subset}"

# Blocked-CV settings (modelling/fold_config.py). The spatial subset has only
# ~200 hourly CLARA cells, so it gets fewer space folds and lower minimum fold
# sizes than the temporal-subset defaults (4 / 4 / 50 / 5).
if [ "$ERA5_SUBSET" = "spatial_subset" ]; then
    export ERA5_N_SPACE_FOLDS="${ERA5_N_SPACE_FOLDS:-3}"
    export ERA5_N_TIME_FOLDS="${ERA5_N_TIME_FOLDS:-4}"
    export ERA5_MIN_TRAIN_ROWS="${ERA5_MIN_TRAIN_ROWS:-30}"
    export ERA5_MIN_TEST_ROWS="${ERA5_MIN_TEST_ROWS:-3}"
fi

# ${VAR:-} so this survives `set -u` on an account where PYTHONPATH is unset.
export PYTHONPATH="$ERA5_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Keep pip/matplotlib/xdg caches off the tiny default locations.
export PIP_CACHE_DIR="$ERA5_DATA_ROOT/../.cache/pip"
export MPLCONFIGDIR="$ERA5_DATA_ROOT/../.cache/matplotlib"

# Fallback if the eth_proxy module did not load for some reason.
if [ -z "${http_proxy:-}" ]; then
    export http_proxy="http://proxy.ethz.ch:3128"
    export https_proxy="http://proxy.ethz.ch:3128"
fi

VENV="$ERA5_PROJECT_ROOT/.venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi

mkdir -p "$ERA5_DATA_ROOT" "$ERA5_RESULTS_ROOT" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

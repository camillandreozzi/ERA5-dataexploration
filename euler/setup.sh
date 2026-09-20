#!/bin/bash
# One-time setup on Euler. Run from the project root ON THE CLUSTER:
#     bash euler/setup.sh
set -euo pipefail

source "$(dirname "$0")/env.sh"

echo "project : $ERA5_PROJECT_ROOT"
echo "data    : $ERA5_DATA_ROOT"
echo "results : $ERA5_RESULTS_ROOT"
echo

if [ ! -d "$ERA5_PROJECT_ROOT/.venv" ]; then
    echo "creating venv"
    python -m venv "$ERA5_PROJECT_ROOT/.venv"
fi
source "$ERA5_PROJECT_ROOT/.venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ERA5_PROJECT_ROOT/euler/requirements.txt"

if [ ! -f "$HOME/.cdsapirc" ]; then
    echo
    echo "WARNING: no ~/.cdsapirc on this machine."
    echo "Copy it from your laptop before running the fetch job:"
    echo "  scp ~/.cdsapirc candreozzi@euler.ethz.ch:~/.cdsapirc"
    echo "  ssh candreozzi@euler.ethz.ch chmod 600 ~/.cdsapirc"
fi

echo
echo "quota check:"
lquota || true

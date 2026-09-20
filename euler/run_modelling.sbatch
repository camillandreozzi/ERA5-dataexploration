#!/bin/bash
#SBATCH --job-name=era5_model
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/era5_model_%j.out
#SBATCH --error=logs/era5_model_%j.err
#SBATCH --mail-type=END,FAIL
# Euler mails your registered ETH address. To use another, uncomment:
##SBATCH --mail-user=you@ethz.ch

# Random forest / kriging runs. Pass the script to run as the first argument:
#     sbatch euler/run_modelling.sbatch modelling/benchmark_rf.py
#     sbatch euler/run_modelling.sbatch modelling/rf_stkriging.py --grid-kriging-backend loop

set -euo pipefail
source euler/env.sh
mkdir -p logs

# scikit-learn picks up the cores SLURM actually granted, not the node's total.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SKLEARN_N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

echo "started $(date) on $(hostname) with ${SLURM_CPUS_PER_TASK:-1} cpus"
python -u "$@"
echo "finished $(date)"

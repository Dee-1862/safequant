#!/bin/bash
#SBATCH --job-name=sq_refit_phi3
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/refit_phi3_%j.out
#SBATCH --error=scripts/logs/refit_phi3_%j.err
source "$(dirname "$0")/../_env.sh"


echo "=== probe refit control Phi-3 | $(date) ==="
python -u -m experiments.boundary.probe_refit_control \
  --model phi3 --n_calib 200
echo "=== done | $(date) ==="

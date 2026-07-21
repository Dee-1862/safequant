#!/bin/bash
#SBATCH --job-name=sq_cosmed_restore
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/cosmed_restore_%j.out
#SBATCH --error=scripts/logs/cosmed_restore_%j.err
source "$(dirname "$0")/../_env.sh"

# RQ1 mediation control: cosmed under quant / +read / +write / +both restore.
echo "=== cosmed after restoration | ${EXP_MODEL:-llama7b}/aqlm2 | $(date) ==="
python -u -m experiments.signal_geometry.cosmed_after_restore \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 --n_calib 200
echo "=== done | $(date) ==="

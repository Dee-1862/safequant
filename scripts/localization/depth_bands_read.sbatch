#!/bin/bash
#SBATCH --job-name=sq_depth_read
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/depth_read_%j.out
#SBATCH --error=scripts/logs/depth_read_%j.err
source "$(dirname "$0")/../_env.sh"

# Table 4 : early/middle/late depth bands (read projections).


echo "=== depth bands read | $(date) ==="
python -u -m experiments.localization.depth_bands \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 --projections read \
  --n_eval 200 --n_reasoning 50
echo "=== done | $(date) ==="

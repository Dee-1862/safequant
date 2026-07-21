#!/bin/bash
#SBATCH --job-name=sq_perproj
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/perproj_llama_%j.out
#SBATCH --error=scripts/logs/perproj_llama_%j.err
source "$(dirname "$0")/../_env.sh"

# Table 3 : one-projection-at-a-time restore.


echo "=== per-projection restore Llama | $(date) ==="
python -u -m experiments.localization.per_projection \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 --n_eval 200 --n_reasoning 50
echo "=== done | $(date) ==="

#!/bin/bash
#SBATCH --job-name=sq_perproj_mid
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/perproj_mid_llama_%j.out
#SBATCH --error=scripts/logs/perproj_mid_llama_%j.err
source "$(dirname "$0")/../_env.sh"

# Middle-layer-only projection ablation (scripts_old/add4).


echo "=== per-projection middle band Llama | $(date) ==="
python -u -m experiments.localization.per_projection \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 --n_eval 200 --n_reasoning 50 \
  --layer_lo ${MID_LO:-11} --layer_hi ${MID_HI:-22}
echo "=== done | $(date) ==="

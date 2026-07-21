#!/bin/bash
#SBATCH --job-name=sq_headline
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/headline_llama_%j.out
#SBATCH --error=scripts/logs/headline_llama_%j.err
source "$(dirname "$0")/../_env.sh"

# Table 5 : FP/quant/restore cells + McNemar + bootstrap.


echo "=== headline eval Llama | $(date) ==="
python -u -m experiments.headline.headline_eval \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 \
  --n_eval 200 --n_reasoning 100 --n_boot 2000
echo "=== done | $(date) ==="

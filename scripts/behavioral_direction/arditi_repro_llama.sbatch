#!/bin/bash
#SBATCH --job-name=sq_arditi
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/arditi_llama_%j.out
#SBATCH --error=scripts/logs/arditi_llama_%j.err
source "$(dirname "$0")/../_env.sh"

# Sec. 5 : behavioral direction (logit refusal-score selection).


echo "=== Arditi logit-selection repro | $(date) ==="
python -u -m experiments.behavioral_direction.arditi_repro \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2
python -u -m experiments.behavioral_direction.arditi_repro \
  --model ${EXP_MODEL:-llama7b} --quantizer fp32
echo "=== done | $(date) ==="

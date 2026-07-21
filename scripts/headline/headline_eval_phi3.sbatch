#!/bin/bash
#SBATCH --job-name=sq_headline_phi3
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/headline_phi3_%j.out
#SBATCH --error=scripts/logs/headline_phi3_%j.err
source "$(dirname "$0")/../_env.sh"


echo "=== headline eval Phi-3 | $(date) ==="
python -u -m experiments.headline.headline_eval \
  --model phi3 --quantizer aqlm2 \
  --n_eval 200 --n_reasoning 100 --n_boot 2000
echo "=== done | $(date) ==="

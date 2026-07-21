#!/bin/bash
#SBATCH --job-name=sq_f3a_phi3
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/f3a_phi3_%j.out
#SBATCH --error=scripts/logs/f3a_phi3_%j.err
source "$(dirname "$0")/../_env.sh"


echo "=== restoration sweep Phi-3 | $(date) ==="
python -u -m experiments.regime_sweep.restoration_sweep \
  --model phi3 --quantizer aqlm2 --n_eval 200 --n_reasoning 50
echo "=== done | $(date) ==="

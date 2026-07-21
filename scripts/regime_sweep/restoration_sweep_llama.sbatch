#!/bin/bash
#SBATCH --job-name=sq_f3a_llama
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/f3a_llama_%j.out
#SBATCH --error=scripts/logs/f3a_llama_%j.err
source "$(dirname "$0")/../_env.sh"

# Table 2 : read/write/both restoration matrix.


echo "=== restoration sweep Llama | $(date) ==="
python -u -m experiments.regime_sweep.restoration_sweep \
  --model llama7b --quantizer aqlm2 --n_eval 200 --n_reasoning 50
echo "=== done | $(date) ==="

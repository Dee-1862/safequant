#!/bin/bash
#SBATCH --job-name=sq_parammatch
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/parammatch_%j.out
#SBATCH --error=scripts/logs/parammatch_%j.err
source "$(dirname "$0")/../_env.sh"

# RQ3 control: read subset matched to the WRITE param budget vs full write.
echo "=== param-matched restore | ${EXP_MODEL:-llama7b}/aqlm2 | $(date) ==="
python -u -m experiments.regime_sweep.param_matched_restore \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 --n_eval 200 --n_reasoning 50 --seeds 3
echo "=== done | $(date) ==="

#!/bin/bash
#SBATCH --job-name=sq_actswap
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/act_swap_%j.out
#SBATCH --error=scripts/logs/act_swap_%j.err
source "$(dirname "$0")/../_env.sh"

# Appendix : swap quantized activations into FP32 at select layers.


echo "=== activation swap | $(date) ==="
python -u -m experiments.boundary.activation_swap \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 \
  --swap_layers ${SWAP_LAYERS:-4,11,16,22,29} --n_eval 200 --max_new_tokens 256
echo "=== done | $(date) ==="

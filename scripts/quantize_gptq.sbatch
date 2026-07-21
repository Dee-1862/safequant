#!/bin/bash
#SBATCH --job-name=quantize_gptq
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/quantize_gptq_%j.out
#SBATCH --error=scripts/logs/quantize_gptq_%j.err
source "$(dirname "$0")/../_env.sh"

# Reproducible GPTQ quantization of one base model via GPTQModel.
# Submit one per model to run them in parallel:
#   sbatch scripts/quantize_gptq.sbatch llama7b
#   sbatch scripts/quantize_gptq.sbatch phi3
#   sbatch scripts/quantize_gptq.sbatch mistral7b-v02
#   sbatch scripts/quantize_gptq.sbatch qwen7b
# Optional 2nd/3rd args override bits/group_size: sbatch ... phi3 4 128

set -euo pipefail

MODEL="${1:-llama7b}"
BITS="${2:-4}"
GROUP="${3:-128}"

echo "=== GPTQ quantize ${MODEL} (bits=${BITS}, group=${GROUP}) | $(date) ==="
python -u scripts/quantize_gptq.py \
  --model "${MODEL}" --bits "${BITS}" --group_size "${GROUP}" \
  --n_calib 128 --seed 42
echo "=== done ${MODEL} | $(date) ==="

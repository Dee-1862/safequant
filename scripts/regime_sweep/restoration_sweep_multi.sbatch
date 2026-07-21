#!/bin/bash
#SBATCH --job-name=sq_f3a_multi
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=16:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/f3a_multi_%j.out
#SBATCH --error=scripts/logs/f3a_multi_%j.err
source "$(dirname "$0")/../_env.sh"

# From scripts_old/f3a_static_restoration_*.sbatch (condensed).


echo "=== restoration sweep multi-model | $(date) ==="
for MODEL in ${SWEEP_MODELS:-llama7b mistral7b-v02 qwen7b}; do
  for QUANTIZER in ${SWEEP_QUANTIZERS:-aqlm2 awq gptq4 gptq3 gptq2}; do
    echo "---- ${MODEL} / ${QUANTIZER} ----"
    python -u -m experiments.regime_sweep.restoration_sweep \
      --model ${MODEL} --quantizer ${QUANTIZER} --n_eval 200 \
      --n_reasoning ${N_REASONING:-100} \
      || echo "FAILED ${MODEL}/${QUANTIZER}"
  done
done
echo "=== done | $(date) ==="

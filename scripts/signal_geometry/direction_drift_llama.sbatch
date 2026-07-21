#!/bin/bash
#SBATCH --job-name=sq_f1_drift
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/f1_drift_llama_%j.out
#SBATCH --error=scripts/logs/f1_drift_llama_%j.err
source "$(dirname "$0")/../_env.sh"

# Table 1 : refusal-direction cos-sim drift across quantizers (RESULTS.md).


echo "=== direction drift multi-model/quant | $(date) ==="
for MODEL in ${SWEEP_MODELS:-llama7b}; do
  for QUANTIZER in ${SWEEP_QUANTIZERS:-gptq4 awq aqlm2 gptq3 gptq2}; do
    echo "---- ${MODEL} / ${QUANTIZER} ----"
    python -u -m experiments.signal_geometry.direction_drift \
      --model ${MODEL} --quantizer ${QUANTIZER} --n_calib 200 || echo "FAILED ${MODEL}/${QUANTIZER}"
  done
done
echo "=== done | $(date) ==="

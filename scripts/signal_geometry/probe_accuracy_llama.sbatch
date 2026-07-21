#!/bin/bash
#SBATCH --job-name=sq_probe_acc
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/probe_acc_llama_%j.out
#SBATCH --error=scripts/logs/probe_acc_llama_%j.err
source "$(dirname "$0")/../_env.sh"


echo "=== probe accuracy multi-model/quant | $(date) ==="
for MODEL in ${SWEEP_MODELS:-llama7b}; do
  for QUANTIZER in ${SWEEP_QUANTIZERS:-aqlm2 awq gptq4 gptq3 gptq2}; do
    echo "---- ${MODEL} / ${QUANTIZER} ----"
    python -u -m experiments.signal_geometry.probe_accuracy \
      --model ${MODEL} --quantizer ${QUANTIZER} --n_calib 200 --probe_type linear \
      || echo "FAILED ${MODEL}/${QUANTIZER}"
  done
done
echo "=== done | $(date) ==="

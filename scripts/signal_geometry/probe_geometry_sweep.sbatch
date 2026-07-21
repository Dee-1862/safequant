#!/bin/bash
#SBATCH --job-name=sq_probe_geo
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/probe_geo_sweep_%j.out
#SBATCH --error=scripts/logs/probe_geo_sweep_%j.err
source "$(dirname "$0")/../_env.sh"

# Extended probe-geometry sweep (from scripts_old/probe_geometry_*.sbatch).


echo "=== probe rotation multi-model sweep | $(date) ==="
for MODEL in ${SWEEP_MODELS:-llama7b mistral7b-v02 qwen7b}; do
  for QUANTIZER in ${SWEEP_QUANTIZERS:-aqlm2 awq gptq4 gptq3 gptq2}; do
    echo "---- ${MODEL} / ${QUANTIZER} ----"
    python -u -m experiments.signal_geometry.probe_rotation \
      --model ${MODEL} --quantizer ${QUANTIZER} --n_calib 200 \
      || echo "FAILED ${MODEL}/${QUANTIZER}"
  done
done
echo "=== done | $(date) ==="

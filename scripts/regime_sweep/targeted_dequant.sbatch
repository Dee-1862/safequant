#!/bin/bash
#SBATCH --job-name=sq_tdequant
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/tdequant_%j.out
#SBATCH --error=scripts/logs/tdequant_%j.err
source "$(dirname "$0")/../_env.sh"

# Section 8 -- targeted 4-bit (AWQ) restoration into the 2-bit (AQLM) cell.
# Two comparable read-side conditions so we can see if the localized middle-band
# repair matches the full read-side one:
#   1) read-side, ALL layers      -> read_awq4_all
#   2) read-side, MIDDLE band 11-22 -> read_awq4_L11-22
MODEL="${EXP_MODEL:-llama7b}"

echo "=== targeted 4-bit dequant | ${MODEL} | $(date) ==="
echo "---- read-side / all layers ----"
python -u -m experiments.regime_sweep.targeted_dequant \
  --model ${MODEL} --quantizer aqlm2 --source_quant awq --role read \
  --n_eval 200 --n_reasoning 50 || echo "FAILED read/all"
echo "---- read-side / middle band (11-22) ----"
python -u -m experiments.regime_sweep.targeted_dequant \
  --model ${MODEL} --quantizer aqlm2 --source_quant awq --role read \
  --layer_lo ${MID_LO:-11} --layer_hi ${MID_HI:-22} --n_eval 200 --n_reasoning 50 || echo "FAILED read/middle"
echo "=== done | $(date) ==="

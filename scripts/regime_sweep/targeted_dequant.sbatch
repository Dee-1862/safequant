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
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Section 8 -- targeted mixed-precision repair: dequantize the role projections
# of a damaged cell (TARGET) to a higher-precision SOURCE quantizer. Runs both
# ALL-layers and MIDDLE-band [MID_LO,MID_HI) conditions. Override via env, e.g.
#   TARGET=aqlm2 SOURCE=gptq4    or    TARGET=gptq2 SOURCE=awq
MODEL="${EXP_MODEL:-llama7b}"
TARGET="${TARGET:-aqlm2}"
SOURCE="${SOURCE:-awq}"
ROLE="${ROLE:-read}"
NR="${N_REASONING:-100}"

echo "=== targeted dequant | ${MODEL} target=${TARGET} source=${SOURCE} role=${ROLE} nr=${NR} | $(date) ==="
echo "---- ${ROLE} / all layers ----"
python -u -m experiments.regime_sweep.targeted_dequant \
  --model ${MODEL} --quantizer ${TARGET} --source_quant ${SOURCE} --role ${ROLE} \
  --n_eval 200 --n_reasoning ${NR} || echo "FAILED ${TARGET}/${SOURCE}/all"
echo "---- ${ROLE} / middle band (${MID_LO:-11}-${MID_HI:-22}) ----"
python -u -m experiments.regime_sweep.targeted_dequant \
  --model ${MODEL} --quantizer ${TARGET} --source_quant ${SOURCE} --role ${ROLE} \
  --layer_lo ${MID_LO:-11} --layer_hi ${MID_HI:-22} --n_eval 200 --n_reasoning ${NR} \
  || echo "FAILED ${TARGET}/${SOURCE}/middle"
echo "=== done | $(date) ==="

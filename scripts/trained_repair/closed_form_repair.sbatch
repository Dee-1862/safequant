#!/bin/bash
#SBATCH --job-name=sq_cfrepair
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=1:30:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/cfrepair_%j.out
#SBATCH --error=scripts/logs/cfrepair_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Training-free closed-form (SVD-residual) repair -- the untrained counterpart to
# LR-QAT, matched at the same INT-b budget. No C4, no gradient steps -> minutes.
# Submit with CFR_ROLE in {read,write,both} x CFR_BITS in {2,3,4}. Optional
# depth-band via CFR_MID_LO / CFR_MID_HI (e.g. 11 / 22) -> whole-model bits ~2.3.
MODEL="${EXP_MODEL:-llama7b}"
ROLE="${CFR_ROLE:-read}"
BITS="${CFR_BITS:-3}"
NR="${N_REASONING:-100}"
NX="${N_XSTEST:-250}"

BAND_ARGS=""
if [ -n "${CFR_MID_LO}" ] && [ -n "${CFR_MID_HI}" ]; then
  BAND_ARGS="--layer_lo ${CFR_MID_LO} --layer_hi ${CFR_MID_HI}"
fi

echo "=== closed-form repair | ${MODEL}/aqlm2 role=${ROLE} int${BITS} band='${BAND_ARGS:-all}' | $(date) ==="
python -u -m experiments.trained_repair.closed_form_repair \
  --model ${MODEL} --quantizer aqlm2 --role ${ROLE} --n_bits ${BITS} \
  --rank 32 --group_size 128 --n_eval 200 \
  --n_reasoning ${NR} --n_xstest ${NX} ${BAND_ARGS}
echo "=== done | $(date) ==="

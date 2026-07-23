#!/bin/bash
#SBATCH --job-name=sq_lrqat
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/lrqat_llama_%j.out
#SBATCH --error=scripts/logs/lrqat_llama_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Table 7 -- ONE matched-budget LR-QAT cell (role x n_bits) per job, so the four
# cells run in PARALLEL instead of 4x sequential in one 24h job (which timed out).
# Submit 4 times with LRQAT_ROLE in {read,write} x LRQAT_BITS in {2,4}.
MODEL="${EXP_MODEL:-llama7b}"
ROLE="${LRQAT_ROLE:-read}"
BITS="${LRQAT_BITS:-2}"

# Optional depth-band hybrid: inject INT-b adapters only on layers [MID_LO,MID_HI)
# so most role matrices stay at AQLM-2 -> whole-model bits sit just above 2.
# Enable by exporting LRQAT_MID_LO / LRQAT_MID_HI (e.g. 11 / 22); else all layers.
BAND_ARGS=""
if [ -n "${LRQAT_MID_LO}" ] && [ -n "${LRQAT_MID_HI}" ]; then
  BAND_ARGS="--layer_lo ${LRQAT_MID_LO} --layer_hi ${LRQAT_MID_HI}"
fi

echo "=== LR-QAT | ${MODEL}/aqlm2 role=${ROLE} int${BITS} band='${BAND_ARGS:-all}' | $(date) ==="
# ARC/XSTest n default to the paper-standard 100/250 (override via env) so LR-QAT
# evals sit on the same n as tab:f3a/headline. n_eval (ASR) stays 200 everywhere.
BUDGET_ARGS="--steps 3000 --lr 1e-4 --rank 32 --alpha 64 \
             --seq_len 512 --grad_accum 8 --n_train_docs 4000 \
             --group_size 128 --n_eval 200 \
             --n_reasoning ${N_REASONING:-100} --n_xstest ${N_XSTEST:-250}"

python -u -m experiments.trained_repair.lrqat \
  --model ${MODEL} --quantizer aqlm2 --role ${ROLE} --n_bits ${BITS} ${BUDGET_ARGS} ${BAND_ARGS}
echo "=== done | $(date) ==="

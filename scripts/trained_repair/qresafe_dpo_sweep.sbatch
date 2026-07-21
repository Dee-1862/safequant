#!/bin/bash
#SBATCH --job-name=sq_qresafe_sw
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/qresafe_sweep_%j.out
#SBATCH --error=scripts/logs/qresafe_sweep_%j.err
source "$(dirname "$0")/../_env.sh"

# Replaces dozens of scripts_old/qresafe_*_r*.sbatch one-offs.


echo "=== Q-Resafe DPO rank/mask sweep | $(date) ==="
# Compact sweep covering paper Table 6-7 cells. Override via env:
#   SWEEP_MASKS="full" SWEEP_RANKS="128 256 512 1024 2048"
# --tag r${R} makes each rank write a DISTINCT file (variant=<mask>_r<R>);
# without it every rank overwrote <mask>.json (why 512/2048 went missing).
for MASK in ${SWEEP_MASKS:-read write full}; do
  for R in ${SWEEP_RANKS:-128 256 512 1024 2048}; do
    ALPHA=$((R * 2))
    echo "---- mask=${MASK} r=${R} alpha=${ALPHA} ----"
    python -u -m experiments.trained_repair.qresafe_dpo \
      --mask ${MASK} --r ${R} --alpha ${ALPHA} --tag r${R} \
      --beta 0.01 --epochs 1 --lr 5e-7 \
      --n_eval 200 --opt8bit --seed 42 \
      || echo "FAILED ${MASK}/r${R}"
  done
done
echo "=== done | $(date) ==="

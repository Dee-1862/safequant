#!/bin/bash
#SBATCH --job-name=sq_qresafe
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/qresafe_dpo_%j.out
#SBATCH --error=scripts/logs/qresafe_dpo_%j.err
source "$(dirname "$0")/../_env.sh"

# Submit: sbatch scripts/trained_repair/qresafe_dpo.sbatch read|write|full


MASK="${1:-read}"
echo "=== Q-Resafe DPO | mask=${MASK} | $(date) ==="
python -u -m experiments.trained_repair.qresafe_dpo \
  --mask "${MASK}" --model llama7b --quantizer aqlm2 --n_eval 200
echo "=== done | $(date) ==="

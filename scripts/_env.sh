#!/bin/bash
# Shared environment for SafeQuant sbatch jobs. Loads .env from repo root.
#   source "$(dirname "$0")/../_env.sh"

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/load_env.sh"

export REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export QM_ROOT="${QM_ROOT:-${HOME}/quantized_models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/opt/packages/cuda/v12.4.0}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export SQ_LOG_DIR="${SQ_LOG_DIR:-${REPO_ROOT}/scripts/logs}"

mkdir -p "${SQ_LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${QM_ROOT}"

if [[ -f /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
  module load gcc/10.2.0 2>/dev/null || true
fi

cd "${REPO_ROOT}"

echo "=== SafeQuant | job=${SLURM_JOB_ID:-local} | $(date) ==="
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  HF_HOME=${HF_HOME}"
echo "  QM_ROOT=${QM_ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

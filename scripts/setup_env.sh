#!/bin/bash
# One-time bootstrap: pip install + create cache dirs. Requires .env to be configured.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/load_env.sh"

if [[ -f /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export QM_ROOT="${QM_ROOT:-${HOME}/quantized_models}"
mkdir -p "${HF_HOME}" "${HF_HOME}/datasets" "${QM_ROOT}"

pip install -q -r requirements.txt

echo "HF_HOME=${HF_HOME}"
echo "QM_ROOT=${QM_ROOT}"
echo "Done. Next: huggingface-cli login && bash scripts/fetch_checkpoints.sh"

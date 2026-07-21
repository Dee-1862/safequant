#!/bin/bash
# scripts/quantize_gptq_all.sh

# Submit AutoGPTQ GPTQ quantization (bits 4/3/2, group 128) for every study model, each as its own SLURM job in the sq_autogptq env. Run from a LOGIN node:
#   bash scripts/quantize_gptq_all.sh


# Override any axis via env vars, e.g. redo just one model / bit-width:
#   MODELS="qwen7b" BITS="4" bash scripts/quantize_gptq_all.sh
#   MODELS="mistral7b-v02 phi3" bash scripts/quantize_gptq_all.sh


# Each job writes <ocean>/quantized_models/<model>-gptq<bits>bit-g<group>, and prints the config.QUANTIZED_CHECKPOINTS line to register.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/load_env.sh"

if [[ -z "${AGENV:-}" ]]; then
  echo "Set AGENV in .env (path to sq_autogptq conda env)." >&2
  exit 1
fi

# NOTE on Qwen: this uses Qwen2.5-7B (qwen7b).
MODELS="${MODELS:-llama7b qwen7b mistral7b-v02 phi3}"
BITS="${BITS:-4 3 2}"
GROUP="${GROUP:-128}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [ ! -d "$AGENV" ]; then
  echo "ERROR: quant env not found: $AGENV" >&2
  echo "Set AGENV in .env (path to sq_autogptq conda env)." >&2
  exit 1
fi

echo "env    = $AGENV"
echo "models = $MODELS"
echo "bits   = $BITS    group = $GROUP"
echo "-----------------------------------------------------"
n=0
for m in $MODELS; do
  for b in $BITS; do
    jid=$(sbatch --parsable --account="${SLURM_ACCOUNT:-YOUR_ALLOCATION}" \
          --export=ALL,CONDA_ENV="$AGENV" \
          scripts/quantize_gptq.sbatch "$m" "$b" "$GROUP")
    printf "  %-16s gptq%s  -> job %s\n" "$m" "$b" "$jid"
    n=$((n + 1))
  done
done
echo "-----------------------------------------------------"
echo "submitted $n jobs.  watch: squeue -u \$USER"
echo "outputs -> <ocean>/quantized_models/<model>-gptq<bits>bit-g${GROUP}"

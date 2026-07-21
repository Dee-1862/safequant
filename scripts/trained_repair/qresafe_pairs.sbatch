#!/bin/bash
#SBATCH --job-name=sq_pairs
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/qresafe_pairs_%j.out
#SBATCH --error=scripts/logs/qresafe_pairs_%j.err
source "$(dirname "$0")/../_env.sh"

# Tables 6-7 data : build pairs before DPO.


echo "=== Q-Resafe displacement pairs | $(date) ==="
python -u -m experiments.trained_repair.qresafe_pairs \
  --model llama7b --quantizer aqlm2 \
  --prompt_source wildjailbreak --wj_config eval \
  --n_prompts 1000 --max_new_tokens 128 \
  --output outputs/trained_repair/qresafe_pairs_llama_aqlm2_wj.json
echo "=== done | $(date) ==="

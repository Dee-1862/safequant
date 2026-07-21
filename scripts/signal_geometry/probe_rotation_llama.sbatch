#!/bin/bash
#SBATCH --job-name=sq_probe_rot
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/probe_rot_llama_%j.out
#SBATCH --error=scripts/logs/probe_rot_llama_%j.err
source "$(dirname "$0")/../_env.sh"


echo "=== probe rotation Llama | $(date) ==="
python -u -m experiments.signal_geometry.probe_rotation \
  --model llama7b --quantizer aqlm2 --n_calib 200
echo "=== done | $(date) ==="

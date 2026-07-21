# The Refusal Signal Survives Quantization but Goes Unconsumed

Mechanistic study of how quantization affects LLM safety. Paper-facing code lives in `core/` + `experiments/`.

Randomness is pinned to seed **42**.

## Prerequisites

- **GPU** with enough VRAM for 7-8B models (A100/H100 80GB recommended for full grid; smaller GPUs can run single experiments).
- **Python 3.10+** and **CUDA** PyTorch (`requirements.txt`).
- **Disk:** ~200 GB free for base + AWQ + AQLM checkpoints under `QM_ROOT` (GPTQ adds ~20 GB per model x bit width).
- **HuggingFace account** with access to gated models (Llama, Mistral). Run `huggingface-cli login` before downloading.

## Setup (any machine)

### 1. Clone and install Python deps

```bash
git clone https://github.com/Dee-1862/safequant.git
cd safequant
cp .gitignore.example .gitignore
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

On Bridges-2 (after `.env` is configured):

```bash
bash scripts/setup_env.sh
```

### 2. Configure paths (`.env`)

Copy the template and edit paths for your machine. Python and shell scripts load `.env` from the repo root automatically.

```bash
cp .env.example .env
# edit .env
```

| Variable | What it is |
|----------|------------|
| `HF_HOME` | Where HuggingFace downloads models (~200 GB for the paper grid) |
| `QM_ROOT` | Where checkpoints live (`base/`, `awq/`, `aqlm/`, `gptq/` subdirs) |
| `OCEAN_ALLOCATION` | Bridges only: folder name under `/ocean/projects/` (from `accounts`) |
| `SLURM_ACCOUNT` | Bridges only: your Slurm account for `sbatch` |

**Local machine:** set `REPO_ROOT`, `HF_HOME`, and `QM_ROOT` to directories on a large disk (see commented block in `.env.example`).

**Bridges-2:** on a login node, run `accounts` for `SLURM_ACCOUNT` and your Ocean allocation folder name, then set `OCEAN_ALLOCATION` and the paths in `.env.example`'s Bridges block. **Do not commit `.env`.**

### 3. Download checkpoints (one-time)

```bash
huggingface-cli login
bash scripts/fetch_checkpoints.sh
```

This fills `$QM_ROOT/base/`, `awq/`, and `aqlm/` from HuggingFace. GPTQ checkpoints are self-generated:

```bash
python scripts/quantize_gptq.py --model llama7b --bits 4
python scripts/quantize_gptq.py --model llama7b --bits 3
python scripts/quantize_gptq.py --model llama7b --bits 2
# repeat for qwen7b, mistral7b-v02, phi3 as needed
```

## Reproduce paper results

Run one table at a time, or the full pipeline. Steps that already have an output JSON under `outputs/` are skipped.

```bash
python run_pipeline.py --run table1    # refusal-direction geometry
python run_pipeline.py --run table2    # restoration matrix
python run_pipeline.py --run table3    # per-projection localization
python run_pipeline.py --run table4    # depth-band localization
python run_pipeline.py --run table5    # headline eval
python run_pipeline.py --run table6    # controls
python run_pipeline.py --run table7    # Q-Resafe + LR-QAT
python run_pipeline.py --run behavioral   # Arditi behavioral ablation
python run_pipeline.py --run full      # everything above
```

Combine tables: `python run_pipeline.py --run table1 table5`

Rebuild result files from `outputs/` (no GPU):

```bash
python run_pipeline.py --results --md
```

Model aliases (example: `llama7b`, `phi3`) and checkpoint paths are in `config.py`.

## Layout

```
safequant/
  run_pipeline.py           --run tableN / full, or --results
  config.py                 model aliases, HF_HOME, QM_ROOT
  core/                     shared library
  experiments/<claim>/      paper experiments
  scripts/                  aggregators + cluster helpers
  outputs/<claim>/          result JSONs (committed or regenerated)
  tests/                    regression anchors
```

## Single experiment

```bash
python -m experiments.headline.headline_eval --model llama7b --quantizer aqlm2
```

## Optional: Bridges-2 / Slurm

Submit from the **repo root** so log paths resolve:

```bash
cd "$REPO_ROOT"
sbatch --account="$SLURM_ACCOUNT" scripts/headline/headline_eval_llama.sbatch
```

Jobs source `scripts/_env.sh`, which loads `.env` when present.

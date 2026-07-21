# Scripts (optional cluster + utilities)

**Primary entry point:** [`run_pipeline.py`](../run_pipeline.py) at the repo root. Use `python run_pipeline.py --run table1` (or `full`) or `--results --md`.

This folder holds result aggregators, checkpoint helpers, and optional Bridges-2 Slurm launchers.

| Resource | Override via |
|----------|----------------|
| Repo checkout | `REPO_ROOT` |
| HF cache | `HF_HOME` |
| Checkpoints | `QM_ROOT` |
| Conda env | `CONDA_ENV` |

Copy `.env.example` to `.env` in the repo root and edit paths before running or submitting jobs.

## Layout

```
scripts/
  _env.sh                 # shared Bridges-2 env (sourced by every job)
  setup_env.sh            # one-time login-node bootstrap
  logs/                   # Slurm stdout/stderr
  tests/                  # smoke jobs: run these before the full grid
  signal_geometry/        # Table 1
  behavioral_direction/   # Sec. 5 Arditi
  regime_sweep/           # Table 2
  localization/           # Tables 3-4
  headline/               # Table 5
  trained_repair/         # Tables 6-7
  boundary/               # Appendix controls
```

## First-time setup (login node)

```bash
bash scripts/setup_env.sh
```

## Python CLI

Same two commands as the repo root:

```bash
python run_pipeline.py --run table1
python run_pipeline.py --run full
python run_pipeline.py --results --md
```

## Optional: Bridges-2 paths

Claim-aligned sbatch jobs mirror `run_pipeline.py`. Paths:

- Jobs `cd` into `REPO_ROOT` and run `python -m experiments.<claim>.<module>` so imports resolve without `sys.path` hacks.
- Result JSONs go under `outputs/<claim>/` via `core.manifest.save_result` (except Q-Resafe pairs/DPO/LR-QAT which may still take `--output`).
- Do **not** put large HF downloads under `$HOME`; `HF_HOME` points at Ocean.
- Model aliases / HuggingFace IDs live in `config.py` (`MODELS`, `QUANTIZED_CHECKPOINTS`), not in these sbatch files.

# SafeQuant reproduction CLI: run paper tables or rebuild results from outputs/.

# Front Matter
import argparse
import os
import subprocess
import sys
from pathlib import Path

from core.manifest import RunConfig, result_path

ROOT = Path(__file__).resolve().parent

MODELS = ["llama7b", "qwen7b", "mistral7b-v02", "phi3"]
QUANTS = ["awq", "gptq4", "gptq3", "gptq2", "aqlm2"]

MOD_SCRIPT = {
    "experiments.signal_geometry.direction_drift": "experiments/signal_geometry/direction_drift.py",
    "experiments.signal_geometry.probe_accuracy": "experiments/signal_geometry/probe_accuracy.py",
    "experiments.signal_geometry.probe_rotation": "experiments/signal_geometry/probe_rotation.py",
    "experiments.regime_sweep.restoration_sweep": "experiments/regime_sweep/restoration_sweep.py",
    "experiments.localization.per_projection": "experiments/localization/per_projection.py",
    "experiments.localization.depth_bands": "experiments/localization/depth_bands.py",
    "experiments.headline.headline_eval": "experiments/headline/headline_eval.py",
    "experiments.regime_sweep.param_matched_restore": "experiments/regime_sweep/param_matched_restore.py",
    "experiments.boundary.probe_refit_control": "experiments/boundary/probe_refit_control.py",
    "experiments.signal_geometry.cosmed_after_restore": "experiments/signal_geometry/cosmed_after_restore.py",
    "experiments.trained_repair.qresafe_pairs": "experiments/trained_repair/qresafe_pairs.py",
    "experiments.trained_repair.qresafe_dpo": "experiments/trained_repair/qresafe_dpo.py",
    "experiments.trained_repair.lrqat": "experiments/trained_repair/lrqat.py",
    "experiments.boundary.probe_leakage_control": "experiments/boundary/probe_leakage_control.py",
    "experiments.behavioral_direction.arditi_repro": "experiments/behavioral_direction/arditi_repro.py",
}

RUN_TARGETS = {
    "table1": "Table 1: refusal-direction geometry (signal_geometry grid)",
    "table2": "Table 2: restoration matrix (regime_sweep)",
    "table3": "Table 3: per-projection localization (Llama AQLM-2)",
    "table4": "Table 4: depth-band localization (Llama AQLM-2)",
    "table5": "Table 5: headline eval (Llama AQLM-2)",
    "table6": "Table 6: controls (param-matched, refit, cosmed, leakage)",
    "table7": "Tables 6-7: trained repair (Q-Resafe + LR-QAT)",
    "behavioral": "Section 5: Arditi behavioral direction (Llama)",
    "full": "All of the above",
}


def cells():
    """
    Generate all the cells for the paper tables.
    """
    for m in MODELS:
        for q in QUANTS:
            if m == "qwen7b" and q == "aqlm2":
                continue
            yield m, q


def M(mod, *args):
    """
    Generate a command to run a module.
    """
    return [sys.executable, "-u", "-m", mod, *map(str, args)]


def parse_flags(argv):
    """
    Parse the flags from the command line.
    """
    flags = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                flags[key] = argv[i + 1]
                i += 2
            else:
                flags[key] = "true"
                i += 1
        else:
            i += 1
    return flags


def cfg_from_cmd(cmd):  
    """
    Generate a configuration from a command.
    """
    mod = cmd[3]
    if mod not in MOD_SCRIPT:
        return None
    f = parse_flags(cmd[4:])
    model = f.get("model", "llama7b")
    quantizer = f.get("quantizer", "aqlm2")
    variant = None
    omit_quantizer = False

    if mod.endswith("probe_refit_control"):
        omit_quantizer = True
    if "layer_lo" in f and "layer_hi" in f:
        variant = f"L{f['layer_lo']}-{f['layer_hi']}"
    elif "projections" in f:
        variant = f["projections"]
    elif "mask" in f:
        variant = f"{f['mask']}_{f['tag']}" if f.get("tag") else f["mask"]
    elif "role" in f and "n_bits" in f:
        variant = f"{f['role']}_int{f['n_bits']}"
    elif mod.endswith("qresafe_pairs") and f.get("wj_config", "eval") != "train":
        variant = f.get("wj_config", "eval")

    return RunConfig(
        model = model,
        quantizer = quantizer,
        variant = variant,
        omit_quantizer = omit_quantizer,
    )


def output_exists(cmd):
    """
    Check if the output exists for a command.
    """
    mod = cmd[3]
    rel = MOD_SCRIPT.get(mod)
    cfg = cfg_from_cmd(cmd)
    if not rel or cfg is None:
        return False
    path = result_path(ROOT / rel, cfg)
    return path.is_file() and path.stat().st_size > 10


def table1_cmds():
    """
    Generate the commands for table 1 (signal geometry direction drift, probe accuracy, probe rotation).
    """
    cmds = []
    for exp in ("direction_drift", "probe_accuracy", "probe_rotation"):
        for m, q in cells():
            cmds.append(M(f"experiments.signal_geometry.{exp}",
                          "--model", m, "--quantizer", q, "--n_calib", 200))
    return cmds


def table2_cmds():
    """
    Generate the commands for table 2 (restoration matrix).
    """
    cmds = []
    for m, q in cells():
        cmds.append(M("experiments.regime_sweep.restoration_sweep",
                      "--model", m, "--quantizer", q,
                      "--n_eval", 200, "--n_reasoning", 100))
    return cmds


def table3_cmds():
    """
    Generate the commands for table 3 (per-projection localization).
    """
    return [M("experiments.localization.per_projection",
                "--model", "llama7b", "--quantizer", "aqlm2",
                "--n_eval", 200, "--n_reasoning", 50,
                "--layer_lo", 11, "--layer_hi", 22)]


def table4_cmds():
    """
    Generate the commands for table 4 (depth-band localization).
    """
    cmds = []
    for side in ("read", "write"):
        cmds.append(M("experiments.localization.depth_bands",
                      "--model", "llama7b", "--quantizer", "aqlm2",
                      "--projections", side, "--n_eval", 200, "--n_reasoning", 50))
    return cmds


def table5_cmds():
    """
    Generate the commands for table 5 (headline eval).
    """
    return [M("experiments.headline.headline_eval",
              "--model", "llama7b", "--quantizer", "aqlm2",
              "--n_eval", 200, "--n_reasoning", 100, "--n_boot", 2000)]


def table6_cmds():
    """
    Generate the commands for table 6 (controls for param-matched, refit, cosmed, leakage).
    """
    cmds = []
    for m in ("llama7b", "phi3"):
        cmds.append(M("experiments.regime_sweep.param_matched_restore",
                      "--model", m, "--quantizer", "aqlm2",
                      "--n_eval", 200, "--n_reasoning", 50, "--seeds", 3))
        cmds.append(M("experiments.boundary.probe_refit_control",
                      "--model", m, "--n_calib", 200))
        cmds.append(M("experiments.signal_geometry.cosmed_after_restore",
                      "--model", m, "--quantizer", "aqlm2", "--n_calib", 200))
        cmds.append(M("experiments.boundary.probe_leakage_control",
                      "--model", m, "--n_calib", 200, "--n_transfer", 200))
    return cmds


def table7_cmds():
    """
    Generate the commands for table 7 (trained repair with Q-Resafe + LR-QAT).
    """
    cmds = [M("experiments.trained_repair.qresafe_pairs",
              "--model", "llama7b", "--quantizer", "aqlm2",
              "--n_prompts", 400)]
    for r in (128, 256, 512, 1024, 2048):
        cmds.append(M("experiments.trained_repair.qresafe_dpo",
                      "--mask", "full", "--r", r, "--alpha", 2 * r, "--tag", f"r{r}",
                      "--beta", 0.01, "--epochs", 1, "--lr", "5e-7",
                      "--n_eval", 200, "--opt8bit", "--seed", 42))
    budget = ["--steps", 3000, "--lr", "1e-4", "--rank", 32, "--alpha", 64,
              "--seq_len", 512, "--grad_accum", 8, "--n_train_docs", 4000,
              "--group_size", 128, "--n_eval", 200, "--n_reasoning", 50, "--n_xstest", 50]
    for role in ("read", "write"):
        for b in (2, 4):
            cmds.append(M("experiments.trained_repair.lrqat",
                        "--model", "llama7b", "--quantizer", "aqlm2",
                        "--role", role, "--n_bits", b, *budget))
    return cmds


def behavioral_cmds():
    """
    Generate the commands for the behavioral experiment.
    """
    cmds = []
    for q in ("aqlm2", "fp32"):
        cmds.append(M("experiments.behavioral_direction.arditi_repro",
                      "--model", "llama7b", "--quantizer", q))
    return cmds


def cmds_for_target(name):
    """
    Generate the commands for a target.
    """
    if name == "table1":
        return table1_cmds()
    if name == "table2":
        return table2_cmds()
    if name == "table3":
        return table3_cmds()
    if name == "table4":
        return table4_cmds()
    if name == "table5":
        return table5_cmds()
    if name == "table6":
        return table6_cmds()
    if name == "table7":
        return table7_cmds()
    if name == "behavioral":
        return behavioral_cmds()
    if name == "full":
        out = []
        for key in ("table1", "table2", "table3", "table4", "table5",
                    "table6", "table7", "behavioral"):
            out.extend(cmds_for_target(key))
        return out
    raise ValueError(name)


def run_pipeline(targets, *, force):
    """
    Run the pipeline for a list of targets.
    """
    cmds = []
    labels = []
    for t in targets:
        part = cmds_for_target(t)
        cmds.extend(part)
        labels.append(f"{t} ({len(part)} steps)")

    failed = skipped = 0
    print(f"SafeQuant run: {', '.join(labels)}")
    print(f"Total: {len(cmds)} steps (GPU required)\n")

    # Run the commands
    for i, cmd in enumerate(cmds, 1):
        mod_tail = " ".join(cmd[4:])
        if not force and output_exists(cmd):
            skipped += 1
            print(f"[{i}/{len(cmds)}] skip (exists)  python -m {mod_tail}")
            continue
        print(f"[{i}/{len(cmds)}] run  python -m {mod_tail}")
        rc = subprocess.run(cmd, cwd = ROOT).returncode

        # Check if the command failed
        if rc != 0:
            failed += 1
            print(f"  !! exited {rc}; continuing")
    print(f"\nDone. ran={len(cmds) - skipped - failed}  skipped={skipped}  failed={failed}")
    return failed


def rebuild_results(md = False):
    """
    Rebuild the paper results from the outputs/.
    """
    docs = ROOT / "docs"
    docs.mkdir(exist_ok = True)
    results_path = docs / "PAPER_RESULTS.md"
    json_path = ROOT / "paper_values.json"
    tables_path = docs / "PAPER_TABLES.md"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    with open(results_path, "w", encoding = "utf-8") as fh:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate_results.py")],
                         cwd = ROOT, stdout = fh, check = True, env = env)
    with open(json_path, "w", encoding = "utf-8") as fh:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "paper_values.py")],
                         cwd = ROOT, stdout = fh, check = True, env = env)
    print(f"Wrote {results_path.relative_to(ROOT)}")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    if md:
        with open(tables_path, "w", encoding = "utf-8") as fh:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "paper_values.py"), "--md"],
                             cwd = ROOT, stdout = fh, check = True, env = env)
        print(f"Wrote {tables_path.relative_to(ROOT)}")


def main():
    """
    Main function to run the pipeline.
    """
    target_help = "\n".join(f"  {k:12} {v}" for k, v in RUN_TARGETS.items())
    ap = argparse.ArgumentParser(
        description = "SafeQuant: run paper tables or rebuild result files from outputs/.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = f"""
Run targets (--run):
{target_help}

Examples:
  python run_pipeline.py --run full
  python run_pipeline.py --run table1
  python run_pipeline.py --run table2 table5
  python run_pipeline.py --results --md

Steps with an existing output JSON are skipped unless you pass --force.
        """,
    )
    mode = ap.add_mutually_exclusive_group(required = True)
    mode.add_argument("--run", nargs = "+", metavar = "TARGET",
                      choices = list(RUN_TARGETS.keys()),
                      help = "run one or more paper tables, or full")
    mode.add_argument("--results", action = "store_true",
                      help = "rebuild paper result files from outputs/")
    ap.add_argument("--force", action = "store_true",
                    help = "with --run: rerun steps even when output JSON exists")
    ap.add_argument("--md", action = "store_true",
                    help = "with --results: also write markdown tables")
    args = ap.parse_args()

    if args.results:
        rebuild_results(md = args.md)
        return

    failed = run_pipeline(args.run, force = args.force)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

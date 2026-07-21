# Aggregate experiment JSONs into paper-ready markdown grids.
import glob
import json
import statistics
from pathlib import Path

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "gptq3", "gptq2", "aqlm2"]
OUT = "outputs"


def load(p):
    try:
        with open(p, encoding = "utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def g(d, *path, default = None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def num(x, r = 3):
    if x is None:
        return "--"
    return f"{float(x):.{r}f}"


def grid(title, note, getter, models = MODELS, quants = QUANTS):
    """Print a model x quantizer markdown table using getter(model, quant)->str."""
    print(f"\n### {title}\n")
    if note:
        print(f"{note}\n")
    print("| model | " + " | ".join(quants) + " |")
    print("|" + "---|" * (len(quants) + 1))
    for m in models:
        cells = []
        for q in quants:
            cells.append(getter(m, q))
        print(f"| {m} | " + " | ".join(cells) + " |")


def sig(exp, m, q):
    return load(f"{OUT}/signal_geometry/{exp}_{m}_{q}.json")


def restore(m, q):
    return load(f"{OUT}/regime_sweep/restoration_sweep_{m}_{q}.json")


def drift_cos_median(m, q):
    d = sig("direction_drift", m, q)
    return num(g(d, "summary", "cos_sim_median"))


def drift_cos_min(m, q):
    d = sig("direction_drift", m, q)
    return num(g(d, "summary", "cos_sim_min"))


def probe_rotation_cos(m, q):
    d = sig("probe_rotation", m, q)
    return num(g(d, "summary", "cos_median"))


def probe_acc_mean(m, q):
    d = sig("probe_accuracy", m, q)
    return num(g(d, "summary", "quant_mean_acc"))


def probe_acc_min(m, q):
    d = sig("probe_accuracy", m, q)
    return num(g(d, "summary", "quant_min_acc"))


def probe_acc_drop_count(m, q):
    d = sig("probe_accuracy", m, q)
    if d is None:
        return "--"
    n = g(d, "summary", "n_layers_with_drop", default = "--")
    return str(n)


def damage_pp(m, q):
    d = restore(m, q)
    return num(g(d, "damage_pp"), 1)


def quantized_asr(m, q):
    d = restore(m, q)
    return num(g(d, "quantized_asr"), 1)


def swap_valid_cell(m, q):
    d = restore(m, q)
    if d is None:
        return "--"
    return str(g(d, "swap_valid", default = "--"))


def dmed(m, q):
    # Median per-layer separation (separation_sigma) of the QUANTIZED model.
    # FP32 reference is roughly 5.1.
    d = sig("direction_drift", m, q)
    if not d:
        return "--"
    qk = [
        k for k in d
        if k.endswith("_separation_per_layer") and not k.startswith("fp32")
    ]
    if not qk:
        return "--"
    vals = []
    for v in d[qk[0]].values():
        if not isinstance(v, dict):
            continue
        sep = v.get("separation_sigma")
        if sep is not None:
            vals.append(sep)
    if not vals:
        return "--"
    return f"{float(statistics.median(vals)):.2f}"


def restore_cell(m, q, cond):
    # Distinguish "no file" (--) from a documented structural N/A.
    d = restore(m, q)
    if d is None:
        return "--"
    if d.get("restore_status") == "arch_mismatch_na":
        return "N/A"
    return num(g(d, "asr_per_condition", cond), 1)


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return num / (sx * sy)


def main():
    print("# SafeQuant Aggregated Results (paper values)\n")

    # -- Table 1: refusal-direction drift ----------------------------------
    print("\n## Table 1: Refusal-direction geometry\n")
    grid("1a. Diff-in-means direction drift (cosine median, FP32 vs quant)",
         "Higher = direction preserved. `direction_drift_*`.",
         drift_cos_median)
    grid("1b. Direction drift (cosine MIN over layers)",
         "Worst-layer cosine. `direction_drift_*`.",
         drift_cos_min)
    grid("1c. Linear-probe rotation (cosine median of probe weight vectors)",
         "The readable/separating axis. `probe_rotation_*`.",
         probe_rotation_cos)
    grid("1d. delta_med: median per-layer separation (quantized; decodability)",
         "Positive separation magnitude; FP32 reference ~5.1. Low = concept "
         "less separable. `direction_drift_*` per-layer separation_sigma.",
         dmed)

    # -- Table 2: probe accuracy -------------------------------------------
    print("\n## Table 2: Linear separability of the safety concept\n")
    grid("2a. Probe accuracy (quant mean over layers)",
         "FP32 mean ~0.994-0.997 for all. `probe_accuracy_*`.",
         probe_acc_mean)
    grid("2b. Probe accuracy (quant MIN over layers)",
         "Worst-layer separability. `probe_accuracy_*`.",
         probe_acc_min)
    grid("2c. # layers with >5pp accuracy drop",
         "0 everywhere except qwen/gptq2. `probe_accuracy_*`.",
         probe_acc_drop_count)

    # -- Table 3: restoration / damage -------------------------------------
    print("\n## Table 3: Damage and projection-restoration\n")
    grid("3a. Safety damage (damage_pp = quant_asr - fp32_asr)",
         "Positive = safety got WORSE. `restoration_sweep_*`.",
         damage_pp)
    grid("3b. Quantized ASR (% harmful complied)",
         "`restoration_sweep_*` quantized_asr.",
         quantized_asr)

    def read_restored(m, q):
        return restore_cell(m, q, "read_restored")

    def write_restored(m, q):
        return restore_cell(m, q, "write_restored")

    def both_restored(m, q):
        return restore_cell(m, q, "both_restored")

    grid("3c. ASR after READ-projection restoration",
         "`restoration_sweep_*` asr_per_condition.read_restored. "
         "N/A = arch-mismatch (restoration undefined); -- = not run.",
         read_restored)
    grid("3d. ASR after WRITE-projection restoration",
         "`restoration_sweep_*` asr_per_condition.write_restored.",
         write_restored)
    grid("3e. ASR after BOTH restored",
         "`restoration_sweep_*` asr_per_condition.both_restored.",
         both_restored)
    grid("3f. swap_valid (activation-swap sanity gate)",
         "False = swap didn't localize; exclude that cell from restore claims.",
         swap_valid_cell)

    # -- FP32 / quant ASR reference row ------------------------------------
    print("\n### 3g. FP32 baseline ASR per model (reference)\n")
    print("| model | fp32_asr |")
    print("|---|---|")
    for m in MODELS:
        d = None
        for q in QUANTS:
            d = restore(m, q)
            if d:
                break
        print(f"| {m} | {num(g(d, 'fp32_asr'), 1)} |")

    # -- damage-vs-displacement correlation --------------------------------
    print("\n## Section 5: Does displacement predict damage?\n")
    rows = []
    for m in MODELS:
        for q in QUANTS:
            dd = sig("direction_drift", m, q)
            pr = sig("probe_rotation", m, q)
            rs = restore(m, q)
            if not dd or not rs:
                continue
            if not g(rs, "swap_valid", default = True):
                continue
            if g(rs, "damage_pp") is None:
                continue
            rows.append((
                g(rs, "damage_pp"),
                g(dd, "summary", "cos_sim_median"),
                g(pr, "summary", "cos_median"),
            ))

    cm = []
    for d, c, _ in rows:
        if c is not None:
            cm.append((d, 1 - c))
    pm = []
    for d, _, p in rows:
        if p is not None:
            pm.append((d, 1 - p))

    if cm:
        c1 = corr([a for a, _ in cm], [b for _, b in cm])
    else:
        c1 = None
    if pm:
        c2 = corr([a for a, _ in pm], [b for _, b in pm])
    else:
        c2 = None

    print(f"- swap-valid cells with damage: **{len(rows)}**")
    print(f"- corr(damage, 1 - diff-in-means cosmed) = "
          f"**{num(c1)}**  (n={len(cm)})")
    print(f"- corr(damage, 1 - probe rotation)       = "
          f"**{num(c2)}**  (n={len(pm)})")
    print("- NOTE: sign is confounded. GPTQ-2/3 cells show large displacement "
          "AND *negative* damage (safety improved via generic degradation), while "
          "AQLM/Phi-3 cells show positive damage at moderate displacement. Report "
          "the causal read-restoration result (Table 3c) as the primary evidence, "
          "not this correlation.")

    print("\n## Table 6: Controls (param-matched / refit / cosmed)\n")
    for m in ("llama", "phi3"):
        pm = load(f"{OUT}/regime_sweep/param_matched_restore_{m}_aqlm2.json")
        refit = load(f"{OUT}/boundary/probe_refit_control_{m}.json")
        cosmed = load(f"{OUT}/signal_geometry/cosmed_after_restore_{m}_aqlm2.json")
        dd = sig("direction_drift", m, "aqlm2")
        pr = sig("probe_rotation", m, "aqlm2")
        print(f"\n### {m}/aqlm2\n")
        if pm:
            rec = g(pm, "recovery_pp", default = {}) or {}
            print(f"- param_matched write_full={num(rec.get('write_full'),1)} "
                  f"read_structured={num(rec.get('read_matched_whole'),1)} "
                  f"read_random={num(rec.get('read_matched_random_mean'),1)}")
        else:
            print("- param_matched: **missing**")
        if refit and pr:
            print(f"- probe refit floor={num(g(refit,'summary','cos_floor_median'),3)} "
                  f"observed_rotation={num(g(pr,'summary','cos_median'),3)}")
        if cosmed and g(cosmed, "summary"):
            sm = cosmed["summary"]
            quant_cos = sm.get("cosmed_quant")
            if quant_cos is None:
                quant_cos = g(dd, "summary", "cos_sim_median")
            print(f"- cosmed after restore: quant={num(quant_cos,3)} "
                  f"read={num(sm.get('cosmed_read_restored'),3)} "
                  f"write={num(sm.get('cosmed_write_restored'),3)} "
                  f"both={num(sm.get('cosmed_both_restored'),3)}")
        elif Path(f"{OUT}/signal_geometry/cosmed_after_restore_{m}_aqlm2.json").exists():
            print("- cosmed_after_restore: **empty or corrupt** (re-run cosmed job)")

    print("\n## Boundary: probe leakage (prose control)\n")
    for f in sorted(glob.glob(f"{OUT}/boundary/probe_leakage_control_*.json")):
        d = load(f)
        summary = g(d, "summary", default = {})
        print(f"- **{Path(f).stem}**: {json.dumps(summary, default=str)}")

    print("\n## Boundary: probe refit (all cells)\n")
    for f in sorted(glob.glob(f"{OUT}/boundary/probe_refit_control_*.json")):
        d = load(f)
        floor = num(g(d, "summary", "cos_floor_median"), 3)
        print(f"- **{Path(f).stem}**: floor_median={floor}")

    print("\n## Signal: cosmed after restore\n")
    for f in sorted(glob.glob(f"{OUT}/signal_geometry/cosmed_after_restore_*.json")):
        d = load(f)
        if not d:
            print(f"- **{Path(f).stem}**: EMPTY (re-run job)")
            continue
        sm = g(d, "summary", default = {}) or {}
        print(f"- **{Path(f).stem}**: {json.dumps(sm, default=str)}")

    print("\n## Regime: param matched restore\n")
    for f in sorted(glob.glob(f"{OUT}/regime_sweep/param_matched_restore_*.json")):
        d = load(f)
        rec = g(d, "recovery_pp", default = {})
        print(f"- **{Path(f).stem}**: recovery_pp={json.dumps(rec)}")

    # -- Behavioral (Arditi), boundary, localization, dequant, repair ------
    print("\n## Behavioral direction (Arditi ablation)\n")
    print("| cell | baseline_asr | ablated_asr | delta_pp | layer | pos | operative |")
    print("|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{OUT}/behavioral_direction/arditi_repro_*.json")):
        d = load(f)
        cell = Path(f).stem.replace("arditi_repro_", "")
        print(f"| {cell} "
              f"| {num(g(d,'baseline_asr'),1)} | {num(g(d,'ablated_asr'),1)} "
              f"| {num(g(d,'delta_asr_pp'),1)} | {g(d,'selected_layer')} "
              f"| {g(d,'selected_pos')} | {g(d,'operative')} |")

    print("\n## Boundary: activation swap (llama/aqlm2)\n")
    for f in sorted(glob.glob(f"{OUT}/boundary/activation_swap_*.json")):
        d = load(f)
        swap = g(d, "swap_asr", default = {})
        label = Path(f).stem.replace("activation_swap_", "")
        print(f"- {label}: {json.dumps(swap)}")

    print("\n## Localization (per-projection / depth-band recovery)\n")
    for f in sorted(glob.glob(f"{OUT}/localization/*.json")):
        d = load(f)
        rec = g(d, "recovery_pp_per_projection")
        if rec is None:
            rec = g(d, "recovery_pp_per_range")
        if rec is None:
            rec = {}
        print(f"- **{Path(f).stem}** (damage {num(g(d,'damage_pp'),1)}): "
              f"{json.dumps(rec)}")

    print("\n## Targeted dequant (source-agnostic read-restore)\n")
    for f in sorted(glob.glob(f"{OUT}/regime_sweep/targeted_dequant_*.json")):
        d = load(f)
        print(f"- **{Path(f).stem}**: recov_4bit={num(g(d,'recovery_pp_4bit'),1)}, "
              f"recov_fp32={num(g(d,'recovery_pp_fp32'),1)}, "
              f"frac_of_fp32={num(g(d,'frac_of_fp32_recovery'),2)} "
              f"(role={g(d,'role')}, layers={g(d,'layer_range')})")

    print("\n## Trained repair (Q-Resafe DPO / LR-QAT)\n")
    for f in sorted(glob.glob(f"{OUT}/trained_repair/*.json")):
        d = load(f)
        stem = Path(f).stem
        if not isinstance(d, dict):
            print(f"- **{stem}**: UNREADABLE")
            continue
        if "asr_lrqat_trained" in d:
            print(f"- **{stem}**: "
                  f"asr fp16={num(g(d,'asr_fp16'),1)} aqlm2={num(g(d,'asr_aqlm2'),1)} "
                  f"ptq={num(g(d,'asr_ptq_untrained'),1)} "
                  f"lrqat={num(g(d,'asr_lrqat_trained'),1)}; "
                  f"arc ptq={num(g(d,'arc_ptq_untrained'),1)} "
                  f"lrqat={num(g(d,'arc_lrqat_trained'),1)}; "
                  f"rho={num(g(d,'rho_lrqat_trained'),2)}; "
                  f"eff_bits={num(g(d,'repaired_effective_bits'),2)}; "
                  f"collapse={g(d,'coherence_collapse_lrqat_trained')}; "
                  f"role={g(d,'role')} n_bits={g(d,'n_bits')}")
            continue
        if any(k in d for k in ("fp_lora", "folded")):
            keep = {}
            for k in ("mean_codes_changed", "fp_lora", "folded", "role"):
                if k in d:
                    keep[k] = d[k]
            print(f"- **{stem}**: {json.dumps(keep, default=str)}")
            continue
        print(f"- **{stem}**: INCOMPLETE stub "
              f"(keys={sorted(d.keys())}); job unfinished or crashed")

    # -- Headline (rich XSTest/ARC/mcnemar) --------------------------------
    print("\n## Headline eval (full XSTest / ARC / McNemar)\n")
    for f in sorted(glob.glob(f"{OUT}/headline/*.json")):
        d = load(f)
        print(f"\n### {Path(f).stem}\n```json")
        subset = {}
        if isinstance(d, dict):
            for k in ("damage_pp", "recovery_pp", "asr", "arc", "recovery_ci_95"):
                if k in d:
                    subset[k] = d[k]
        print(json.dumps(subset, indent = 2, default=str))
        xs = g(d, "xstest", default = {})
        if xs:
            fr = {}
            for k, v in xs.items():
                fr[k] = g(v, "false_refusal_rate")
            print("xstest false_refusal_rate: " + json.dumps(fr))
        mc = g(d, "mcnemar", default = {})
        if mc:
            pvals = {}
            for k, v in mc.items():
                pvals[k] = g(v, "p_value")
            print("mcnemar p-values: " + json.dumps(pvals, default=str))
        print("```")

    # -- Gaps (computed from disk, not hardcoded) --------------------------
    print("\n## Gaps / caveats (do NOT invent values for these)\n")


if __name__ == "__main__":
    main()

# Pull paper-table values from outputs/ JSON files (one block per LaTeX table label).
import glob
import json
import statistics
import sys
from pathlib import Path

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "aqlm2", "gptq3", "gptq2"]
CONDS = ["fp32", "quant", "write_restored", "read_restored", "both_restored"]
OUT = "outputs"

MODEL_LABEL = {
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B-v0.2",
    "qwen": "Qwen-2.5-7B",
    "phi3": "Phi-3-mini",
}

QUANT_LABEL = {
    "awq": "AWQ-4",
    "gptq4": "GPTQ-4",
    "aqlm2": "AQLM-2",
    "gptq3": "GPTQ-3",
    "gptq2": "GPTQ-2",
}

PROJ_ORDER = [
    ("self_attn.q_proj", "W_q", "Attention routing", 1),
    ("self_attn.k_proj", "W_k", "Attention routing", 1),
    ("mlp.up_proj", "W_up", "MLP feature lift", 1),
    ("mlp.down_proj", "W_down", "MLP output", 1),
    ("self_attn.v_proj", "W_v", "Attention content", 1),
    ("mlp.gate_proj", "W_gate", "MLP gating", 1),
    ("self_attn.o_proj", "W_o", "Attention output", 1),
]

DEPTH_RANGES = [
    ("early", "Early", "[0,11)"),
    ("middle", "Middle", "[11,22)"),
    ("late", "Late", "[22,32)"),
]


def load(p):
    """
    Load one JSON result file, returning None on missing or corrupt data.

    Inputs:
        - p (str | Path): Path to a result JSON

    Outputs:
        - data (dict | None): Parsed JSON or None
    """
    try:
        with open(p, encoding = "utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def g(d, *path, default = None):
    """
    Walk nested dict keys safely.

    Inputs:
        - d (dict | None): Root object
        - path: Sequence of keys
        - default: Fallback when a key is missing

    Outputs:
        - value: Nested value or default
    """
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def rnd(x, n = 3):
    """
    Round a numeric value for paper display.

    Inputs:
        - x: Numeric or None
        - n (int): Decimal places

    Outputs:
        - out (float | None): Rounded float or None
    """
    return None if x is None else round(float(x), n)


def sig(exp, m, q):
    """
    Path helper for signal_geometry JSON.

    Inputs:
        - exp (str): Experiment stem
        - m (str): Filename model token
        - q (str): Quantizer id

    Outputs:
        - data (dict | None): Loaded JSON
    """
    return load(f"{OUT}/signal_geometry/{exp}_{m}_{q}.json")


def restore(m, q):
    """
    Path helper for restoration_sweep JSON.

    Inputs:
        - m (str): Filename model token
        - q (str): Quantizer id

    Outputs:
        - data (dict | None): Loaded JSON
    """
    return load(f"{OUT}/regime_sweep/restoration_sweep_{m}_{q}.json")


def dmed_quant(m, q):
    """
    Median per-layer separation_sigma on the quantized model (Table 1 dmed).

    Inputs:
        - m (str): Filename model token
        - q (str): Quantizer id

    Outputs:
        - dmed (float | None): Median separation across layers
    """
    d = sig("direction_drift", m, q)
    if not d:
        return None
    qk = [k for k in d if k.endswith("_separation_per_layer") and not k.startswith("fp32")]
    if not qk:
        return None
    vals = [
        v.get("separation_sigma")
        for v in d[qk[0]].values()
        if isinstance(v, dict) and v.get("separation_sigma") is not None
    ]
    if not vals:
        return None
    med = statistics.median(vals)
    return rnd(med, 2)


def cond_map(d, block):
    """
    Normalize restoration_sweep ASR/ARC blocks to fixed condition keys.

    Inputs:
        - d (dict): restoration_sweep JSON
        - block (str): Top-level key ('asr_per_condition' or 'arc')

    Outputs:
        - out (dict): Keys fp32, quant, write_restored, read_restored, both_restored
    """
    src = g(d, block, default = {}) or {}
    qtz = next(
        (k for k in src if k not in ("fp32", "write_restored", "read_restored", "both_restored")),
        None,
    )
    out = {}
    for c in CONDS:
        if c == "quant":
            key = qtz
        else:
            key = c
        val = src.get(key)
        if val is None:
            out[c] = None
        else:
            out[c] = rnd(val, 1)
    return out


def screen_verdict(rs):
    """
    Heuristic damaged-coherent screen label (Table 2 verdict column).

    Inputs:
        - rs (dict): restoration_sweep JSON

    Outputs:
        - verdict (str | None): Screen label or None if file incomplete
    """
    if not rs:
        return None
    if g(rs, "restore_status") == "arch_mismatch_na":
        return "N/A (arch mismatch)"
    damage = g(rs, "damage_pp")
    fp_asr = g(rs, "fp32_asr")
    arc_q = cond_map(rs, "arc").get("quant")
    read_rec = g(rs, "read_recovery_pp")

    # Coherence collapse: utility near chance
    if arc_q is not None and arc_q < 35:
        return "fails coherence"
    # High baseline refusal already; hard to interpret damage
    if fp_asr is not None and fp_asr >= 70:
        return "fails baseline"
    if damage is None:
        return None
    if damage < 5:
        return "no damage"
    if read_rec is not None and read_rec >= 5:
        return "studiable, damaged"
    if damage >= 5:
        return "damaged (read recovery weak)"
    return "no damage"


def recovery_rho(recovery_pp, damage_pp):
    """
    Share of safety damage recovered (percent).

    Inputs:
        - recovery_pp (float | None): Points recovered
        - damage_pp (float | None): Baseline damage in pp

    Outputs:
        - rho (int | None): Integer percent or None
    """
    if recovery_pp is None or not damage_pp:
        return None
    pct = 100.0 * float(recovery_pp) / float(damage_pp)
    return int(round(pct))


def build_geometry():
    """
    Build tab:geometry: dmed, a_min, cosmed, coswmed per model/quant cell.

    Inputs:
        - None

    Outputs:
        - block (dict): Table payload keyed by model then quant
    """
    by_model = {}
    for m in MODELS:
        rows = []
        for q in QUANTS:
            if m == "qwen" and q == "aqlm2":
                continue
            dd = sig("direction_drift", m, q)
            if not dd:
                continue
            rows.append({
                "quant": q,
                "quant_label": QUANT_LABEL[q],
                "dmed": dmed_quant(m, q),
                "a_min": rnd(g(sig("probe_accuracy", m, q), "summary", "quant_min_acc"), 3),
                "cosmed": rnd(g(dd, "summary", "cos_sim_median"), 2),
                "coswmed": rnd(g(sig("probe_rotation", m, q), "summary", "cos_median"), 2),
            })
        if rows:
            by_model[m] = {"model_label": MODEL_LABEL[m], "rows": rows}
    return {
        "columns": ["dmed", "a_min", "cosmed", "coswmed"],
        "source": (
            "direction_drift_* (cosmed + per-layer separation_sigma), "
            "probe_rotation_* (coswmed), probe_accuracy_* (a_min)"
        ),
        "by_model": by_model,
    }


# AQLM inference is nondeterministic at +-1-2pp on 200 prompts, so the
# restoration re-run's ASR for AQLM cells can differ from the headline run.
# For the two studiable AQLM cells (which also appear in tab:f3a and carry the
# significance stats), source ASR/ARC/damage from the single headline run so
# every table agrees by construction. gptq/awq cells are deterministic -> use
# restoration directly.
STUDIABLE_HEADLINE = {("llama", "aqlm2"), ("phi3", "aqlm2")}


def build_regime():
    """
    Build tab:regime: ASR/ARC per condition + screen verdict.

    Inputs:
        - None

    Outputs:
        - block (dict): Table payload for damaged-coherent screen
    """
    cells = {}
    for m in MODELS:
        for q in QUANTS:
            rs = restore(m, q)
            if not rs:
                continue
            hl_path = f"{OUT}/headline/headline_{m}_{q}.json"
            if (m, q) in STUDIABLE_HEADLINE:
                hl = load(hl_path)
            else:
                hl = None
            if hl:
                # headline-canonical (consistent single run + stats)
                asr = cond_map(hl, "asr")
                arc = cond_map(hl, "arc")
                damage = rnd(g(hl, "damage_pp"), 1)
                src = "headline"
            else:
                # deterministic cells: restoration re-run (ARC n=100)
                asr = cond_map(rs, "asr_per_condition")
                arc = cond_map(rs, "arc")
                damage = rnd(g(rs, "damage_pp"), 1)
                src = "restoration_sweep"
            cells[f"{m}/{q}"] = {
                "model_label": MODEL_LABEL[m],
                "quant_label": QUANT_LABEL[q],
                "asr": asr,
                "arc": arc,
                "damage_pp": damage,
                "swap_valid": g(rs, "swap_valid"),
                "restore_status": g(rs, "restore_status", default = "ok"),
                "screen_verdict": screen_verdict(rs),
                "source": src,
            }
    return {
        "conditions": CONDS,
        "source": ("restoration_sweep_* (ARC n=100); AQLM studiable cells "
                   "llama/aqlm2 + phi3/aqlm2 sourced from headline_* for "
                   "ASR/ARC consistency with tab:f3a"),
        "cells": cells,
    }


def build_f3a_proj():
    """
    Build tab:f3a_proj: per-projection recovery rho on Llama AQLM-2 middle band.

    Inputs:
        - None

    Outputs:
        - block (dict): Projection rows with rho percent
    """
    pp = load(f"{OUT}/localization/per_projection_llama_aqlm2_L11-22.json")
    damage = g(pp, "damage_pp")
    rec = g(pp, "recovery_pp_per_projection", default = {}) or {}
    rows = []
    for path, sym, func, n_mat in PROJ_ORDER:
        rec_pp = rec.get(path)
        rows.append({
            "projection": sym,
            "function": func,
            "matrices": n_mat,
            "recovery_pp": rnd(rec_pp, 1),
            "rho_pct": recovery_rho(rec_pp, damage),
        })
    return {
        "source": "per_projection_llama_aqlm2_L11-22.json",
        "layer_range": g(pp, "layer_range"),
        "damage_pp": rnd(damage, 1),
        "rows": rows,
    }


def build_f3a_range():
    """
    Build tab:f3a_range: depth-band read/write recovery.

    Inputs:
        - None

    Outputs:
        - block (dict): Early/middle/late bands for read and write sides
    """
    dbr = load(f"{OUT}/localization/depth_bands_llama_aqlm2_read.json")
    dbw = load(f"{OUT}/localization/depth_bands_llama_aqlm2_write.json")
    damage = g(dbr, "damage_pp") or g(dbw, "damage_pp")

    def side_rows(db, side):
        asr = g(db, "asr_per_condition", default = {}) or {}
        arc = g(db, "arc_per_condition", default = {}) or {}
        rec = g(db, "recovery_pp_per_range", default = {}) or {}
        out = []
        for key, label, layer_str in DEPTH_RANGES:
            asr_key = f"layers_{key}"
            out.append({
                "side": side,
                "range": label,
                "layers": layer_str,
                "asr": rnd(asr.get(asr_key), 1),
                "arc": rnd(arc.get(asr_key), 1),
                "recovery_pp": rnd(rec.get(key), 1),
                "rho_pct": recovery_rho(rec.get(key), damage),
            })
        return out

    return {
        "source": "depth_bands_llama_aqlm2_read/write.json",
        "damage_pp": rnd(damage, 1),
        "read": side_rows(dbr, "Read"),
        "write": side_rows(dbw, "Write"),
    }


def build_f3a():
    """
    Build tab:f3a: static restoration headline (Llama AQLM-2).

    Inputs:
        - None

    Outputs:
        - block (dict): FP / quant / +W / +R / +Both with rho and XSTest
    """
    hl = load(f"{OUT}/headline/headline_llama_aqlm2.json")
    damage = g(hl, "damage_pp")
    rec = g(hl, "recovery_pp", default = {}) or {}
    xs = g(hl, "xstest", default = {}) or {}

    def row(restoration, asr_key, rec_key = None, arc_key = None, xs_key = None):
        rec_pp = rec.get(rec_key) if rec_key else None
        delta = None
        if rec_pp is not None:
            delta = rnd(-rec_pp, 1)
        return {
            "restoration": restoration,
            "asr": rnd(g(hl, "asr", asr_key), 1),
            "delta_quant": delta,
            "rho_pct": recovery_rho(rec_pp, damage),
            "arc": rnd(g(hl, "arc", arc_key or asr_key), 0),
            "xstest_fr": rnd(g(xs.get(xs_key or asr_key, {}), "false_refusal_rate"), 1),
        }

    rows = [
        row("FP32", "fp32"),
        row("quant", "aqlm2"),
        row("+ Write-side", "write_restored", rec_key = "write"),
        row("+ Read-side", "read_restored", rec_key = "read"),
        row("+ Both", "both_restored", rec_key = "both"),
    ]
    return {
        "source": "headline_llama_aqlm2.json",
        "damage_pp": rnd(damage, 1),
        "rows": rows,
        "mcnemar_p": {k: g(v, "p_value") for k, v in (g(hl, "mcnemar", default = {}) or {}).items()},
    }


def build_controls():
    """
    Build tab:controls: localization controls for Llama and Phi-3 AQLM-2.

    Inputs:
        - None

    Outputs:
        - block (dict): Param-matched restore, probe refit floor, cosmed after restore
    """
    out = {}
    for m, label in (("llama", "Llama-AQLM-2"), ("phi3", "Phi-3-AQLM-2")):
        pm = load(f"{OUT}/regime_sweep/param_matched_restore_{m}_aqlm2.json")
        refit = load(f"{OUT}/boundary/probe_refit_control_{m}.json")
        rot = sig("probe_rotation", m, "aqlm2")
        cosmed = load(f"{OUT}/signal_geometry/cosmed_after_restore_{m}_aqlm2.json")
        dd = sig("direction_drift", m, "aqlm2")

        cell = {
            "label": label,
            "param_matched_pp": {
                "full_write": rnd(g(pm, "recovery_pp", "write_full"), 1),
                "matched_read_structured": rnd(g(pm, "recovery_pp", "read_matched_whole"), 1),
                "matched_read_random": rnd(g(pm, "recovery_pp", "read_matched_random_mean"), 1),
            },
            "probe_rotation": {
                "refit_floor": rnd(g(refit, "summary", "cos_floor_median"), 3),
                "observed_rotation": rnd(g(rot, "summary", "cos_median"), 3),
            },
            "direction_cosine": {
                "quantized": rnd(g(dd, "summary", "cos_sim_median"), 3),
            },
        }
        if cosmed and g(cosmed, "summary"):
            sm = cosmed["summary"]
            cell["direction_cosine"]["read_restored"] = rnd(
                sm.get("cosmed_read_restored") or sm.get("read_restored_cos_median"), 3)
            cell["direction_cosine"]["write_restored"] = rnd(
                sm.get("cosmed_write_restored") or sm.get("write_restored_cos_median"), 3)
            cell["direction_cosine"]["both_restored"] = rnd(
                sm.get("cosmed_both_restored") or sm.get("both_restored_cos_median"), 3)
        else:
            cell["direction_cosine"]["note"] = (
                "cosmed_after_restore missing or empty; run "
                f"experiments.signal_geometry.cosmed_after_restore --model {m}")
        out[m] = cell
    return {"source": "param_matched_restore, probe_refit_control, cosmed_after_restore", "cells": out}


def build_trained_repair():
    """
    Build tab:trained_repair: Q-Resafe folded patches (rank 128 target).

    Inputs:
        - None

    Outputs:
        - block (dict): Reader / Writer / Both rows when JSON exists
    """
    role_files = {
        "Reader": "qresafe_dpo_llama_aqlm2_read_r128.json",
        "Writer": "qresafe_dpo_llama_aqlm2_write_r128.json",
        "Both": "qresafe_dpo_llama_aqlm2_full_r128.json",
    }
    rows = []
    for patch, fname in role_files.items():
        d = load(f"{OUT}/trained_repair/{fname}")
        if not d:
            rows.append({"patch": patch, "rank": 128, "status": "missing or running"})
            continue
        rank = g(d, "manifest", "r") or g(d, "manifest", "extra", "r") or 128
        rows.append({
            "patch": patch,
            "rank": rank,
            "asr": rnd(g(d, "folded", "asr"), 1),
            "arc": rnd(g(d, "folded", "arc"), 0),
            "xstest_fr": rnd(g(d, "folded", "xstest_fr"), 1),
            "mean_codes_changed": g(d, "mean_codes_changed"),
            "status": "complete",
        })
    # Also surface any other qresafe_dpo variants for in-progress sweeps
    extras = {}
    for f in sorted(glob.glob(f"{OUT}/trained_repair/qresafe_dpo_*.json")):
        stem = Path(f).stem.replace("qresafe_dpo_", "")
        if stem in ("llama_aqlm2_read", "llama_aqlm2_write", "llama_aqlm2_full"):
            d = load(f)
            extras[stem] = {
                "r": g(d, "manifest", "r") or g(d, "manifest", "extra", "r"),
                "folded_asr": rnd(g(d, "folded", "asr"), 1),
                "folded_arc": rnd(g(d, "folded", "arc"), 0),
            }
    return {
        "source": "qresafe_dpo_* (folded metrics)",
        "reference": {"fp_asr": 41.0, "aqlm2_asr": 62.5, "fp_xstest": 6.0, "aqlm2_arc": 46},
        "rows": rows,
        "other_variants": extras,
    }


def build_lrqat():
    """
    Build tab:lrqat: LR-QAT cells for Llama AQLM-2 read/write x int2/int4.

    Inputs:
        - None

    Outputs:
        - block (dict): Role x bits grid with coherence flag
    """
    rows = []
    refs = {"fp16": {"asr": 41.0, "arc": 55, "coherent": True},
            "aqlm2": {"asr": 62.5, "arc": 46, "coherent": None}}
    rows.append({"role": "FP reference", "bits": 16, **refs["fp16"]})
    rows.append({"role": "AQLM-2", "bits": 2.0, **refs["aqlm2"]})

    for f in sorted(glob.glob(f"{OUT}/trained_repair/lrqat_llama_aqlm2_*.json")):
        d = load(f)
        if not d or "asr_lrqat_trained" not in d:
            continue
        role = g(d, "role", default = "unknown")
        n_bits = g(d, "n_bits")
        ptq_coll = g(d, "coherence_collapse_ptq_untrained", default = False)
        train_coll = g(d, "coherence_collapse_lrqat_trained", default = False)
        rows.append({
            "role": f"{role} (PTQ untrained)",
            "bits": float(n_bits) if n_bits is not None else None,
            "asr": rnd(g(d, "asr_ptq_untrained"), 1),
            "arc": rnd(g(d, "arc_ptq_untrained"), 0),
            "coherent": "no" if ptq_coll else "yes",
        })
        rows.append({
            "role": f"{role} (LR-QAT trained)",
            "bits": float(n_bits) if n_bits is not None else None,
            "asr": rnd(g(d, "asr_lrqat_trained"), 1),
            "arc": rnd(g(d, "arc_lrqat_trained"), 0),
            "coherent": "no" if train_coll else "yes",
            "effective_bits": rnd(g(d, "repaired_effective_bits"), 2),
        })
    return {"source": "lrqat_llama_aqlm2_*.json", "rows": rows}


def build():
    """
    Assemble all paper-table blocks into one JSON document.

    Inputs:
        - None

    Outputs:
        - pv (dict): Full paper-values payload
    """
    return {
        "meta": {
            "generated_by": "scripts/paper_values.py",
            "note": (
                "ASR/ARC/damage in percentage points. dASR = quant_asr - fp32_asr. "
                "Null = cell absent. Incomplete Q-Resafe read/write r128 jobs stay "
                "marked 'missing or running' until JSON lands."
            ),
            "seed": 42,
        },
        "tab:geometry": build_geometry(),
        "tab:regime": build_regime(),
        "tab:f3a_proj": build_f3a_proj(),
        "tab:f3a_range": build_f3a_range(),
        "tab:f3a": build_f3a(),
        "tab:controls": build_controls(),
        "tab:trained_repair": build_trained_repair(),
        "tab:lrqat": build_lrqat(),
    }


def fmt_cell(x):
    """
    Format one table cell for markdown output.

    Inputs:
        - x: Value or None

    Outputs:
        - s (str): Display string
    """
    return "--" if x is None else str(x)


def as_markdown(pv):
    """
    Render paper_values JSON as human-readable markdown tables.

    Inputs:
        - pv (dict): Output of build()

    Outputs:
        - text (str): Markdown document
    """
    lines = ["# Paper values (LaTeX table alignment)\n", pv["meta"]["note"] + "\n"]

    lines.append("\n## tab:geometry\n")
    lines.append("| Model | Quant | dmed | a_min | cosmed | coswmed |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for m, block in pv["tab:geometry"]["by_model"].items():
        for row in block["rows"]:
            lines.append(
                f"| {block['model_label']} | {row['quant_label']} | "
                f"{fmt_cell(row['dmed'])} | {fmt_cell(row['a_min'])} | "
                f"{fmt_cell(row['cosmed'])} | {fmt_cell(row['coswmed'])} |")

    lines.append("\n## tab:regime\n")
    lines.append("| Model | Quant | FP | Quant | +Write | +Read | +Both | Verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for key, v in pv["tab:regime"]["cells"].items():
        a, r = v["asr"], v["arc"]
        cells = " | ".join(f"{fmt_cell(a[c])}/{fmt_cell(r[c])}" for c in CONDS)
        lines.append(
            f"| {v['model_label']} | {v['quant_label']} | {cells} | "
            f"{fmt_cell(v['screen_verdict'])} |")

    lines.append("\n## tab:f3a_proj\n")
    lines.append("| Projection | Function | Matrices | rho % |")
    lines.append("|---|---|---:|---:|")
    for row in pv["tab:f3a_proj"]["rows"]:
        lines.append(
            f"| {row['projection']} | {row['function']} | {row['matrices']} | "
            f"{fmt_cell(row['rho_pct'])} |")

    lines.append("\n## tab:f3a_range\n")
    lines.append("| Side | Range | Layers | ASR | rho % | ARC |")
    lines.append("|---|---|---|---:|---:|---:|")
    for side_key in ("read", "write"):
        for row in pv["tab:f3a_range"][side_key]:
            lines.append(
                f"| {row['side']} | {row['range']} | {row['layers']} | "
                f"{fmt_cell(row['asr'])} | {fmt_cell(row['rho_pct'])} | "
                f"{fmt_cell(row['arc'])} |")

    lines.append("\n## tab:f3a\n")
    lines.append("| Restoration | ASR | delta quant | rho % | ARC | XSTest |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in pv["tab:f3a"]["rows"]:
        lines.append(
            f"| {row['restoration']} | {fmt_cell(row['asr'])} | "
            f"{fmt_cell(row['delta_quant'])} | {fmt_cell(row['rho_pct'])} | "
            f"{fmt_cell(row['arc'])} | {fmt_cell(row['xstest_fr'])} |")

    lines.append("\n## tab:controls\n")
    for m, cell in pv["tab:controls"]["cells"].items():
        lines.append(f"\n### {cell['label']}\n")
        pm = cell["param_matched_pp"]
        pr = cell["probe_rotation"]
        dc = cell["direction_cosine"]
        lines.append("| Control | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Full write (pp) | {fmt_cell(pm['full_write'])} |")
        lines.append(f"| Matched read structured (pp) | {fmt_cell(pm['matched_read_structured'])} |")
        lines.append(f"| Matched read random (pp) | {fmt_cell(pm['matched_read_random'])} |")
        lines.append(f"| Refit floor | {fmt_cell(pr['refit_floor'])} |")
        lines.append(f"| Observed rotation | {fmt_cell(pr['observed_rotation'])} |")
        for k in ("quantized", "read_restored", "write_restored", "both_restored"):
            if k in dc:
                lines.append(f"| Direction cos {k} | {fmt_cell(dc[k])} |")

    lines.append("\n## tab:trained_repair\n")
    lines.append("| Patch | Rank | ASR | ARC | XSTest | Status |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in pv["tab:trained_repair"]["rows"]:
        lines.append(
            f"| {row['patch']} | {row['rank']} | {fmt_cell(row.get('asr'))} | "
            f"{fmt_cell(row.get('arc'))} | {fmt_cell(row.get('xstest_fr'))} | "
            f"{row.get('status', '--')} |")

    lines.append("\n## tab:lrqat\n")
    lines.append("| Role | Bits | ASR | ARC | Coherent |")
    lines.append("|---|---:|---:|---:|---|")
    for row in pv["tab:lrqat"]["rows"]:
        lines.append(
            f"| {row['role']} | {fmt_cell(row.get('bits'))} | {fmt_cell(row.get('asr'))} | "
            f"{fmt_cell(row.get('arc'))} | {fmt_cell(row.get('coherent'))} |")

    return "\n".join(lines)


def main():
    # Build payload and emit JSON or markdown
    pv = build()
    if "--md" in sys.argv[1:]:
        print(as_markdown(pv))
    else:
        print(json.dumps(pv, indent = 2, default = str))


if __name__ == "__main__":
    main()

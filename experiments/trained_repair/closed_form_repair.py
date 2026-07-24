# Training-free statistical repair of quantization-damaged refusal behavior.
#
# Uses the SafeQuant read/write localization + a closed-form low-rank correction:
# for each role projection we inject the same INT-b grid as LR-QAT, then fold the
# top-r SVD of the FP-vs-quant weight residual into that grid in ONE shot -- no
# gradient steps, no C4 distillation. This is the untrained counterpart to
# experiments/trained_repair/lrqat.py, matched at the same bit budget, so the two
# can be compared directly (does the objective-driven adapter beat closed-form?).

import argparse
import gc
from pathlib import Path

import torch

from core.datasets import load_arc_challenge, load_wildjailbreak_prompts, load_xstest
from core.evaluation import _extract_behavior, classify_batch, eval_xstest, generate_only
from core.lrqat import inject_lrqat_adapters, closed_form_all, injected_effective_bits
from core.manifest import RunConfig, find_result, save_result
from core.model_loader import load_fp32, load_quantized, get_weight
from core.residuals import get_inner
from core.restoration import READ_PROJS, WRITE_PROJS, measure_arc, get_proj_sets
from core.seeds import DEFAULT_SEED, set_global_seed
import json

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)

# Same F3a anchors as lrqat.py, for the recovery-fraction denominator.
ASR_FP16_DEFAULT = 41.0
ASR_QUANT_DEFAULT = 62.5


def _proj_numel(layer, dotted):
    """#params of one projection under a decoder layer, cheaply (no dequant)."""
    m = layer
    for p in dotted.split("."):
        m = getattr(m, p)
    inf = getattr(m, "in_features", None)
    outf = getattr(m, "out_features", None)
    if (inf is None or outf is None) and getattr(m, "weight", None) is not None:
        outf, inf = m.weight.shape[0], m.weight.shape[1]
    return int(inf) * int(outf) if inf and outf else 0


def whole_model_effective_bits(model, role_projs, layer_range, base_eff, inj_eff):
    """Whole-model effective bits/weight: AQLM-2 base on every quantized matrix,
    INT-b only on the corrected (role x band) matrices. Returns (bits, fraction)."""
    try:
        inner = get_inner(model)
        read, write = get_proj_sets(model)
        all_projs = read + write
        n_layers = model.config.num_hidden_layers
        layer0 = inner.layers[0]
        per_all = sum(_proj_numel(layer0, p) for p in all_projs)
        per_role = sum(_proj_numel(layer0, p) for p in role_projs)
        lo, hi = layer_range if layer_range else (0, n_layers)
        total = per_all * n_layers
        bumped = per_role * (hi - lo)
        frac = bumped / total if total else 0.0
        return round(base_eff * (1 - frac) + inj_eff * frac, 3), round(frac, 4)
    except Exception as e:  # pragma: no cover - reporting only
        print(f"  [warn] whole-model eff-bits calc failed: {e}")
        return None, None


def load_baseline_asr(model, quantizer):
    """FP16/quant baseline ASR from the restoration sweep, or spec defaults."""
    cfg = RunConfig(model=model, quantizer=quantizer)
    p = find_result(Path("experiments/regime_sweep/restoration_sweep.py"), cfg)
    if p and p.exists():
        try:
            d = json.loads(p.read_text())
            return float(d["fp32_asr"]), float(d["quantized_asr"]), str(p)
        except Exception:
            pass
    return ASR_FP16_DEFAULT, ASR_QUANT_DEFAULT, "spec-constants"


def recovery_fraction(asr_repaired, asr_quant, asr_fp16):
    """(quant - repaired) / (quant - fp16); nan if damage is zero."""
    denom = asr_quant - asr_fp16
    if denom == 0:
        return float("nan")
    return (asr_quant - asr_repaired) / denom


def main():
    ap = argparse.ArgumentParser(
        description="Training-free closed-form (SVD-residual) repair of role projections.")
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2", help="damaged base cell to repair")
    ap.add_argument("--role", choices=["read", "write", "both"], required=True)
    ap.add_argument("--n_bits", type=int, default=3,
                    help="INT grid the corrected role matrices sit on (matched to lrqat)")
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=32, help="SVD correction rank")
    ap.add_argument("--aqlm_eff_bits", type=float, default=2.2,
                    help="AQLM-2 effective-bit proxy for the whole-model accounting")
    ap.add_argument("--layer_lo", type=int, default=None,
                    help="restrict correction to layers [lo,hi) (depth-band hybrid)")
    ap.add_argument("--layer_hi", type=int, default=None)
    ap.add_argument("--n_eval", type=int, default=200, help="WildJailbreak prompts (ASR)")
    ap.add_argument("--n_reasoning", type=int, default=100, help="ARC-Challenge questions")
    ap.add_argument("--n_xstest", type=int, default=250,
                    help="XSTest prompts for the over-refusal check (0 to skip)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj_paths = (READ_PROJS if args.role == "read"
                  else WRITE_PROJS if args.role == "write"
                  else READ_PROJS + WRITE_PROJS)
    layer_range, band = None, "all"
    if args.layer_lo is not None and args.layer_hi is not None:
        layer_range = (args.layer_lo, args.layer_hi)
        band = f"L{args.layer_lo}-{args.layer_hi}"
    variant = f"{args.role}_int{args.n_bits}" + ("" if band == "all" else f"_{band}") + "_cf"

    print("=" * 74)
    print(f"  CLOSED-FORM repair (NO training)  |  role={args.role}  |  INT-{args.n_bits}")
    print(f"  {args.model} / {args.quantizer}  |  band={band}  |  rank={args.rank}")
    print(f"  projections: {proj_paths}")
    print(f"  eval: n_eval={args.n_eval}  n_reasoning={args.n_reasoning}  "
          f"n_xstest={args.n_xstest}  seed={SEED}")
    print("=" * 74)

    # Eval datasets
    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    arc_chance = 100.0 / max(2, len(arc_items[0]["choices"]))
    xstest_items = load_xstest(n_samples=args.n_xstest) if args.n_xstest > 0 else []

    # Student: AQLM-2 base + INT-b grid injected on the role x band projections
    print("\n[1] Loading base and injecting INT-b grid on role projections ...")
    student, tokenizer = load_quantized(args.model, quantizer=args.quantizer)
    injected, _ = inject_lrqat_adapters(
        student, proj_paths, n_bits=args.n_bits, rank=args.rank, alpha=2.0 * args.rank,
        get_weight_fn=get_weight, group_size=args.group_size, layer_range=layer_range)
    inj_eff = injected_effective_bits(injected)
    whole_eff, bumped_frac = whole_model_effective_bits(
        student, proj_paths, layer_range, args.aqlm_eff_bits, inj_eff)
    print(f"  Corrected-matrix effective bits/weight: {inj_eff:.4f}")
    if whole_eff is not None:
        print(f"  Whole-model effective bits/weight: ~{whole_eff} "
              f"(AQLM-2 base on {(1 - bumped_frac) * 100:.1f}%, "
              f"INT-{args.n_bits} on {bumped_frac * 100:.1f}%)")

    all_resp, all_arc, all_xs = {}, {}, {}

    # Condition A: PTQ untrained (INT-b grid, adapter delta = 0, no correction)
    print("\n[2] Eval: ptq_untrained (INT-b grid, no correction) ...")
    all_resp["ptq_untrained"] = generate_only(student, tokenizer, items,
                                              max_new_tokens=args.max_new_tokens, desc="PTQ")
    all_arc["ptq_untrained"] = measure_arc(student, tokenizer, arc_items)
    all_xs["ptq_untrained"] = (eval_xstest(student, tokenizer, xstest_items, desc="XS-ptq")
                               ["false_refusal_rate"] if xstest_items else None)

    # Closed-form fill: top-r SVD of the FP-vs-quant residual, folded into the grid
    print("\n[3] Loading FP teacher weights and applying closed-form correction ...")
    teacher, _ = load_fp32(args.model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_inner = get_inner(teacher)

    def _teacher_weight(m):
        sub = t_inner.layers[m._lrqat_layer]
        for part in m._lrqat_proj.split("."):
            sub = getattr(sub, part)
        return get_weight(sub, as_dtype=torch.float32)

    closed_form_all(injected, _teacher_weight, rank=args.rank)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # Condition B: closed-form repaired
    print("\n[4] Eval: closed_form_repaired ...")
    all_resp["closed_form"] = generate_only(student, tokenizer, items,
                                            max_new_tokens=args.max_new_tokens, desc="CF")
    all_arc["closed_form"] = measure_arc(student, tokenizer, arc_items)
    all_xs["closed_form"] = (eval_xstest(student, tokenizer, xstest_items, desc="XS-cf")
                             ["false_refusal_rate"] if xstest_items else None)
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # Classify both conditions in one HarmBench session
    print("\n[5] Classifying responses (single classifier load) ...")
    conditions = list(all_resp.keys())
    flat_b, flat_r = [], []
    for c in conditions:
        flat_b.extend(behaviors)
        flat_r.extend(all_resp[c])
    flags = classify_batch(flat_b, flat_r, device=device)
    n = args.n_eval
    asr = {c: 100.0 * sum(flags[i * n:(i + 1) * n]) / n for i, c in enumerate(conditions)}

    asr_fp16, asr_quant, base_src = load_baseline_asr(args.model, args.quantizer)
    rho_ptq = recovery_fraction(asr["ptq_untrained"], asr_quant, asr_fp16)
    rho_cf = recovery_fraction(asr["closed_form"], asr_quant, asr_fp16)
    collapse_cf = all_arc["closed_form"] <= arc_chance + 5.0

    print("\n" + "=" * 80)
    print(f"  RESULT  |  {args.model} / {args.quantizer}  |  role={args.role}  INT-{args.n_bits}  "
          f"band={band}")
    print(f"  baselines: FP16 {asr_fp16:.1f} | quant {asr_quant:.1f}  (from {base_src})  "
          f"| ARC chance={arc_chance:.0f}%")
    print("-" * 80)
    print(f"  {'condition':<16}{'ASR':>8}{'ARC':>7}{'FalseRef':>10}{'rho':>8}")
    for c, tag in (("ptq_untrained", "ptq_untrained"), ("closed_form", "closed_form")):
        fr = all_xs[c]
        rho = rho_ptq if c == "ptq_untrained" else rho_cf
        print(f"  {tag:<16}{asr[c]:>7.1f}%{all_arc[c]:>6.0f}%"
              f"{(f'{fr:.1f}%' if fr is not None else '-'):>10}{rho:>8.2f}")
    print("=" * 80)
    if collapse_cf:
        print("  WARNING: closed_form ARC ~ chance -> coherence COLLAPSE; ASR/rho not "
              "interpretable.")

    data = {
        "model": args.model, "quantizer": args.quantizer,
        "repair_mode": "closed_form_svd", "role": args.role, "proj_paths": proj_paths,
        "n_bits": args.n_bits, "rank": args.rank, "group_size": args.group_size,
        "band": band, "layer_range": list(layer_range) if layer_range else None,
        "seed": SEED,
        "corrected_effective_bits": inj_eff,
        "whole_model_effective_bits": whole_eff, "bumped_fraction": bumped_frac,
        "baseline_source": base_src, "asr_fp16": asr_fp16, "asr_quant": asr_quant,
        "asr_ptq_untrained": asr["ptq_untrained"], "asr_repaired": asr["closed_form"],
        "arc_ptq_untrained": all_arc["ptq_untrained"], "arc_repaired": all_arc["closed_form"],
        "false_refusal_ptq_untrained": all_xs["ptq_untrained"],
        "false_refusal_repaired": all_xs["closed_form"],
        "rho_ptq_untrained": rho_ptq, "rho_repaired": rho_cf,
        "coherence_collapse_repaired": bool(collapse_cf),
        "arc_chance": arc_chance,
        "eval": {"asr_prompts": args.n_eval, "asr_set": "wildjailbreak/eval",
                 "judge": "harmbench", "arc_questions": args.n_reasoning,
                 "n_xstest": args.n_xstest},
    }
    cfg = RunConfig(model=args.model, quantizer=args.quantizer, variant=variant,
                    extra={"role": args.role, "n_bits": args.n_bits, "band": band,
                           "repair": "closed_form", "seed": SEED})
    out_path = save_result(_SCRIPT, cfg, data)
    print(f"\n> Saved {out_path}")


if __name__ == "__main__":
    main()

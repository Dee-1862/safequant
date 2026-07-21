# Fold trained LoRA delta back into AQLM quantized weights.

# Front Matter
import os
import sys

import torch

from core.datasets import load_arc_challenge
from core.model_loader import load_quantized
from core.restoration import measure_arc
from experiments.trained_repair.qresafe_repair import select_target_modules

AQLM_REPO = os.environ.get("AQLM_REPO", "")
if AQLM_REPO and AQLM_REPO not in sys.path:
    sys.path.insert(0, AQLM_REPO)

from aqlm.utils import _dequantize_weight, pack_int_data, unpack_int_data  # noqa: E402

try:
    from src.aq import QuantizedWeight
except ImportError:
    QuantizedWeight = None


def quant_cfg(model):
    """
    Extract AQLM codebook / group sizes from the model config.

    Inputs:
        - model (nn.Module): Quantized model with quantization_config

    Outputs:
        - cfg (dict): nbits, num_codebooks, in_group_size, out_group_size
    """
    # Normalize config object to a plain dict
    qc = model.config.quantization_config
    if not isinstance(qc, dict):
        qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
    return {
        "nbits": int(qc["nbits_per_codebook"]),
        "num_codebooks": int(qc["num_codebooks"]),
        "in_group_size": int(qc["in_group_size"]),
        "out_group_size": int(qc["out_group_size"]),
    }


def is_quantized_linear(module):
    """
    Return whether a module exposes AQLM codes/codebooks/scales.

    Inputs:
        - module (nn.Module): Candidate module

    Outputs:
        - is_quant (bool): True if quantized linear fields are present
    """
    # Require the three AQLM storage tensors
    return (
        hasattr(module, "codes")
        and hasattr(module, "codebooks")
        and hasattr(module, "scales")
    )


def dequantize_module(module, nbits):
    """
    Dequantize an AQLM linear module to a dense weight matrix.

    Inputs:
        - module (nn.Module): Quantized linear module
        - nbits (int): Bits per codebook entry

    Outputs:
        - weight (Tensor): Dense dequantized weight
    """
    # Unpack signed codes then reconstruct with codebooks/scales
    codes_u = unpack_int_data(module.codes.data, nbits)
    return _dequantize_weight(codes_u, module.codebooks.data, module.scales.data)


def fold_and_writeback_(module, delta, cfg, beam_size = 5):
    """
    Fold a dense delta into AQLM codes and write codes/codebooks/scales back.

    Inputs:
        - module (nn.Module): Quantized linear module (mutated in place)
        - delta (Tensor | None): Dense [out, in] update, or None/zero for pure writeback
        - cfg (dict): Quantization configuration from quant_cfg
        - beam_size (int): Beam width for code updates

    Outputs:
        - changed (float): Fraction of codes that changed
    """
    # Unpack current codes and storage tensors
    nbits = cfg["nbits"]
    codes_u = unpack_int_data(module.codes.data, nbits)          # [out_g,in_g,ncb] >=0
    codebooks = module.codebooks.data
    scales = module.scales.data

    # Non-zero delta: rebuild Codes via QuantizedWeight beam search
    changed = 0.0
    if delta is not None and float(delta.abs().sum()) > 0.0:
        if QuantizedWeight is None:
            raise ImportError("src.aq.QuantizedWeight requires AQLM_REPO on sys.path")
        W = _dequantize_weight(codes_u, codebooks, scales)
        W_target = (W + delta.to(W.dtype)).float()
        qw = QuantizedWeight.__new__(QuantizedWeight)   # bypass kmeans re-init
        torch.nn.Module.__init__(qw)
        qw.out_features, qw.in_features = module.out_features, module.in_features
        qw.out_group_size, qw.in_group_size = cfg["out_group_size"], cfg["in_group_size"]
        qw.num_codebooks = cfg["num_codebooks"]
        qw.nbits_per_codebook = nbits
        qw.codebook_size = 2 ** nbits
        qw.codebook_value_nbits = 16
        qw.codebook_value_num_groups = 1
        qw.codebook_value_clusters = None
        qw.scales_clusters = qw.scales_indices = None
        qw.scale_nbits = 0
        qw.straight_through_gradient = False
        qw.scales_are_lossless = True
        qw.codes_storage = None
        qw.codes = torch.nn.Parameter(codes_u.to(torch.int32), requires_grad = False)
        qw.codebooks = torch.nn.Parameter(codebooks.float(), requires_grad = False)
        qw.scales = torch.nn.Parameter(scales.float(), requires_grad = False)
        new_codes = qw.beam_search_update_codes_(reference_weight = W_target, beam_size = beam_size)
        new_codes_u = new_codes.to(torch.int64)
        changed = float((new_codes_u != codes_u).float().mean())
        codes_u = new_codes_u
        codebooks = qw.get_codebooks().to(codebooks.dtype)
        scales = qw.get_scales().to(scales.dtype)

    # Re-pack codes to the module's signed int dtype and write back
    module.codes.data = pack_int_data(codes_u, nbits).to(module.codes.dtype)
    module.codebooks.data = codebooks.to(module.codebooks.dtype)
    module.scales.data = scales.to(module.scales.dtype)
    return changed


def roundtrip_gate(model, target_names, cfg, atol = 0.0):
    """
    Zero-delta writeback gate: dense weights and codes must stay bit-identical.

    Inputs:
        - model (nn.Module): Quantized model
        - target_names (list[str]): Module names to check
        - cfg (dict): Quantization configuration
        - atol (float): Allowed max absolute dense difference

    Outputs:
        - n_ok (int): Modules that passed
        - n_total (int): Quantized modules checked
        - max_diff (float): Max absolute dense difference observed
    """
    # Walk targets and measure dense + codes identity after delta=0 writeback
    n_ok, n_total, max_diff = 0, 0, 0.0
    mods = dict(model.named_modules())
    for name in target_names:
        m = mods[name]
        if not is_quantized_linear(m):
            continue
        n_total += 1
        W0 = dequantize_module(m, cfg["nbits"]).float()
        codes0 = m.codes.data.clone()
        fold_and_writeback_(m, delta = None, cfg = cfg)          # pure writeback
        W1 = dequantize_module(m, cfg["nbits"]).float()
        d = float((W1 - W0).abs().max())
        codes_identical = bool(torch.equal(m.codes.data, codes0))
        max_diff = max(max_diff, d)
        if d <= atol and codes_identical:
            n_ok += 1
        else:
            print(f"    [mismatch] {name}: max|dW|={d:.2e} codes_identical={codes_identical}")
    return n_ok, n_total, max_diff


def run_gate(model_alias = "llama7b", quantizer = "aqlm2", n_arc = 100):
    """
    Real-checkpoint round-trip gate: bit-identity plus ARC unchanged.

    Inputs:
        - model_alias (str): Model alias to load
        - quantizer (str): Quantizer id
        - n_arc (int): Number of ARC items

    Outputs:
        - ok (bool): True if the gate passed
    """
    # Load checkpoint and resolve targets
    model, tok = load_quantized(model_alias, quantizer = quantizer)
    cfg = quant_cfg(model)
    print("quant cfg:", cfg)
    targets = select_target_modules(model, "full")
    print(f"target modules: {len(targets)}")

    # ARC before/after pure writeback
    arc_items = load_arc_challenge(n_samples = n_arc)
    arc_before = measure_arc(model, tok, arc_items)
    n_ok, n_total, max_diff = roundtrip_gate(model, targets, cfg)
    arc_after = measure_arc(model, tok, arc_items)

    print(f"\n  round-trip: {n_ok}/{n_total} modules bit-identical | max|dW|={max_diff:.2e}")
    print(f"  ARC through bridge: {arc_before:.1f}% -> {arc_after:.1f}% "
          f"(headline aqlm2 ARC was 46%)")
    ok = (
        n_ok == n_total
        and n_total > 0
        and max_diff == 0.0
        and abs(arc_before - arc_after) < 1e-9
    )
    print("\n  ROUND-TRIP GATE:", "PASS" if ok else "FAIL")
    return ok


def delta_fold_smoke(model_alias = "llama7b", quantizer = "aqlm2"):
    """
    Smoke-test non-zero delta fold on one real reader module.

    Inputs:
        - model_alias (str): Model alias to load
        - quantizer (str): Quantizer id

    Outputs:
        - ok (bool): True if codes moved and reconstruction stayed finite
    """
    # Load one reader module and build a small random LoRA-like delta
    model, tok = load_quantized(model_alias, quantizer = quantizer)
    cfg = quant_cfg(model)
    name = select_target_modules(model, "read")[0]
    m = dict(model.named_modules())[name]
    W0 = dequantize_module(m, cfg["nbits"]).float()
    out_f, in_f = m.out_features, m.in_features
    r = 16
    A = torch.randn(r, in_f, device = W0.device) * 0.01
    B = torch.randn(out_f, r, device = W0.device) * 0.01
    delta = (B @ A)
    rel = float(delta.norm() / W0.norm())
    dtype0 = m.codes.dtype
    frac = fold_and_writeback_(m, delta, cfg, beam_size = 5)

    # Measure reconstruction move and dtype preservation
    W1 = dequantize_module(m, cfg["nbits"]).float()
    moved = float((W1 - W0).norm() / W0.norm())
    print(f"  module={name} delta_rel={rel:.4f} codes_changed={100 * frac:.2f}% "
          f"recon_moved={moved:.4f} codes_dtype={m.codes.dtype}")
    ok = (frac > 0.0) and bool(torch.isfinite(W1).all()) and (m.codes.dtype == dtype0)
    print("  DELTA-FOLD SMOKE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "delta":
        sys.exit(0 if delta_fold_smoke() else 1)
    sys.exit(0 if run_gate() else 1)

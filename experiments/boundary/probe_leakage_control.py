# Probe leakage control: FP32 refusal probe vs surface-feature baseline (XSTest + HarmBench).
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from core.datasets import (
    load_harmbench_prompts, load_harmless_prompts,
    load_wildjailbreak_prompts, load_xstest,
)
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _as_text(x):
    """
    Normalize a prompt item to plain text for TF-IDF or logging.

    Inputs:
        - x (str | dict | other): Prompt field or item dict

    Outputs:
        - text (str): User-facing prompt string
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("behavior") or x.get("prompt") or x.get("text") or ""
    return str(x)


def _fit(X, y):
    """
    Fit a logistic regression classifier on feature matrix X.

    Inputs:
        - X: Feature matrix
        - y: Binary labels

    Outputs:
        - clf: Fitted LogisticRegression
    """
    clf = LogisticRegression(C = 1.0, max_iter = 2000, random_state = SEED)
    clf.fit(X, y)
    return clf


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--n_calib", type = int, default = 200)
    ap.add_argument("--n_transfer", type = int, default = 200)
    ap.add_argument("--token_pos", type = int, default = -1)
    args = ap.parse_args()

    # -- prompt sets --
    harmful_txt = [_as_text(x) for x in
                   load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")]
    harmless_txt = [_as_text(x) for x in load_harmless_prompts(n_samples=args.n_calib)]
    xs = load_xstest()
    xs_safe = [it["prompt"] for it in xs if it["type"] == "safe"]
    xs_unsafe = [it["prompt"] for it in xs if it["type"] == "unsafe"]
    hb_txt = [_as_text(x) for x in load_harmbench_prompts(n_samples=args.n_transfer)]
    print(f"[data] wj harmful={len(harmful_txt)} harmless={len(harmless_txt)} "
          f"xstest safe={len(xs_safe)} unsafe={len(xs_unsafe)} harmbench={len(hb_txt)}")

    # -- FP32 residuals for every set --
    model, tok = load_fp32(args.model)
    device = next(model.parameters()).device
    layers = list(range(len(get_inner(model).layers)))

    def resid(texts):
        return extract_last_token_residuals(
            model, tok, texts, layers, device, token_pos=args.token_pos)

    h, s = resid(harmful_txt), resid(harmless_txt)
    safe_r, unsafe_r, hb_r = resid(xs_safe), resid(xs_unsafe), resid(hb_txt)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -- (1)+(3) residual probe per layer: fit on WJ, score XSTest + HarmBench --
    def stack(cap, L):
        return np.stack([t.numpy() for t in cap[L]])

    per_layer = {}
    for L in layers:
        X = np.concatenate([stack(h, L), stack(s, L)])
        y = np.array([1] * len(h[L]) + [0] * len(s[L]))
        clf = _fit(X, y)
        safe_pred = clf.predict(stack(safe_r, L))       # correct = 0 (harmless)
        unsafe_pred = clf.predict(stack(unsafe_r, L))   # correct = 1 (harmful)
        hb_pred = clf.predict(stack(hb_r, L))           # correct = 1 (harmful)
        per_layer[L] = {
            "xstest_safe_specificity": float(np.mean(safe_pred == 0)),
            "xstest_unsafe_recall": float(np.mean(unsafe_pred == 1)),
            "harmbench_transfer_recall": float(np.mean(hb_pred == 1)),
        }

    # best layer by balanced XSTest accuracy (specificity + recall)/2
    def bal(L):
        return 0.5 * (per_layer[L]["xstest_safe_specificity"]
                      + per_layer[L]["xstest_unsafe_recall"])

    best = max(layers, key=bal)

    # -- (2) strong surface baseline: TF-IDF 1-2gram logistic regression --
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    Xtr = vec.fit_transform(harmful_txt + harmless_txt)
    ytr = np.array([1] * len(harmful_txt) + [0] * len(harmless_txt))
    surf = _fit(Xtr, ytr)
    surface = {
        "xstest_safe_specificity": float(np.mean(surf.predict(vec.transform(xs_safe)) == 0)),
        "xstest_unsafe_recall": float(np.mean(surf.predict(vec.transform(xs_unsafe)) == 1)),
        "harmbench_transfer_recall": float(np.mean(surf.predict(vec.transform(hb_txt)) == 1)),
        "n_features": int(len(vec.vocabulary_)),
    }

    b = per_layer[best]
    data = {
        "summary": {
            "best_layer": best,
            "residual_xstest_safe_specificity": b["xstest_safe_specificity"],
            "residual_xstest_unsafe_recall": b["xstest_unsafe_recall"],
            "residual_harmbench_transfer_recall": b["harmbench_transfer_recall"],
            "surface_xstest_safe_specificity": surface["xstest_safe_specificity"],
            "surface_xstest_unsafe_recall": surface["xstest_unsafe_recall"],
            # the money number: how much better the residual probe is at the
            # surface-harmful-but-benign confound cases than a strong text model
            "specificity_gap_residual_minus_surface": (
                b["xstest_safe_specificity"] - surface["xstest_safe_specificity"]),
            "n_xstest_safe": len(xs_safe), "n_xstest_unsafe": len(xs_unsafe),
        },
        "residual_per_layer": {str(L): per_layer[L] for L in layers},
        "surface_tfidf_baseline": surface,
    }
    cfg = RunConfig(
        model=args.model, omit_quantizer=True,
        extra={"n_calib": args.n_calib, "n_transfer": args.n_transfer,
               "token_pos": args.token_pos, "seed": SEED,
               "surface_baseline": "tfidf_ngram_1_2"},
    )
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")
    print(f"[best L{best}] residual: XSTest-safe spec={b['xstest_safe_specificity']:.3f} "
          f"unsafe recall={b['xstest_unsafe_recall']:.3f} "
          f"harmbench transfer={b['harmbench_transfer_recall']:.3f}")
    print(f"           surface : XSTest-safe spec={surface['xstest_safe_specificity']:.3f}  "
          f"(specificity gap = {data['summary']['specificity_gap_residual_minus_surface']:+.3f})")


if __name__ == "__main__":
    main()

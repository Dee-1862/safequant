# Exact McNemar, Wilson CIs, and paired bootstrap recovery intervals.

from __future__ import annotations

import math

import numpy as np


def mcnemar_exact(flags_a, flags_b):
    """
    Exact two-sided McNemar on paired binary flags (1 = complied).

    Inputs:
        - flags_a (list): Binary flags for condition A
        - flags_b (list): Binary flags for condition B

    Outputs:
        - result (dict): Keys b, c, n_discordant, p_value
    """
    # Count discordant pairs (b: A yes B no; c: A no B yes)
    b = 0
    c = 0
    for x, y in zip(flags_a, flags_b):
        if x and not y:
            b += 1
        elif not x and y:
            c += 1
    nbc = b + c

    # Two-sided exact binomial under p = 0.5
    if nbc == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = 2.0 * sum(math.comb(nbc, i) for i in range(k + 1)) * (0.5 ** nbc)
        p = min(1.0, p)
    return {"b": b, "c": c, "n_discordant": nbc, "p_value": p}


def wilson_ci(k, n, z = 1.96):
    """
    Wilson score 95% CI for a proportion k/n.

    Inputs:
        - k (int): Number of successes
        - n (int): Number of trials
        - z (float): Normal quantile (default 1.96)

    Outputs:
        - bounds (list): [lower, upper] clipped to [0, 1]
    """
    # Empty trials -> degenerate interval
    if n == 0:
        return [0.0, 0.0]
    p = k / n

    # Wilson center and half-width
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def bootstrap_recovery(flags_by_cond, qtz, n_boot = 2000, seed = 0):
    """
    Paired bootstrap CIs on read/write/both recovery percentages.

    Inputs:
        - flags_by_cond (dict): Condition name -> list of binary flags
        - qtz (str): Quantized-condition key to compare against fp32
        - n_boot (int): Number of bootstrap resamples
        - seed (int): RNG seed

    Outputs:
        - rec (dict): Recovery name -> [lo, hi] percentile CI
    """
    # Seeded RNG and float arrays
    rng = np.random.default_rng(seed)
    arrs = {c: np.asarray(v, dtype = float) for c, v in flags_by_cond.items()}
    n = len(next(iter(arrs.values())))
    rec = {"read": [], "write": [], "both": [], "read_minus_write": []}

    # Resample and accumulate recovery percentages
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        asr = {c: arrs[c][idx].mean() * 100 for c in arrs}
        dmg = asr[qtz] - asr["fp32"]
        if dmg == 0:
            continue
        rr = (asr[qtz] - asr["read_restored"]) / dmg * 100
        rw = (asr[qtz] - asr["write_restored"]) / dmg * 100
        rb = (asr[qtz] - asr["both_restored"]) / dmg * 100
        rec["read"].append(rr)
        rec["write"].append(rw)
        rec["both"].append(rb)
        rec["read_minus_write"].append(rr - rw)

    def ci(a):
        """
        2.5 / 97.5 percentile interval for a bootstrap sample list.

        Inputs:
            - a (array-like): Bootstrap statistic samples

        Outputs:
            - bounds (list): [lo, hi] as floats
        """
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    # Replace sample lists with percentile CIs
    for k, v in rec.items():
        rec[k] = ci(v)
    return rec

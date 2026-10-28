"""BCa bootstrap interval and Holm correction; the bootstrap resamples physical groups (videos of the same camera)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bca_macro(values: dict, stratum: dict, group: dict, B: int = 10000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Mean over strata, equal weight per stratum and per video within it; groups are resampled within each stratum.
    The two-sided p-value inverts the interval at zero."""
    vals = {k: float(v) for k, v in dict(values).items()}
    strata: dict = {}
    for v in sorted(vals):
        strata.setdefault(stratum[v], {}).setdefault(group[v], []).append(vals[v])
    names = sorted(strata)
    S = {e: np.array([sum(x) for _, x in sorted(strata[e].items())], float) for e in names}
    N = {e: np.array([len(x) for _, x in sorted(strata[e].items())], float) for e in names}
    point = float(np.mean([S[e].sum() / N[e].sum() for e in names]))
    base = {"point": point, "alpha": alpha, "B": int(B), "seed": int(seed),
            "n_groups": int(sum(len(S[e]) for e in names)), "n_videos": len(vals)}
    empty = {"ci": None, "p_value": None}
    if all(len(S[e]) < 2 for e in names):
        return {**base, **empty, "status": "too_few_groups"}
    rng = np.random.default_rng(seed)
    boot = np.zeros(int(B))
    for e in names:
        G = len(S[e])
        if G < 2:
            boot += S[e].sum() / N[e].sum()
            continue
        idx = rng.integers(0, G, size=(int(B), G))
        boot += S[e][idx].sum(axis=1) / N[e][idx].sum(axis=1)
    boot /= len(names)
    if np.ptp(boot) <= 1e-12 * max(1.0, abs(point)):
        return {**base, **empty, "status": "zero_variance", "ci": [point, point]}
    jack = []
    for e in names:
        if len(S[e]) < 2:
            continue
        others = sum(S[o].sum() / N[o].sum() for o in names if o != e)
        for g in range(len(S[e])):
            jack.append((others + (S[e].sum() - S[e][g]) / (N[e].sum() - N[e][g])) / len(names))
    d = np.mean(jack) - np.asarray(jack)
    den = 6.0 * float((d ** 2).sum()) ** 1.5
    a = float((d ** 3).sum() / den) if den > 0 else 0.0
    lim = 0.5 / B

    def fraction_below(x):
        return min(max((np.sum(boot < x) + 0.5 * np.sum(boot == x)) / B, lim), 1 - lim)
    z0 = float(norm.ppf(fraction_below(point)))

    def adjust(q):
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    lo, hi = np.quantile(boot, [adjust(alpha / 2), adjust(1 - alpha / 2)])
    w = float(norm.ppf(fraction_below(0.0))) - z0
    tail = float(norm.cdf(w / (1 + a * w) - z0))
    return {**base, "status": "ok", "ci": [float(lo), float(hi)], "p_value": float(min(1.0, 2 * min(tail, 1 - tail)))}


def holm(pvalues: dict) -> dict:
    members = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m, current, out = len(members), 0.0, {}
    for i, k in enumerate(members):
        current = max(current, min(1.0, (m - i) * float(pvalues[k])))
        out[k] = current
    return out

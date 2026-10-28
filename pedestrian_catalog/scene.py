"""Scene properties of each video and their relation to the gain of each configuration choice.

    python -m pedestrian_catalog scene-properties --paths <local paths> --out <folder>/scene_properties.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from .stats import holm

PROPERTIES = ("density", "overlap", "person_size", "camera_motion", "pedestrian_motion")
# choice: (factor, option, option it is compared with)
CHOICES = {
    "threshold": ("high_score_thresh", 0.6, 0.3),
    "resolution": ("resolution", 1280, 640),
    "detector_size": ("detector", "yolo26m", "yolo26n"),
    "detector_architecture": ("detector", "frcnn_r50_fpn_v2", "yolo26m"),
    "tracker": ("tracker", "botsort", "bytetrack"),
}
FACTORS = ("detector", "resolution", "tracker", "high_score_thresh")
N_PERMUTATIONS = 10_000
SEED = 42
# pedestrian motion compares annotations of a person about 0.4 s apart, at most 0.8 s
LAG_S, LAG_MAX_S = 0.4, 0.8
IGNORE_ID = 1e9                # id of the PersonPath22 ignore regions, set by the converter
TOL = 1e-12


def highest_iou(boxes: np.ndarray) -> np.ndarray:
    """Highest IoU of each (x, y, w, h) box with another box of the same frame."""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    inter = iw * ih
    area = boxes[:, 2] * boxes[:, 3]
    iou = inter / np.maximum(area[:, None] + area[None] - inter, 1e-9)
    np.fill_diagonal(iou, 0.0)
    return iou.max(axis=1)


def annotated_frames(gt: pd.DataFrame, n_frames: int, width: float) -> int:
    """PersonPath22 frames other than keyframes carry an ignore region as wide as the image and are not counted."""
    covered = gt[(gt[1] >= IGNORE_ID) & (gt[4] >= width)][0]
    return int(n_frames - covered[covered <= n_frames].nunique())


def pedestrian_motion(gt: pd.DataFrame, fps: float) -> float:
    """Median speed of the box center between annotations about LAG_S apart, in body heights per second."""
    lag, lag_max = int(round(LAG_S * fps)), int(round(LAG_MAX_S * fps))
    steps = []
    for _, q in gt[(gt[1] < IGNORE_ID) & (gt[5] > 0)].groupby(1):
        q = q.sort_values(0)
        t = q[0].to_numpy(np.int64)
        cx = (q[2] + q[4] / 2.0).to_numpy(float)
        cy = (q[3] + q[5] / 2.0).to_numpy(float)
        h = q[5].to_numpy(float)
        j = np.searchsorted(t, t + lag, side="left")
        i = np.flatnonzero(j < len(t))
        j = j[i]
        dt = t[j] - t[i]
        ok = dt <= lag_max
        i, j, dt = i[ok], j[ok], dt[ok]
        if len(i):
            d = np.hypot(cx[j] - cx[i], cy[j] - cy[i])
            steps.append(d / ((h[i] + h[j]) / 2.0) / (dt / fps))
    return float(np.median(np.concatenate(steps))) if steps else float("nan")


def properties(corpus: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Annotations are filtered to considered pedestrians (class 1) up to the last frame of the video."""
    rows = {}
    for _, c in corpus.iterrows():
        n = int(c["n_frames"])
        g0 = pd.read_csv(Path(data_root) / c["source_path"] / "gt" / "gt.txt", header=None)
        g = g0[(g0[6] == 1) & (g0[7] == 1) & (g0[0] <= n)]
        ious = np.concatenate([highest_iou(q[[2, 3, 4, 5]].to_numpy(float)) for _, q in g.groupby(0)])
        rows[(c["dataset"], c["sequence_id"])] = {
            "density": len(g) / annotated_frames(g0, n, float(c["width"])),
            "overlap": float(ious.mean()),
            "person_size": float((g[5] / float(c["height"])).median()),
            "camera_motion": float(c["camera_motion_median_frac"]),
            "pedestrian_motion": pedestrian_motion(g, float(c["fps"]))}
    out = pd.DataFrame(rows).T[list(PROPERTIES)].astype(float)
    out.index.names = ["dataset", "sequence_id"]
    return out.reset_index().sort_values(["dataset", "sequence_id"]).reset_index(drop=True)


def pairs(factors: pd.DataFrame, factor: str, a, b) -> list[tuple[str, str]]:
    others = [f for f in FACTORS if f != factor]
    meta = factors[list(FACTORS)].to_dict(orient="index")
    k = {c: tuple(m[f] for f in others) for c, m in meta.items()}
    A = {k[c]: c for c, m in meta.items() if m[factor] == a}
    Bm = {k[c]: c for c, m in meta.items() if m[factor] == b}
    return [(A[x], Bm[x]) for x in sorted(A) if x in Bm]


def gains(H: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, (factor, a, b) in CHOICES.items():
        ps = pairs(factors, factor, a, b)
        if not ps:
            raise ValueError(f"no pair of configurations with {factor} = {a!r} and {b!r} and the other factors equal")
        out[name] = pd.concat([H[x] - H[y] for x, y in ps], axis=1).mean(axis=1)
    return pd.DataFrame(out).loc[H.index]


def within_ranks(x: pd.Series, ds: pd.Series) -> pd.Series:
    """Ranks centered within each dataset, so that each dataset weighs in proportion to its size when pooled."""
    out = pd.Series(np.nan, index=x.index, dtype=float)
    for _, idx in x.groupby(ds.loc[x.index]).groups.items():
        r = rankdata(x.loc[idx].to_numpy(float))
        out.loc[idx] = (r - 0.5) / len(idx) - 0.5
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b) / den if den > 0 else float("nan")


def within_spearman(x: pd.Series, y: pd.Series, ds: pd.Series) -> float:
    return _corr(within_ranks(x, ds).to_numpy(), within_ranks(y.loc[x.index], ds).to_numpy())


def permutations_within(ds: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Permutations within each dataset; row k gives, for each video, the position its value is taken from."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(ds == d) for d in sorted(set(ds))]
    P = np.tile(np.arange(len(ds)), (n, 1))
    for k in range(n):
        for g in groups:
            P[k, g] = g[rng.permutation(len(g))]
    return P


def permutation_test(a: np.ndarray, b: np.ndarray, P: np.ndarray) -> tuple[float, float]:
    """rho = corr(a, b) and the two-sided p-value with `a` permuted by the rows of P: (1 + #{|rho*| >= |rho|}) / (1 + n)."""
    observed = _corr(a, b)
    den = float(np.sqrt((a @ a) * (b @ b)))
    perm = (a[P] @ b) / den
    p = (1 + int(np.sum(np.abs(perm) >= abs(observed) - TOL))) / (1 + len(P))
    return observed, p


def relations(G: pd.DataFrame, X: pd.DataFrame, ds: pd.Series, n_perm: int = N_PERMUTATIONS,
              seed: int = SEED) -> pd.DataFrame:
    ds = ds.loc[G.index]
    X = X.loc[G.index]
    P = permutations_within(ds.to_numpy(), n_perm, seed)
    U = {c: within_ranks(X[c], ds) for c in X}
    V = {c: within_ranks(G[c], ds) for c in G}
    rows = []
    for choice in CHOICES:
        for prop in PROPERTIES:
            rho, p = permutation_test(U[prop].to_numpy(), V[choice].to_numpy(), P)
            by_ds = {d: within_spearman(X.loc[ds == d, prop], G.loc[ds == d, choice], ds[ds == d])
                     for d in sorted(ds.unique())}
            rows.append({"choice": choice, "property": prop, "rho": rho, "p_value": p,
                         **{f"rho_{d}": r for d, r in by_ds.items()}})
    out = pd.DataFrame(rows)
    adjusted = holm({i: p for i, p in out["p_value"].items()})
    out.insert(4, "p_holm", [adjusted[i] for i in out.index])
    return out


def write_properties(paths: Path, corpus_csv: Path, out: Path) -> int:
    data_root = Path(yaml.safe_load(Path(paths).read_text(encoding="utf-8"))["data_root"])
    corpus = pd.read_csv(corpus_csv, float_precision="round_trip")
    corpus = corpus[corpus["eligible"].astype(str).str.lower().isin(["true", "1"])]
    X = properties(corpus, data_root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(out, index=False, lineterminator="\n")
    print(f"{len(X)} videos -> {out}")
    return 0

"""The results of an experiment, computed from its exported tables (no GPU).

    python -m pedestrian_catalog analyze [--results <folder of an export>]    # default: paper_results/

Each video weighs the same, the fixed configuration has the highest mean HOTA, the oracle takes the best configuration
of each video, and intervals are 95 % BCa with physical groups resampled.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import results, scene
from .stats import bca_macro

FACTORS = {"detector": "detector", "resolution": "resolution", "tracker": "tracker", "threshold": "high_score_thresh"}
SCENE_FACTOR = {"high_score_thresh": "threshold", "resolution": "resolution", "detector": "detector",
                "tracker": "tracker"}
# the budgets are the frame intervals at these rates
FPS = (30, 25, 20, 14)
# HOTA points below the oracle (coverage) and below the fixed configuration (savings)
TOLERANCES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
SAVING_TOLERANCES = (0.0, 0.5, 1.0, 2.0)
RUNS_BUDGET_MS = 1000 / 30


def factor_table(factors: pd.DataFrame) -> pd.DataFrame:
    P = pd.DataFrame({k: factors[col] for k, col in FACTORS.items()})
    return P.infer_objects()


def varying_factors(P: pd.DataFrame) -> list[str]:
    return [f for f in FACTORS if f in P and P[f].nunique() > 1]


def levels(P: pd.DataFrame, factor: str) -> list:
    return sorted(P[factor].unique(), key=lambda x: (isinstance(x, str), x))


def load(results_dir: Path) -> dict:
    D = results.load(results_dir)
    D["P"] = factor_table(D["factors"])
    D["gains"] = scene.gains(D["H"], D["factors"])
    return D


def bca(D: dict, values: pd.Series) -> list[float] | None:
    """95 % BCa interval of the mean over videos (one stratum, physical groups resampled)."""
    r = bca_macro(values.to_dict(), {v: "all" for v in values.index}, D["group"])
    return r["ci"] if r["status"] == "ok" else None


def fixed_of(H: pd.DataFrame) -> str:
    return H.mean().idxmax()


def under_budget(D: dict, Lc: pd.Series, B: float) -> dict:
    H = D["H"]
    V = Lc.index[Lc <= B + 1e-9]
    if len(V) == 0:
        return {"n": 0}
    Hv = H[V]
    f = fixed_of(Hv)
    g = Hv.max(axis=1) - Hv[f]
    trk = D["P"]["tracker"]
    return {"n": len(V), "n_botsort": int((trk[V] == "botsort").sum()), "fixed": f,
            "fixed_hota": float(Hv[f].mean()), "oracle": float(Hv.max(axis=1).mean()), "gap": float(g.mean()),
            "per_video": g, "fixed_above_B": int((D["L"][f] > B + 1e-9).sum())}


def factor_pairs(P: pd.DataFrame, factor: str) -> list[tuple]:
    """The pairs of levels of a factor: for numeric levels, the larger minus the smaller; otherwise alphabetical."""
    lv = levels(P, factor)
    numeric = all(not isinstance(x, str) for x in lv)
    return [((b, a) if numeric else (a, b)) for a, b in combinations(lv, 2)]


def pair_gain(X: pd.DataFrame, P: pd.DataFrame, fats: list[str], factor: str, a, b) -> tuple[pd.Series, list[float]]:
    """Gain of level a over b, averaged over the other factors per video, and over videos per combination."""
    others = [o for o in fats if o != factor]
    diffs = []
    for _, grp in (P.groupby(others) if others else [(None, P)]):
        ca, cb = grp.index[grp[factor] == a], grp.index[grp[factor] == b]
        if len(ca) == 1 and len(cb) == 1:
            diffs.append(X[ca[0]] - X[cb[0]])
    if not diffs:
        return pd.Series(np.nan, index=X.index), []
    M = pd.concat(diffs, axis=1)
    return M.mean(axis=1), [float(x) for x in M.mean(axis=0)]


def coverage(H: pd.DataFrame, f: str) -> dict:
    """Per tolerance, the videos where the fixed and the best single configuration are within it of the oracle; also the
    smallest tolerance at which the fixed configuration stops being strictly the most often adequate."""
    oracle = H.max(axis=1)
    loss = oracle.to_numpy()[:, None] - H.to_numpy()
    cov = {}
    for e in TOLERANCES:
        A = loss <= e + 1e-12
        cov[f"{e:g}"] = {"best_constant": int(A.sum(0).max()), "fixed": int(A[:, H.columns.get_loc(f)].sum()),
                         "best_constant_id": H.columns[int(A.sum(0).argmax())]}
    jf = H.columns.get_loc(f)
    eps = np.unique(np.round(loss.ravel(), 9))
    eps = eps[eps > 0]
    strictly_until = None
    for e in eps:
        A = (loss <= e + 1e-9).sum(axis=0)
        if A[jf] <= np.delete(A, jf).max():
            strictly_until = float(e)
            break
    return {"by_epsilon": cov, "fixed_strictly_best_below": strictly_until}


def scene_relations(D: dict) -> pd.DataFrame:
    return scene.relations(D["gains"], D["scene"][list(scene.PROPERTIES)], D["dataset"])


def scene_summary(D: dict, fats: list[str], rel: pd.DataFrame) -> dict:
    """Per choice, the property with the largest within-dataset |rho|, the same relation pooled over datasets, and the
    DetA and AssA gains on each half of that property (halves split within each dataset)."""
    pv = D["scene"].join(D["gains"].add_prefix("gain_"))
    ds = D["dataset"]
    out = {"relations": len(rel), "significant": int((rel["p_holm"] < 0.05).sum()), "choices": {}}
    for choice, (column, a_, b_) in scene.CHOICES.items():
        sub = rel[rel["choice"] == choice]
        k = sub["rho"].abs().idxmax()
        prop = rel.loc[k, "property"]
        pooled = {i: float(spearmanr(pv[rel.loc[i, "property"]], pv[f"gain_{choice}"]).statistic) for i in sub.index}
        kp = max(pooled, key=lambda i: abs(pooled[i]))
        fat = SCENE_FACTOR[column]
        row = {"property": prop, "rho": float(rel.loc[k, "rho"]), "p_holm": float(rel.loc[k, "p_holm"]),
               "factor": fat, "a": a_, "b": b_,
               "pooled_property": rel.loc[kp, "property"], "pooled_rho": pooled[kp],
               "pooled_same_relation_and_sign": bool(kp == k and np.sign(pooled[kp]) == np.sign(rel.loc[k, "rho"]))}
        upper = pd.Series(False, index=pv.index)
        for d in D["datasets"]:
            idx = ds.index[ds == d]
            upper[idx] = pv.loc[idx, prop] > pv.loc[idx, prop].median()
        for met in ("DetA", "AssA"):
            gp, _ = pair_gain(D["metrics"][met], D["P"], fats, fat, a_, b_)
            row[f"{met}_upper_half"] = float(gp[upper].mean())
            row[f"{met}_lower_half"] = float(gp[~upper].mean())
        row["videos_upper_half"] = int(upper.sum())
        out["choices"][choice] = row
    out["all_shown_significant"] = all(x["p_holm"] < 0.05 for x in out["choices"].values())
    out["pooled_same_relation_and_sign"] = all(x["pooled_same_relation_and_sign"] for x in out["choices"].values())
    return out


def run_summary(D: dict, budget: float) -> dict:
    """The cost variation of a cell is (max - min) / median over runs, summarized by its median over cells."""
    L = D["L_runs"]
    S = np.stack([X.to_numpy(dtype=float) for X in L.values()])
    hours = D["hours"]
    return {"runs": list(L), "n": len(L), "typical_cost_variation": float(np.median((S.max(0) - S.min(0)) / np.median(S, 0))),
            "budget": budget, "feasible": {r: int((X.max() <= budget + 1e-9).sum()) for r, X in L.items()},
            "hours": hours,
            "mean_hours": {k: float(np.mean([h[k] for h in hours.values()])) for k in ("detection", "tracking")}}


def analyses(D: dict, budgets) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns the results as a nested dict, the scene relations and the per-video savings table."""
    H, ds, L, P = D["H"], D["dataset"], D["L"], D["P"]
    fats = varying_factors(P)
    f = fixed_of(H)
    oracle = H.max(axis=1)
    g = oracle - H[f]
    N: dict = {"fixed": f,
               "fixed_levels": {k: (v.item() if hasattr(v, "item") else v) for k, v in P.loc[f].items()},
               "n_videos": int(len(H)), "n_configs": int(H.shape[1]), "datasets": D["datasets"], "factors": fats,
               "levels": {k: [x.item() if hasattr(x, "item") else x for x in levels(P, k)] for k in fats}}
    cov = coverage(H, f)
    N["gap"] = {
        "fixed_mean": float(H[f].mean()), "fixed_sd": float(H[f].std()),
        "oracle_mean": float(oracle.mean()), "oracle_sd": float(oracle.std()),
        "gap": float(g.mean()), "gap_sd": float(g.std()), "gap_ci": bca(D, g),
        "videos_fixed_not_best": int((g > 1e-9).sum()), "gap_median": float(g.median()),
        "videos_above_5": int((g > 5).sum()), "gap_max": float(g.max()), "video_gap_max": g.idxmax(),
        "distinct_winners": int(H.idxmax(axis=1).nunique()),
        "by_dataset": {d: {"n": int((ds == d).sum()), "fixed": float(H.loc[ds == d, f].mean()),
                           "fixed_sd": float(H.loc[ds == d, f].std()), "oracle": float(oracle[ds == d].mean()),
                           "oracle_sd": float(oracle[ds == d].std()), "gap": float(g[ds == d].mean()),
                           "gap_sd": float(g[ds == d].std())} for d in D["datasets"]},
        "fixed_without_each_dataset": {d: fixed_of(H[ds != d]) for d in D["datasets"] if (ds != d).any()},
        "fixed_without_each_video_other": sorted({fixed_of(H.drop(v)) for v in H.index} - {f}) if len(H) > 1 else [],
    }
    N["gap"]["fixed_stable"] = (all(x == f for x in N["gap"]["fixed_without_each_dataset"].values())
                                and not N["gap"]["fixed_without_each_video_other"])
    if "MOT20" in D["datasets"] and (ds != "MOT20").sum() > 1:
        keep = ds != "MOT20"
        f_wo = fixed_of(H[keep])
        g_wo = H[keep].max(axis=1) - H.loc[keep, f_wo]
        N["gap"]["without_mot20"] = {"n": int(keep.sum()), "fixed": f_wo, "same_fixed": f_wo == f,
                                     "gap": float(g_wo.mean()), "gap_ci": bca(D, g_wo)}
    ordered = np.sort(H.to_numpy(), axis=1)
    if H.shape[1] > 1:
        d12 = ordered[:, -1] - ordered[:, -2]
        N["gap"]["first_second_median_difference"] = float(np.median(d12))
        N["gap"]["videos_first_second_within_0_5"] = int((d12 <= 0.5).sum())
    N["gap"]["coverage"] = cov["by_epsilon"]
    N["gap"]["coverage_fixed_strictly_best_below"] = cov["fixed_strictly_best_below"]
    N["gap"]["videos_within_1_point"] = int((g <= 1 + 1e-12).sum())
    c = D["videos"]
    side = c[["width", "height"]].max(axis=1)
    N["corpus"] = {"videos_16_9": int((c.width * 9 == c.height * 16).sum()), "videos_4k": int((side >= 3840).sum()),
                   "fps_min": float(c.fps.min()), "fps_by_dataset": {d: sorted(set(map(float, c[c.dataset == d].fps)))
                                                                     for d in D["datasets"]},
                   "frames": int(c.n_frames.sum())}
    # oracle drop when each level is removed, and gain of adjusting one factor of the fixed configuration
    removal = {}
    for fat in fats:
        for lv in levels(P, fat):
            keep = P.index[P[fat] != lv]
            removal[f"{fat}={lv}"] = float(oracle.mean() - H[keep].max(axis=1).mean())
    pf = P.loc[f]
    one_factor = {}
    for fat in fats:
        others = [x for x in fats if x != fat]
        keep = P.index[(P[others] == pf[others]).all(axis=1)] if others else P.index
        one_factor[fat] = float((H[keep].max(axis=1) - H[f]).mean())
    # single-factor gains do not add up (the factors interact), so the top three are also adjusted jointly
    top3 = sorted(one_factor, key=lambda k: -one_factor[k])[:3]
    rest = [x for x in fats if x not in top3]
    keep = P.index[(P[rest] == pf[rest]).all(axis=1)] if rest else P.index
    top3_joint = {"factors": top3, "gain": float((H[keep].max(axis=1) - H[f]).mean())}
    contrasts = {}
    for fat in fats:
        for a_, b_ in factor_pairs(P, fat):
            _, by_comb = pair_gain(H, P, fats, fat, a_, b_)
            if by_comb:
                contrasts[f"{a_} - {b_}"] = {"factor": fat, "combinations": len(by_comb),
                                             "positive": int(sum(x > 0 for x in by_comb)),
                                             "mean": float(np.mean(by_comb))}
    N["origin"] = {"removal": removal, "one_factor": one_factor, "one_factor_sum": float(sum(one_factor.values())),
                   "top3_joint": top3_joint,
                   "top3_joint_pct_of_gap": (100 * top3_joint["gain"] / float(g.mean()) if g.mean() > 0 else None),
                   "oracle_without_each_level": {k: float(oracle.mean() - v) for k, v in removal.items()},
                   "contrasts": contrasts}
    if "botsort - bytetrack" in contrasts:
        N["origin"]["mean_difference_botsort_bytetrack"] = contrasts["botsort - bytetrack"]["mean"]
    rel = scene_relations(D)
    N["origin"]["scene"] = scene_summary(D, fats, rel)
    C = {"worst": L.max(), "mean": L.mean()}
    side_v = side.loc[H.index]
    k4 = side_v >= 3840
    trk = P["tracker"]
    bot = [x for x in H.columns if trk[x] == "botsort"]
    budget = {"fixed_cost_mean": float(L[f].mean()), "fixed_cost_sd": float(L[f].std()),
              "fixed_cost_min": float(L[f].min()), "fixed_cost_max": float(L[f].max()),
              "oracle_cost_mean": float(np.mean([L.loc[v, H.loc[v].idxmax()] for v in H.index])),
              "videos_with_cheaper_config_as_good": int(sum(
                  bool(((H.loc[v] >= H.loc[v, f] - 1e-9) & (L.loc[v] < L.loc[v, f])).any()) for v in H.index)),
              "videos_4k": int(k4.sum()),
              "configs_whose_worst_video_is_4k": int(sum(bool(k4[L[c_].idxmax()]) for c_ in H.columns)),
              "cheapest_botsort_worst_without_4k": (float(L.loc[~k4, bot].max().min()) if bot and (~k4).any()
                                                    else None)}
    # per video, the cheapest configuration at least as good as the fixed one; like the oracle, known only in hindsight
    sav = []
    for v in H.index:
        ok = (H.loc[v] >= H.loc[v, f] - 1e-9) & (L.loc[v] < L.loc[v, f])
        if ok.any():
            cb = L.loc[v, H.columns[ok]].idxmin()
            sav.append({"video": v, "config_id": cb, "saving_ms": float(L.loc[v, f] - L.loc[v, cb]),
                        "saving_pct": float(100 * (1 - L.loc[v, cb] / L.loc[v, f])),
                        "hota_gain": float(H.loc[v, cb] - H.loc[v, f]), **P.loc[cb].to_dict()})
    E = pd.DataFrame(sav)
    by_tol = {}
    for e in SAVING_TOLERANCES:
        cs, hs = [], []
        for v in H.index:
            ok = (H.loc[v] >= H.loc[v, f] - e - 1e-9) & (L.loc[v] <= L.loc[v, f] + 1e-12)   # the fixed one always qualifies
            c_ = L.loc[v, H.columns[ok]].idxmin()
            cs.append(float(L.loc[v, c_]))
            hs.append(float(H.loc[v, c_]))
        by_tol[f"{e:g}"] = {"videos_with_cheaper_option": int(sum(
            bool(((H.loc[v] >= H.loc[v, f] - e - 1e-9) & (L.loc[v] < L.loc[v, f])).any()) for v in H.index)),
            "mean_cost": float(np.mean(cs)), "mean_hota": float(np.mean(hs)),
            "reduction_pct": float(100 * (1 - np.mean(cs) / L[f].mean()))}
    budget["savings"] = {
        "videos": int(len(E)),
        "videos_at_least_10pct": int(sum(bool(((H.loc[v] >= H.loc[v, f] - 1e-9) & (L.loc[v] <= 0.9 * L.loc[v, f])).any())
                                         for v in H.index)),
        "videos_at_least_half": int(sum(bool(((H.loc[v] >= H.loc[v, f] - 1e-9) & (L.loc[v] <= 0.5 * L.loc[v, f])).any())
                                        for v in H.index)),
        "by_tolerance": by_tol}
    if len(E):
        budget["savings"].update({
            "saving_ms_median": float(E.saving_ms.median()), "saving_pct_median": float(E.saving_pct.median()),
            "hota_gain_median": float(E.hota_gain.median()),
            "without_botsort": int((E.tracker != "botsort").sum()),
            "lower_resolution_than_fixed": int((E.resolution < pf["resolution"]).sum()),
            "same_detector_as_fixed": int((E.detector == pf["detector"]).sum())})
    for name, Lc in C.items():
        rows = []
        for B, fps in budgets:
            s = under_budget(D, Lc, B)
            if not s["n"]:
                rows.append({"B": B, "fps": fps, "n": 0})
                continue
            diff = s["per_video"] - g
            rows.append({"B": B, "fps": fps, "n": s["n"], "n_botsort": s["n_botsort"], "fixed": s["fixed"],
                         "fixed_hota": s["fixed_hota"], "oracle": s["oracle"],
                         "gap": s["gap"], "gap_ci": bca(D, s["per_video"]),
                         "fixed_above_B": s["fixed_above_B"],
                         "fixed_cost_mean": float(L[s["fixed"]].mean()), "fixed_cost_sd": float(L[s["fixed"]].std()),
                         "fixed_cost_max": float(L[s["fixed"]].max()),
                         "diff_vs_unconstrained": float(diff.mean()), "diff_vs_unconstrained_ci": bca(D, diff)})
        curve = []
        for B in sorted(set(Lc.round(9))):
            s = under_budget(D, Lc, B)
            curve.append((float(B), s["gap"], s["n_botsort"], s["n"], bool(trk[s["fixed"]] == "botsort")))
        imax = int(np.argmax([c_[1] for c_ in curve]))
        with_bot = [c_ for c_ in curve if c_[4]]
        budget[name] = {"budgets": rows, "curve": curve,
                        "botsort_fits_from": (float(min(Lc[x] for x in bot)) if bot else None),
                        "fixed_uses_botsort_from": (with_bot[0][0] if with_bot else None),
                        "gap_when_fixed_switches_to_botsort": (with_bot[0][1] if with_bot else None),
                        "gap_max": curve[imax][1], "gap_max_at": curve[imax][0],
                        "lowest_cost": float(Lc.min()), "highest_cost": float(Lc.max())}
    N["budget"] = budget
    N["runs"] = {"aggregate": "median per cell" if len(D["L_runs"]) > 1 else "single run",
                 **run_summary(D, RUNS_BUDGET_MS)}
    return N, rel, E


def _ci(ci) -> tuple[float, float]:
    return (float(ci[0]), float(ci[1])) if ci else (np.nan, np.nan)


def _split(video: str) -> tuple[str, str]:
    d, s = video.split("/", 1)
    return d, s


def tables(D: dict, N: dict, rel: pd.DataFrame, E: pd.DataFrame) -> dict[str, pd.DataFrame]:
    H, ds, L = D["H"], D["dataset"], D["L"]
    f, G, O, Bu, R = N["fixed"], N["gap"], N["origin"], N["budget"], N["runs"]
    oracle = H.max(axis=1)
    ordered = np.sort(H.to_numpy(), axis=1)
    T: dict[str, pd.DataFrame] = {}

    T["gap_by_video"] = pd.DataFrame({
        "dataset": [_split(v)[0] for v in H.index], "sequence_id": [_split(v)[1] for v in H.index],
        "fixed_hota": H[f].to_numpy(), "oracle_config": H.idxmax(axis=1).to_numpy(), "oracle_hota": oracle.to_numpy(),
        "gap": (oracle - H[f]).to_numpy(), "best_minus_second": ordered[:, -1] - ordered[:, -2]})

    lo, hi = _ci(G["gap_ci"])
    rows = [{"scope": "all", "videos": N["n_videos"], "fixed_config": f, "fixed_mean": G["fixed_mean"],
             "fixed_sd": G["fixed_sd"], "oracle_mean": G["oracle_mean"], "oracle_sd": G["oracle_sd"],
             "gap_mean": G["gap"], "gap_sd": G["gap_sd"], "gap_ci_low": lo, "gap_ci_high": hi}]
    for d, x in G["by_dataset"].items():
        rows.append({"scope": d, "videos": x["n"], "fixed_config": f, "fixed_mean": x["fixed"], "fixed_sd": x["fixed_sd"],
                     "oracle_mean": x["oracle"], "oracle_sd": x["oracle_sd"], "gap_mean": x["gap"],
                     "gap_sd": x["gap_sd"], "gap_ci_low": np.nan, "gap_ci_high": np.nan})
    if "without_mot20" in G:
        w, keep = G["without_mot20"], ds != "MOT20"
        Hk = H[keep]
        lo, hi = _ci(w["gap_ci"])
        rows.append({"scope": "all except MOT20 (fixed elected again)", "videos": w["n"], "fixed_config": w["fixed"],
                     "fixed_mean": float(Hk[w["fixed"]].mean()), "fixed_sd": float(Hk[w["fixed"]].std()),
                     "oracle_mean": float(Hk.max(axis=1).mean()), "oracle_sd": float(Hk.max(axis=1).std()),
                     "gap_mean": w["gap"], "gap_sd": float((Hk.max(axis=1) - Hk[w["fixed"]]).std()),
                     "gap_ci_low": lo, "gap_ci_high": hi})
    T["gap_summary"] = pd.DataFrame(rows)

    rows = [{"left_out": "none", "fixed_config": f}]
    rows += [{"left_out": f"dataset {d}", "fixed_config": c} for d, c in G["fixed_without_each_dataset"].items()]
    rows += [{"left_out": f"video {v}", "fixed_config": fixed_of(H.drop(v))} for v in H.index]
    T["fixed_configuration"] = pd.DataFrame(rows)

    T["coverage"] = pd.DataFrame([{"epsilon": float(e), "videos_fixed_within": x["fixed"],
                                   "videos_best_single_within": x["best_constant"],
                                   "best_single_config": x["best_constant_id"]} for e, x in G["coverage"].items()])

    T["option_removal"] = pd.DataFrame([{"factor": k.split("=", 1)[0], "option": k.split("=", 1)[1],
                                         "oracle_mean_without": O["oracle_without_each_level"][k], "oracle_drop": v}
                                        for k, v in O["removal"].items()])

    gap = G["gap"]
    rows = [{"adjusted_factors": fat, "gain": v, "share_of_gap_pct": 100 * v / gap} for fat, v in O["one_factor"].items()]
    rows.append({"adjusted_factors": "+".join(O["top3_joint"]["factors"]), "gain": O["top3_joint"]["gain"],
                 "share_of_gap_pct": O["top3_joint_pct_of_gap"]})
    T["factor_adjustment"] = pd.DataFrame(rows)

    T["option_contrasts"] = pd.DataFrame([{"factor": x["factor"], "option": k.split(" - ")[0],
                                           "compared_with": k.split(" - ")[1], "combinations": x["combinations"],
                                           "combinations_option_better": x["positive"], "mean_gain": x["mean"]}
                                          for k, x in O["contrasts"].items()])

    T["scene_relations"] = rel

    T["scene_choices"] = pd.DataFrame([{
        "choice": c, "factor": x["factor"], "option": x["a"], "compared_with": x["b"], "property": x["property"],
        "rho": x["rho"], "p_holm": x["p_holm"], "pooled_property": x["pooled_property"], "pooled_rho": x["pooled_rho"],
        "pooled_same_relation_and_sign": x["pooled_same_relation_and_sign"], "videos_upper_half": x["videos_upper_half"],
        "DetA_gain_upper_half": x["DetA_upper_half"], "DetA_gain_lower_half": x["DetA_lower_half"],
        "AssA_gain_upper_half": x["AssA_upper_half"], "AssA_gain_lower_half": x["AssA_lower_half"]}
        for c, x in O["scene"]["choices"].items()])

    T["configuration_cost"] = pd.DataFrame({
        "config_id": L.columns, "cost_mean_ms": L.mean().to_numpy(), "cost_sd_ms": L.std().to_numpy(),
        "cost_worst_video_ms": L.max().to_numpy(), "worst_video": L.idxmax().to_numpy()})

    criteria = {"worst": "worst video", "mean": "mean over videos"}
    rows = []
    for k, name in criteria.items():
        for b in Bu[k]["budgets"]:
            lo, hi = _ci(b.get("gap_ci"))
            dlo, dhi = _ci(b.get("diff_vs_unconstrained_ci"))
            rows.append({"criterion": name, "budget_ms": b["B"], "fps": b["fps"], "feasible": b["n"],
                         "feasible_botsort": b.get("n_botsort", 0), "fixed_config": b.get("fixed", ""),
                         "fixed_hota": b.get("fixed_hota", np.nan), "oracle_hota": b.get("oracle", np.nan),
                         "gap": b.get("gap", np.nan), "gap_ci_low": lo, "gap_ci_high": hi,
                         "gap_minus_unconstrained": b.get("diff_vs_unconstrained", np.nan),
                         "gap_minus_unconstrained_ci_low": dlo, "gap_minus_unconstrained_ci_high": dhi,
                         "fixed_videos_above_budget": b.get("fixed_above_B", np.nan),
                         "fixed_cost_mean_ms": b.get("fixed_cost_mean", np.nan),
                         "fixed_cost_sd_ms": b.get("fixed_cost_sd", np.nan),
                         "fixed_cost_max_ms": b.get("fixed_cost_max", np.nan)})
    T["budget"] = pd.DataFrame(rows)

    T["budget_curve"] = pd.DataFrame([{"criterion": name, "budget_ms": B, "feasible": n, "feasible_botsort": nb,
                                       "gap": gp, "fixed_uses_botsort": fb}
                                      for k, name in criteria.items() for B, gp, nb, n, fb in Bu[k]["curve"]])

    T["savings_by_video"] = pd.DataFrame({
        "dataset": [_split(v)[0] for v in E["video"]] if len(E) else [],
        "sequence_id": [_split(v)[1] for v in E["video"]] if len(E) else [],
        **{c: (E[c].to_numpy() if len(E) else []) for c in ("config_id", "saving_ms", "saving_pct", "hota_gain")}})

    T["savings_by_tolerance"] = pd.DataFrame([{"tolerance": float(e), "videos_with_cheaper_option":
                                               x["videos_with_cheaper_option"], "mean_cost_ms": x["mean_cost"],
                                               "mean_hota": x["mean_hota"], "cost_reduction_pct": x["reduction_pct"]}
                                              for e, x in Bu["savings"]["by_tolerance"].items()])

    T["runs"] = pd.DataFrame([{"run": r, "detection_hours": R["hours"][r]["detection"],
                               "tracking_hours": R["hours"][r]["tracking"],
                               "configs_within_33.3_ms": R["feasible"][r]} for r in R["runs"]])

    S = Bu["savings"]
    items = [
        ("videos", N["n_videos"]), ("configurations", N["n_configs"]), ("fixed_config", f),
        ("videos_where_fixed_is_not_best", G["videos_fixed_not_best"]), ("gap_median", G["gap_median"]),
        ("videos_gap_above_5", G["videos_above_5"]), ("gap_max", G["gap_max"]), ("video_with_gap_max", G["video_gap_max"]),
        ("distinct_oracle_configs", G["distinct_winners"]),
        ("best_minus_second_median", G.get("first_second_median_difference")),
        ("videos_best_minus_second_within_0.5", G.get("videos_first_second_within_0_5")),
        ("videos_within_1_point_of_oracle", G["videos_within_1_point"]),
        ("epsilon_below_which_fixed_is_strictly_most_adequate", G["coverage_fixed_strictly_best_below"]),
        ("fixed_same_without_any_dataset_or_video", G["fixed_stable"]),
        ("single_factor_gains_sum", O["one_factor_sum"]),
        ("scene_relations", O["scene"]["relations"]), ("scene_relations_significant_holm", O["scene"]["significant"]),
        ("fixed_cost_mean_ms", Bu["fixed_cost_mean"]), ("fixed_cost_sd_ms", Bu["fixed_cost_sd"]),
        ("fixed_cost_min_ms", Bu["fixed_cost_min"]), ("fixed_cost_max_ms", Bu["fixed_cost_max"]),
        ("oracle_cost_mean_ms", Bu["oracle_cost_mean"]),
        ("videos_with_cheaper_config_as_good", Bu["videos_with_cheaper_config_as_good"]),
        ("videos_with_cheaper_config_as_good_at_least_10pct", S["videos_at_least_10pct"]),
        ("videos_with_cheaper_config_as_good_at_least_half", S["videos_at_least_half"]),
        ("saving_ms_median", S.get("saving_ms_median")), ("saving_pct_median", S.get("saving_pct_median")),
        ("saving_hota_gain_median", S.get("hota_gain_median")),
        ("cheaper_config_without_botsort", S.get("without_botsort")),
        ("cheaper_config_lower_resolution", S.get("lower_resolution_than_fixed")),
        ("cheaper_config_same_detector", S.get("same_detector_as_fixed")),
        ("videos_4k", Bu["videos_4k"]), ("configs_whose_worst_video_is_4k", Bu["configs_whose_worst_video_is_4k"]),
        ("cheapest_botsort_worst_video_without_4k_ms", Bu["cheapest_botsort_worst_without_4k"]),
    ]
    for k, name in criteria.items():
        x = Bu[k]
        items += [(f"{name}: botsort_feasible_from_ms", x["botsort_fits_from"]),
                  (f"{name}: fixed_uses_botsort_from_ms", x["fixed_uses_botsort_from"]),
                  (f"{name}: gap_when_fixed_switches_to_botsort", x["gap_when_fixed_switches_to_botsort"]),
                  (f"{name}: gap_max", x["gap_max"]), (f"{name}: gap_max_at_ms", x["gap_max_at"]),
                  (f"{name}: lowest_cost_ms", x["lowest_cost"]), (f"{name}: highest_cost_ms", x["highest_cost"])]
    items += [("runs", R["n"]), ("cost_per_cell", R["aggregate"]),
              ("typical_cost_variation_across_runs", R["typical_cost_variation"]),
              ("mean_detection_hours", R["mean_hours"]["detection"]),
              ("mean_tracking_hours", R["mean_hours"]["tracking"]),
              ("videos_16_9", N["corpus"]["videos_16_9"]), ("frames", N["corpus"]["frames"]),
              ("fps_min", N["corpus"]["fps_min"])]
    T["summary"] = pd.DataFrame(items, columns=["quantity", "value"])

    T["inputs"] = pd.DataFrame([{"file": x["path"], "sha256": x["sha256"]} for x in D["sources"].values()])
    return T


def run(results_dir: Path, out: Path | None = None, fps=FPS) -> int:
    budgets = [(1000 / x, int(x) if float(x).is_integer() else x) for x in fps]
    D = load(results_dir)
    if len(D["H"]) < 2:
        raise SystemExit("fewer than two videos: the mean gap, its deviation and its interval do not exist")
    N, rel, E = analyses(D, budgets)
    out = Path(out) if out else Path(results_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    T = tables(D, N, rel, E)
    for name, df in T.items():
        df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    G = N["gap"]
    print(f"fixed configuration {N['fixed']}: HOTA {G['fixed_mean']:.2f}; oracle {G['oracle_mean']:.2f}; "
          f"gap {G['gap']:.2f}")
    print(f"{len(T)} tables in {out}")
    return 0

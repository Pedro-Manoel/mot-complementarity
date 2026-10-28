"""Loads the exported tables of an experiment as video x configuration matrices.

The cost of a cell is the detector plus tracker time per frame, taken as the median over the runs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ("videos.csv", "configurations.csv", "quality.csv", "detection_runs.csv", "tracking_runs.csv",
         "scene_properties.csv")


def read_csv(path: Path) -> pd.DataFrame:
    """Round-trip float parsing, so every value is exactly the one written."""
    return pd.read_csv(path, float_precision="round_trip")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(df: pd.DataFrame) -> pd.Series:
    return df["dataset"] + "/" + df["sequence_id"]


def cost_matrices(detection: pd.DataFrame, tracking: pd.DataFrame, configurations: pd.DataFrame,
                  videos: list[str], configs: list[str]) -> dict[int, pd.DataFrame]:
    """Per run, a video x configuration cost matrix in ms per frame."""
    variant = configurations.set_index("config_id")["detector_variant"]
    out = {}
    for run in sorted(set(tracking["run"])):
        d = detection[detection["run"] == run].assign(video=lambda x: key(x))
        d = d.set_index(["video", "detector_variant"])["ms_per_frame"]
        t = tracking[tracking["run"] == run].assign(video=lambda x: key(x))
        t = t.pivot(index="video", columns="config_id", values="ms_per_frame").loc[videos, configs]
        L = pd.DataFrame(np.nan, index=videos, columns=configs)
        for c in configs:
            L[c] = d.loc[[(v, variant[c]) for v in videos]].to_numpy() + t[c].to_numpy()
        if L.isna().any().any():
            raise SystemExit(f"run {run}: missing cost for some cells")
        out[int(run)] = L
    return out


def median_cost(L: dict[int, pd.DataFrame]) -> pd.DataFrame:
    first = next(iter(L.values()))
    for run, X in L.items():
        if not (X.index.equals(first.index) and X.columns.equals(first.columns)):
            raise SystemExit(f"run {run} does not cover the same cells as the first run")
    S = np.stack([X.to_numpy(dtype=float) for X in L.values()])
    return pd.DataFrame(np.median(S, axis=0), index=first.index, columns=first.columns)


def run_hours(detection: pd.DataFrame, tracking: pd.DataFrame) -> dict[int, dict[str, float]]:
    return {int(r): {"detection": float(detection.loc[detection["run"] == r, "seconds"].sum() / 3600),
                     "tracking": float(tracking.loc[tracking["run"] == r, "seconds"].sum() / 3600)}
            for r in sorted(set(tracking["run"]))}


def load(results_dir: Path) -> dict:
    R = Path(results_dir)
    missing = [f for f in FILES if not (R / f).is_file()]
    if missing:
        raise SystemExit(f"{R}: missing {', '.join(missing)}")
    videos = read_csv(R / "videos.csv")
    videos.index = key(videos)
    configurations = read_csv(R / "configurations.csv")
    quality = read_csv(R / "quality.csv")
    quality["video"] = key(quality)
    index = sorted(videos.index)
    configs = sorted(configurations["config_id"])
    if set(quality["video"]) != set(index) or set(quality["config_id"]) != set(configs) \
            or len(quality) != len(index) * len(configs):
        raise SystemExit("quality.csv is not the full video x configuration matrix of videos.csv and configurations.csv")
    metrics = {m: quality.pivot(index="video", columns="config_id", values=m).loc[index, configs]
               for m in ("HOTA", "DetA", "AssA")}
    detection = read_csv(R / "detection_runs.csv")
    tracking = read_csv(R / "tracking_runs.csv")
    L_runs = cost_matrices(detection, tracking, configurations, index, configs)
    factors = configurations.set_index("config_id").loc[configs]
    scene = read_csv(R / "scene_properties.csv")
    scene.index = key(scene)
    if set(scene.index) != set(index):
        raise SystemExit("scene_properties.csv does not cover the videos of videos.csv")
    return {"H": metrics["HOTA"], "metrics": metrics, "dataset": videos["dataset"].loc[index],
            "group": videos["physical_group_id"].loc[index].to_dict(), "videos": videos.loc[index],
            "factors": factors, "L_runs": L_runs, "L": median_cost(L_runs), "hours": run_hours(detection, tracking),
            "scene": scene.loc[index], "datasets": sorted(videos["dataset"].unique()),
            "sources": {f: {"path": f, "sha256": sha256(R / f)} for f in FILES}}

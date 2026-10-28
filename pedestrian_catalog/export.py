"""Export of one or more runs of an experiment to the tables the analysis reads.

The cost of a cell is the mean detector time per frame (preprocessing, inference, postprocessing) plus the mean tracker
update time per frame; neither includes reading the frame. Repeated runs must have identical quality metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .detection import cell_dir as detection_cell_dir
from .evaluation import collect_metrics
from .provenance import sha256_file
from .schema import stable_id
from .tracking import cell_dir as tracking_cell_dir

VIDEO_COLUMNS = ["dataset", "sequence_id", "n_frames", "fps", "width", "height", "physical_group_id"]


def valid_cell(cell: Path, expected: dict) -> tuple[dict | None, str | None]:
    """(manifest, None) for a complete, intact cell of the expected protocol; otherwise (None, reason)."""
    p = cell / "manifest.json"
    if not p.is_file():
        return None, "no manifest"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "unreadable manifest"
    ident = m.get("identity") or {}
    if m.get("status") != "complete" or m.get("cell_id") != stable_id(ident):
        return None, "incomplete manifest or inconsistent identity"
    differ = sorted(k for k, v in expected.items() if ident.get(k) != v)
    if differ:
        return None, "of another protocol (" + "; ".join(f"{k}: {ident.get(k)!r} instead of {expected[k]!r}"
                                                          for k in differ) + ")"
    for name, meta in (m.get("files") or {}).items():
        if not (cell / name).is_file() or sha256_file(cell / name) != meta["sha256"]:
            return None, f"{name} missing or modified"
    return m, None


def stage_times(run_dir: Path, configs: list[dict], sequences: list[dict], seed: int) -> tuple[list[dict], list[str]]:
    run_dir = Path(run_dir)
    rows, missing = [], []
    for c in configs:
        for s in sequences:
            key = f"{s['dataset']}/{s['sequence_id']}"
            dm, why = valid_cell(detection_cell_dir(run_dir, c, s, seed), {
                "stage": "detection", "cache_key": c["cache_key"], "replicate_seed": int(seed),
                "sequence": {"dataset": s["dataset"], "sequence_id": s["sequence_id"],
                             "frames_manifest_sha256": s["frames_manifest_sha256"]}})
            if dm is None:
                missing.append(f"{c['config_id']} {key}: detection {why}")
                continue
            tcell = tracking_cell_dir(run_dir, c, {"sequence_key": key, "identity": {"replicate_seed": int(seed)}})
            tm, why = valid_cell(tcell, {"stage": "tracking", "config_hash": c["config_hash"],
                                         "detection_cell_id": dm["cell_id"], "fps": float(s["fps"])})
            if tm is None:
                missing.append(f"{c['config_id']} {key}: tracking {why}")
                continue
            d = float(pd.read_parquet(Path(dm["cell_dir"]) / "frames_index.parquet")["elapsed_ms_cache_pass"].mean())
            u = float(pd.read_parquet(tcell / "frames.parquet")["elapsed_ms_update"].mean())
            rows.append({"config_id": c["config_id"], "detector_variant": c["detector_variant_id"],
                         "dataset": s["dataset"], "sequence_id": s["sequence_id"], "detector_ms": d, "tracker_ms": u,
                         "detection_seconds": float(dm["research_cost"]["seconds_this_call"]),
                         "tracking_seconds": float(tm["research_cost"]["seconds_this_call"])})
    return rows, missing


def configuration_table(configs: list[dict]) -> pd.DataFrame:
    rows = []
    for c in configs:
        p = c["tracker"]["profile"]
        rows.append({"config_id": c["config_id"], "detector": c["detector"]["name"],
                     "resolution": int(c["resolution"]["requested"]), "tracker": c["tracker"]["family"],
                     "high_score_thresh": float(p["high_score_thresh"]), "lost_track_seconds": float(p["lost_track_seconds"]),
                     "confirm_frames": int(p["confirm_frames"]), "detector_variant": c["detector_variant_id"]})
    return pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)


def quality_table(run_dir: Path, configs: list[dict], sequences: list[dict], seeds: list[int],
                  do_preproc: bool) -> pd.DataFrame:
    rows, problems = collect_metrics(run_dir, configs, sequences, seeds, do_preproc)
    if problems:
        raise ValueError(f"{run_dir}: {len(problems)} cells without a valid evaluation, e.g. {problems[0]}")
    q = pd.DataFrame(rows).drop(columns=["evaluation_cell_id", "seed"])
    first = ["dataset", "sequence_id", "config_id"]
    q = q[first + [c for c in q.columns if c not in first]]
    return q.sort_values(first).reset_index(drop=True)


def export(run_dirs: list[Path], configs: list[dict], corpus: list[dict], seeds: list[int], do_preproc: bool,
           out: Path) -> dict:
    """Runs are numbered 1, 2, ... in the given order."""
    sequences = [s for s in corpus if s["eligible"]]
    if len(seeds) != 1:
        raise ValueError("the export expects one replicate seed per run")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    quality, det_rows, trk_rows = None, [], []
    for run, run_dir in enumerate(run_dirs, 1):
        q = quality_table(run_dir, configs, sequences, seeds, do_preproc)
        if quality is None:
            quality = q
        elif not q.equals(quality):
            raise ValueError(f"{run_dir}: the quality metrics differ from those of {run_dirs[0]}; "
                             "the runs are not repetitions of the same experiment")
        times, missing = stage_times(run_dir, configs, sequences, seeds[0])
        if missing:
            raise ValueError(f"{run_dir}: {len(missing)} cells without valid times, e.g. {missing[0]}")
        seen = {}
        for x in times:
            k = (x["dataset"], x["sequence_id"], x["detector_variant"])
            if seen.setdefault(k, (x["detector_ms"], x["detection_seconds"])) != (x["detector_ms"], x["detection_seconds"]):
                raise ValueError(f"{run_dir}: configurations of {k} read different detection cells")
            trk_rows.append({"run": run, "dataset": x["dataset"], "sequence_id": x["sequence_id"],
                             "config_id": x["config_id"], "ms_per_frame": x["tracker_ms"],
                             "seconds": x["tracking_seconds"]})
        det_rows += [{"run": run, "dataset": d, "sequence_id": s, "detector_variant": v, "ms_per_frame": ms,
                      "seconds": sec} for (d, s, v), (ms, sec) in seen.items()]
    videos = pd.DataFrame(sequences)[VIDEO_COLUMNS].sort_values(["dataset", "sequence_id"]).reset_index(drop=True)
    tables = {
        "videos.csv": videos,
        "configurations.csv": configuration_table(configs),
        "quality.csv": quality,
        "detection_runs.csv": pd.DataFrame(det_rows).sort_values(["run", "dataset", "sequence_id", "detector_variant"]),
        "tracking_runs.csv": pd.DataFrame(trk_rows).sort_values(["run", "dataset", "sequence_id", "config_id"]),
    }
    for name, t in tables.items():
        t.to_csv(out / name, index=False, lineterminator="\n")
    return {"runs": [str(r) for r in run_dirs], "videos": len(videos), "configurations": len(configs),
            "files": sorted(tables)}

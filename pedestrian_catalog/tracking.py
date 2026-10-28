"""Tracking over the detection cache; the tracker is updated on every frame, including frames without detections.

BoxMOT numbers tracks with a per-process global counter, so identities are renumbered by order of appearance and the
raw id is kept in `raw_track_id`.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .backends import trackers as BT
from .catalog import lowest_score_field
from . import progress as PG
from . import resources as RS
from .detection import InvalidCache, _now, _event, _read_frame, _samples_table, load_detections
from .provenance import sha256_file, write_json_atomic, write_text_atomic
from . import runtime as RT
from .schema import SCHEMA_VERSION, stable_id

TRACK_COLUMNS = ["frame_id", "track_id", "raw_track_id", "x1", "y1", "x2", "y2", "conf", "cls", "det_index"]
FRAME_COLUMNS = ["frame_id", "n_input_detections", "n_tracks", "elapsed_ms_update", "elapsed_ms_read"]

_TRACK_SCHEMA = pa.schema([("frame_id", pa.int32()), ("track_id", pa.int32()), ("raw_track_id", pa.int64()),
                           ("x1", pa.float32()), ("y1", pa.float32()), ("x2", pa.float32()), ("y2", pa.float32()),
                           ("conf", pa.float32()), ("cls", pa.float32()), ("det_index", pa.int32())])
_FRAME_SCHEMA = pa.schema([("frame_id", pa.int32()), ("n_input_detections", pa.int32()), ("n_tracks", pa.int32()),
                           ("elapsed_ms_update", pa.float64()), ("elapsed_ms_read", pa.float64())])


def tracking_cell_identity(config: dict, det_manifest: dict, fps: float, versions: dict, reid_sha256=None,
                           environment=None) -> dict:
    if not environment:
        raise ValueError("the tracking identity requires the execution environment (runtime.execution_environment)")
    family, profile = config["tracker"]["family"], config["tracker"]["profile"]
    ident = {"stage": "tracking", "config_hash": config["config_hash"], "config_id": config["config_id"],
             "tracker_family": family, "detection_cell_id": det_manifest["cell_id"],
             "fps": float(fps), "tracker_versions": dict(sorted(versions.items())), "environment": dict(environment)}
    if BT.needs_image(family):
        if profile.get("with_reid") and not reid_sha256:
            raise ValueError("BoT-SORT with ReID requires the hash of the ReID weights in the cell identity")
        ident["reid_sha256"] = reid_sha256
    return {"cell_id": stable_id(ident), **ident}


def cell_dir(run_dir, config: dict, det_manifest: dict) -> Path:
    seq = det_manifest["sequence_key"].replace("/", "__")
    return Path(run_dir) / "tracks" / config["config_hash"] / seq / f"seed{det_manifest['identity']['replicate_seed']}"


def _check_existing(cell: Path, ident: dict) -> dict:
    m = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
    if m.get("cell_id") != ident["cell_id"] or m.get("identity") != {k: v for k, v in ident.items() if k != "cell_id"}:
        raise InvalidCache(f"{cell}: the existing tracks manifest has a different identity; nothing was overwritten")
    for fname, meta in m["files"].items():
        p = cell / fname
        if not p.is_file() or sha256_file(p) != meta["sha256"]:
            raise InvalidCache(f"{cell.name}: {fname} missing or tampered with")
    return m


def _write_parquet(tbl: pa.Table, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp")
    pq.write_table(tbl, tmp)
    tmp.replace(dest)


def track_sequence(sequence: dict, config: dict, cache: dict, run: dict) -> dict:
    """`cache` is the detection manifest; `run` holds run_dir, factory(config, fps), versions, environment and,
    for BoT-SORT, reid_sha256."""
    family, profile = config["tracker"]["family"], config["tracker"]["profile"]
    conf_floor = float(cache["detection_params"]["conf_floor"])
    threshold = float(profile[lowest_score_field(family)])
    if threshold < conf_floor:
        raise ValueError(f"{config['config_id']}: the tracker consumes scores from {threshold}, below the floor "
                         f"{conf_floor} of the cache; the cache does not have those detections")
    fps = float(sequence["fps"])
    ident = tracking_cell_identity(config, cache, fps, run.get("versions") or {}, run.get("reid_sha256"),
                                   run.get("environment"))
    cell = cell_dir(run["run_dir"], config, cache)
    if (cell / "manifest.json").is_file():
        m = _check_existing(cell, ident)
        _event(run["run_dir"], "track", ident["cell_id"], "skipped")
        return m
    _event(run["run_dir"], "track", ident["cell_id"], "running")
    try:
        m = _track(sequence, config, cache, run, ident, cell, fps)
    except Exception as e:
        _event(run["run_dir"], "track", ident["cell_id"], "failed", error=f"{type(e).__name__}: {e}")
        raise
    _event(run["run_dir"], "track", ident["cell_id"], "complete")
    return m


def _track(sequence, config, cache, run, ident, cell: Path, fps: float) -> dict:
    t0 = time.perf_counter()
    idx, dets = load_detections(cache)
    n = int(sequence["n_frames"])
    if list(idx["frame_id"]) != list(range(1, n + 1)):
        raise InvalidCache(f"{cache['sequence_key']}: the cache index does not cover frames 1..{n}")
    family = config["tracker"]["family"]
    pixels = BT.needs_image(family)
    tracker = run["factory"](config, fps)
    dets = dets.sort_values(["frame_id", "detection_id"], kind="stable")
    frame_ids = dets["frame_id"].to_numpy()
    boxes = dets[["x1", "y1", "x2", "y2", "score", "class_id"]].to_numpy(np.float32)
    starts = np.searchsorted(frame_ids, np.arange(1, n + 2))
    dims = idx.set_index("frame_id")[["orig_height", "orig_width"]]
    new_id_of: dict[int, int] = {}
    trk = {c: [] for c in TRACK_COLUMNS}
    frames = {c: [] for c in FRAME_COLUMNS}
    rec = run.get("resources") or {}
    # monitor the frame loop only, not the cache reading or the output writing
    with RS.ResourceMonitor(rec.get("device", "cpu"), sample_hz=float(rec.get("sample_hz", 4.0))) as monitor:
        _frame_loop(n, boxes, starts, pixels, sequence, dims, tracker, new_id_of, trk, frames)
    usage = monitor.summary()
    usage.update(RS.per_frame(usage, n))
    cell.mkdir(parents=True, exist_ok=True)
    return _write_cell(sequence, config, cache, run, ident, cell, fps, n, family, tracker, new_id_of, trk, frames,
                       usage, monitor.samples(), t0)


def _frame_loop(n, boxes, starts, pixels, sequence, dims, tracker, new_id_of, trk, frames) -> None:
    for i in range(1, n + 1):
        d = boxes[starts[i - 1]:starts[i]]
        r0 = time.perf_counter()
        if pixels:
            frame = _read_frame(sequence, i)
        else:
            h, w = (int(x) for x in dims.loc[i])
            frame = np.zeros((h, w, 3), np.uint8)
        r1 = time.perf_counter()
        out = np.asarray(tracker.update(d, frame), dtype=np.float64)
        out = out.reshape(-1, out.shape[-1]) if out.size else np.empty((0, 8))
        r2 = time.perf_counter()
        for row in out:
            raw = int(row[4])
            new = new_id_of.setdefault(raw, len(new_id_of) + 1)
            for c, v in zip(TRACK_COLUMNS, (i, new, raw, row[0], row[1], row[2], row[3],
                                             row[5] if len(row) > 5 else np.nan, row[6] if len(row) > 6 else np.nan,
                                             int(row[7]) if len(row) > 7 else -1)):
                trk[c].append(v)
        for c, v in zip(FRAME_COLUMNS, (i, len(d), len(out), (r2 - r1) * 1000.0, (r1 - r0) * 1000.0)):
            frames[c].append(v)


def _write_cell(sequence, config, cache, run, ident, cell: Path, fps: float, n: int, family: str, tracker, new_id_of,
                trk, frames, usage: dict, sample_rows: list[dict], t0: float) -> dict:
    tbl = pa.table(trk, schema=_TRACK_SCHEMA).sort_by([("frame_id", "ascending"), ("track_id", "ascending")])
    _write_parquet(tbl, cell / "tracks.parquet")
    _write_parquet(pa.table(frames, schema=_FRAME_SCHEMA), cell / "frames.parquet")
    _write_parquet(_samples_table(sample_rows, 0), cell / "resources.parquet")
    buf = io.StringIO()
    for f, tid, x1, y1, x2, y2, conf in zip(*(tbl.column(c).to_pylist()
                                               for c in ("frame_id", "track_id", "x1", "y1", "x2", "y2", "conf"))):
        c = 1.0 if conf is None or conf != conf else conf
        buf.write(f"{f},{tid},{x1:.3f},{y1:.3f},{x2 - x1:.3f},{y2 - y1:.3f},{c:.5f},-1,-1,-1\n")
    write_text_atomic(cell / "tracks_mot.txt", buf.getvalue())
    empty_frames = [q for q, k in zip(frames["frame_id"], frames["n_tracks"]) if k == 0]
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "created_utc": _now(),
        "cell_id": ident["cell_id"], "identity": {k: v for k, v in ident.items() if k != "cell_id"},
        "cell_dir": str(cell),
        "config_id": config["config_id"], "sequence_key": cache["sequence_key"], "detection_cell_id": cache["cell_id"],
        "replicate_seed": cache["identity"]["replicate_seed"], "fps": fps, "tracker_family": family,
        "tracker_profile": dict(config["tracker"]["profile"]), "tracker_resolved": getattr(tracker, "resolved", None),
        "counts": {"n_frames": n, "n_track_rows": tbl.num_rows, "n_tracks": len(new_id_of),
                   "n_frames_without_tracks": len(empty_frames)},
        "frames_without_tracks": empty_frames,
        "timing": {"clocks": {"elapsed_ms_update": "tracker: association and, for BoT-SORT, ReID and camera motion "
                                                   "compensation",
                              "elapsed_ms_read": "reading the boxes of the frame and, for BoT-SORT, the frame itself"},
                   "update_ms_mean": (float(np.mean(frames["elapsed_ms_update"])) if frames["elapsed_ms_update"]
                                      else None),
                   "read_ms_mean": float(np.mean(frames["elapsed_ms_read"])) if frames["elapsed_ms_read"] else None},
        "resources": usage,
        "research_cost": {"seconds_this_call": round(time.perf_counter() - t0, 3),
                          "note": "duration of tracking from the cache; not the protocol clock"},
        "files": {fname: {"sha256": sha256_file(cell / fname), "size": (cell / fname).stat().st_size}
                  for fname in ("tracks.parquet", "frames.parquet", "tracks_mot.txt", "resources.parquet")},
    }
    write_json_atomic(cell / "manifest.json", manifest)
    return manifest


def run_tracking(configs: list[dict], variants: dict, sequences: list[dict], seeds: list[int], run_dir, *,
                 factory, tracker_versions: dict, detection_versions_of, reid_sha256, device: str,
                 seeds_by_seq: dict | None = None, sample_hz: float = 4.0) -> dict:
    """All configuration x sequence x seed cells, resuming; a cell without a detection cache is not a failure."""
    from . import detection as D
    from .replicates import seeds_for
    run_dir = Path(run_dir)
    expected_cells, completed, failures, no_detection = [], [], [], []
    # BoT-SORT runs ReID on the requested device (results can differ from the CPU); the other trackers run on the CPU
    env_device, env_cpu = RT.execution_environment(device), RT.execution_environment("cpu")
    RT.require_run_environment(run_dir, "tracking", {"device": env_device, "cpu": env_cpu})
    env_detection = RT.run_environment(run_dir, "detection")
    # cells left for the progress counter; only the manifest existence is checked here
    already_done, to_do = set(), {}
    for c in configs:
        v = variants[c["detector_variant_id"]]
        for s in sequences:
            for seed in seeds_for(s, seeds, seeds_by_seq):
                key = (c["config_id"], f"{s['dataset']}/{s['sequence_id']}", int(seed))
                if not (D.cell_dir(run_dir, v, s, seed) / "manifest.json").is_file():
                    continue
                tpath = (run_dir / "tracks" / c["config_hash"] / f"{s['dataset']}__{s['sequence_id']}"
                         / f"seed{int(seed)}")
                if (tpath / "manifest.json").is_file():
                    already_done.add(key)
                else:
                    to_do[key] = int(s["n_frames"])
    prog = PG.Progress("track", len(to_do), sum(to_do.values()))
    prog.info(f"{len(to_do)} cells to do ({len(already_done)} already complete), {int(prog.work_total)} frames")
    for c in configs:
        v = variants[c["detector_variant_id"]]
        dver = detection_versions_of(v["detector"]["backend"])
        extra = ({"reid_sha256": reid_sha256, "environment": env_device} if BT.needs_image(c["tracker"]["family"])
                 else {"environment": env_cpu})
        for s in sequences:
            for seed in seeds_for(s, seeds, seeds_by_seq):
                key = {"config_id": c["config_id"], "sequence": f"{s['dataset']}/{s['sequence_id']}", "seed": int(seed)}
                expected_cells.append(key)
                dcell = D.cell_dir(run_dir, v, s, seed)
                if not (dcell / "manifest.json").is_file():
                    no_detection.append(key)
                    continue
                if env_detection is None:
                    failures.append({**key, "error": "the run did not record the detection environment (older run "
                                                     "format): redo the detection in a new run"})
                    continue
                new = (key["config_id"], key["sequence"], key["seed"]) in to_do
                where = f"{c['config_id']} {key['sequence']}"
                try:
                    cache = D._check_cell(dcell, D.detection_cell_identity(v, s, seed, dver, env_detection))
                    m = track_sequence(s, c, cache, {"run_dir": run_dir, "factory": factory,
                                                     "versions": tracker_versions,
                                                     "resources": {"device": device, "sample_hz": sample_hz}, **extra})
                except (InvalidCache, OSError, RuntimeError, ValueError) as e:
                    failures.append({**key, "error": f"{type(e).__name__}: {e}"})
                    if new:
                        prog.advance(f"{where}: {type(e).__name__}: {e}", int(s["n_frames"]), failed=True)
                    else:
                        PG.say(f"[track] FAILED {where} (existing cell): {type(e).__name__}: {e}")
                    continue
                completed.append({**key, "cell_id": m["cell_id"], "detection_cell_id": cache["cell_id"]})
                if new:
                    timing, extra_txt = m.get("timing") or {}, PG.resources_text(m.get("resources"))
                    prog.advance(f"{where}: {m['counts']['n_frames']} frames in "
                                 f"{PG.fmt_duration((m.get('resources') or {}).get('duration_s'))} (update "
                                 f"{timing.get('update_ms_mean') or float('nan'):.1f} ms/frame, "
                                 f"{m['counts']['n_tracks']} tracks)"
                                 + (f" {extra_txt}" if extra_txt else ""), int(s["n_frames"]))
    done = {(x["config_id"], x["sequence"], x["seed"]) for x in completed + failures + no_detection}
    report = {"schema_version": SCHEMA_VERSION, "created_utc": _now(), "stage": "tracking", "device": str(device),
              "expected": len(expected_cells), "complete": completed, "failed": failures,
              "missing_detection": no_detection,
              "missing": [x for x in expected_cells if (x["config_id"], x["sequence"], x["seed"]) not in done]}
    write_json_atomic(run_dir / "reports" / "tracking_coverage.json", report)
    return report


def direct_tracks(sequence: dict, config: dict, backend, factory) -> pd.DataFrame:
    """Detector and tracker on the real frames, without the cache; used to check the cached path."""
    n = int(sequence["n_frames"])
    tracker = factory(config, float(sequence["fps"]))
    new_id_of: dict[int, int] = {}
    trk = {c: [] for c in TRACK_COLUMNS}
    for i in range(1, n + 1):
        frame = _read_frame(sequence, i)
        dets, _ = backend.detect(frame)
        out = np.asarray(tracker.update(np.asarray(dets, np.float32).reshape(-1, 6), frame), dtype=np.float64)
        out = out.reshape(-1, out.shape[-1]) if out.size else np.empty((0, 8))
        for row in out:
            raw = int(row[4])
            new = new_id_of.setdefault(raw, len(new_id_of) + 1)
            for c, v in zip(TRACK_COLUMNS, (i, new, raw, row[0], row[1], row[2], row[3],
                                             row[5] if len(row) > 5 else np.nan, row[6] if len(row) > 6 else np.nan,
                                             int(row[7]) if len(row) > 7 else -1)):
                trk[c].append(v)
    return (pa.table(trk, schema=_TRACK_SCHEMA).sort_by([("frame_id", "ascending"), ("track_id", "ascending")])
            .to_pandas())


def compare_tracks(cache: pd.DataFrame, direct: pd.DataFrame, tol: float = 1e-3) -> dict:
    keys = ["frame_id", "track_id"]
    a = cache.sort_values(keys).reset_index(drop=True)
    b = direct.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not (a[keys].to_numpy() == b[keys].to_numpy()).all():
        return {"equal": False, "rows": [len(a), len(b)], "max_abs_box_diff": None,
                "reason": "different rows or identities"}
    diff = (float(np.abs(a[["x1", "y1", "x2", "y2"]].to_numpy(float)
                         - b[["x1", "y1", "x2", "y2"]].to_numpy(float)).max())
            if len(a) else 0.0)
    return {"equal": diff <= tol, "rows": [len(a), len(b)], "max_abs_box_diff": diff,
            "reason": None if diff <= tol else f"boxes differ by up to {diff:.4f} px"}


def load_tracks(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = Path(manifest["cell_dir"])
    for fname, meta in manifest["files"].items():
        if not (cell / fname).is_file() or sha256_file(cell / fname) != meta["sha256"]:
            raise InvalidCache(f"{cell.name}: {fname} missing or tampered with")
    return pq.read_table(cell / "tracks.parquet").to_pandas(), pq.read_table(cell / "frames.parquet").to_pandas()

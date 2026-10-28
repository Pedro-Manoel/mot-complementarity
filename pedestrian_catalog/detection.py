"""Detection cache: person detections above the score floor, per detector variant, sequence and seed.

Boxes are in original-image coordinates; the full model output is kept in `raw_detections.parquet` and the person
cache is checked to be exactly its subset. A cell is built in parts of `part_size` frames so that it can resume.
"""
from __future__ import annotations

import json
import math
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import sha256_file, verify_checkpoint, write_json_atomic
from . import progress as PG
from . import runtime as RT
from .schema import SCHEMA_VERSION, stable_id

DETECTION_COLUMNS = ["frame_id", "detection_id", "x1", "y1", "x2", "y2", "score", "class_id"]
INDEX_COLUMNS = ["frame_id", "file", "read_status", "orig_height", "orig_width", "n_detections",
                 "n_before_nms", "input_height", "input_width", "elapsed_ms_cache_pass", "n_raw_detections",
                 "elapsed_ms_decode"]
RAW_COLUMNS = ["frame_id", "detection_id", "x1", "y1", "x2", "y2", "score", "label"]
PERSON = 0
# fraction of the image a box center may fall outside (backend rounding); larger offsets mean wrong coordinates
BORDER_TOLERANCE = 0.01

_DET_SCHEMA = pa.schema([("frame_id", pa.int32()), ("detection_id", pa.int32()), ("x1", pa.float32()),
                         ("y1", pa.float32()), ("x2", pa.float32()), ("y2", pa.float32()),
                         ("score", pa.float32()), ("class_id", pa.int16())])
_IDX_SCHEMA = pa.schema([("frame_id", pa.int32()), ("file", pa.string()), ("read_status", pa.string()),
                         ("orig_height", pa.int32()), ("orig_width", pa.int32()), ("n_detections", pa.int32()),
                         ("n_before_nms", pa.int32()), ("input_height", pa.int32()), ("input_width", pa.int32()),
                         ("elapsed_ms_cache_pass", pa.float64()), ("n_raw_detections", pa.int32()),
                         ("elapsed_ms_decode", pa.float64())])
_RAW_SCHEMA = pa.schema([("frame_id", pa.int32()), ("detection_id", pa.int32()), ("x1", pa.float32()),
                         ("y1", pa.float32()), ("x2", pa.float32()), ("y2", pa.float32()),
                         ("score", pa.float32()), ("label", pa.int16())])


class InvalidCache(RuntimeError):
    """An existing cell has another identity or a missing or tampered file."""


class InvalidOutput(ValueError):
    """The backend returned something that violates the cache contract."""


class UnreadableFrame(RuntimeError):
    """A corpus frame could not be read."""


def sha256_of(path) -> str:
    return sha256_file(path)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_input_hw(backend: str, requested: int, height: int, width: int, stride: int = 32) -> tuple[int, int]:
    """The input shape each backend should produce, to check what the hooks capture; it never replaces the capture."""
    if backend == "ultralytics":
        side = int(math.ceil(requested / stride) * stride)
        r = min(side / height, side / width)
        h, w = int(round(height * r)), int(round(width * r))
        return h + (side - h) % stride, w + (side - w) % stride
    if backend == "dfine":
        return int(requested), int(requested)
    if backend == "torchvision":
        scale = min(requested / min(height, width), requested / max(height, width))
        h, w = int(math.floor(height * scale)), int(math.floor(width * scale))
        return int(math.ceil(h / stride) * stride), int(math.ceil(w / stride) * stride)
    raise ValueError(f"unknown backend: {backend!r}")


def detection_cell_identity(variant: dict, seq_info: dict, seed_value: int, versions: dict, environment: dict) -> dict:
    """Includes the execution environment, so a CPU cache is never reused by a CUDA run or the reverse."""
    if not environment:
        raise ValueError("the detection identity requires the execution environment (runtime.execution_environment)")
    ident = {
        "stage": "detection",
        "cache_key": variant["cache_key"],
        "detector_variant_id": variant.get("detector_variant_id"),
        "checkpoint_sha256": variant["checkpoint"]["sha256"],
        "sequence": {"dataset": seq_info["dataset"], "sequence_id": seq_info["sequence_id"],
                     "frames_manifest_sha256": seq_info["frames_manifest_sha256"]},
        "replicate_seed": int(seed_value),
        "backend_versions": dict(sorted(versions.items())),
        "environment": dict(environment),
    }
    return {"cell_id": stable_id(ident), **ident}


def cell_dir(run_dir, variant: dict, seq_info: dict, seed_value: int) -> Path:
    return (Path(run_dir) / "detections" / variant["cache_key"] /
            f"{seq_info['dataset']}__{seq_info['sequence_id']}" / f"seed{int(seed_value)}")


_EVENTS_LOCK = threading.Lock()     # parallel evaluation writes events from several threads


def _event(run_dir, stage: str, cell_id: str, state: str, **extra) -> None:
    line = json.dumps({"ts": _now(), "stage": stage, "cell": cell_id, "state": state, **extra},
                      ensure_ascii=False)
    p = Path(run_dir) / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _EVENTS_LOCK, open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def _seed_all(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value % (2 ** 32))
    if "torch" in sys.modules:                     # never imports torch just to seed it
        torch = sys.modules["torch"]
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)


def _frame_name(seq: dict, i: int) -> str:
    return f"{i:0{int(seq['index_width'])}d}{seq['im_ext']}"


def _read_frame(seq: dict, i: int):
    import cv2
    path = Path(seq["dir"]) / seq["im_dir"] / _frame_name(seq, i)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise UnreadableFrame(f"unreadable frame: {seq['dataset']}/{seq['sequence_id']}/{_frame_name(seq, i)}")
    return img


def _degenerate_mask(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])


def drop_degenerate(dets) -> tuple[np.ndarray, int]:
    """Drops person boxes with no width or height (YOLO26-x emits a few) and counts them; the raw output keeps them.

    Not part of the cell identity: before this rule such a box stopped the cell, so no stored cell has one."""
    dets = np.asarray(dets, np.float32).reshape(-1, 6)
    if not len(dets):
        return dets, 0
    deg = _degenerate_mask(dets)
    return dets[~deg], int(deg.sum())


def _validate_output(dets: np.ndarray, height: int, width: int, conf_floor: float, where: str) -> np.ndarray:
    dets = np.asarray(dets, np.float32).reshape(-1, 6)
    if len(dets) == 0:
        return dets
    if not np.isfinite(dets).all():
        raise InvalidOutput(f"{where}: non-finite values in the backend output")
    if (dets[:, 4] < conf_floor - 1e-6).any():
        raise InvalidOutput(f"{where}: score {float(dets[:, 4].min()):.4f} below the floor {conf_floor}")
    if (dets[:, 5] != PERSON).any():
        raise InvalidOutput(f"{where}: class {sorted(set(dets[:, 5].tolist()))} is not person")
    degenerate = (dets[:, 2] <= dets[:, 0]) | (dets[:, 3] <= dets[:, 1])
    if degenerate.any():
        raise InvalidOutput(f"{where}: {int(degenerate.sum())} degenerate boxes (no width or no height)")
    # boxes may extend past the border (D-FINE does) and are not clipped; only the center is checked
    tx, ty = BORDER_TOLERANCE * width, BORDER_TOLERANCE * height
    cx, cy = (dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2
    outside = (cx < -tx) | (cy < -ty) | (cx > width + tx) | (cy > height + ty)
    if outside.any():
        raise InvalidOutput(f"{where}: center of {int(outside.sum())} boxes outside the original image "
                            f"{width}x{height}; coordinates in another system?")
    return dets


def person_label(descr: dict) -> int:
    """The checkpoint's native person label (COCO80: 0; torchvision COCO91: 1), declared by the backend."""
    label_space = descr.get("label_space") or {}
    if "person_label" not in label_space:
        raise InvalidOutput(f"backend {descr.get('backend')!r} does not declare its label space (label_space); without "
                            "it the raw model output can be neither stored nor checked")
    return int(label_space["person_label"])


def _validate_raw(raw_dets, dets: np.ndarray, conf_floor: float, person: int, where: str) -> np.ndarray:
    if raw_dets is None:
        raise InvalidOutput(f"{where}: the adapter did not return the raw model output (info['raw'])")
    raw_dets = np.asarray(raw_dets, np.float32)
    raw_dets = raw_dets.reshape(-1, 6) if raw_dets.size else np.empty((0, 6), np.float32)
    if not np.isfinite(raw_dets).all():
        raise InvalidOutput(f"{where}: non-finite values in the raw model output")
    people = raw_dets[(raw_dets[:, 5] == person) & (raw_dets[:, 4] >= conf_floor) & ~_degenerate_mask(raw_dets)]
    if len(people) != len(dets) or not np.array_equal(people[:, :5], dets[:, :5]):
        raise InvalidOutput(f"{where}: the {len(dets)} person boxes are not the {len(people)} of the raw model output "
                            f"with label {person} and score >= {conf_floor}")
    return raw_dets


def _process_part(seq: dict, frames: range, backend, conf_floor: float,
                  person: int) -> tuple[pa.Table, pa.Table, pa.Table, dict]:
    det_rows, idx_rows = {c: [] for c in DETECTION_COLUMNS}, {c: [] for c in INDEX_COLUMNS}
    raw_rows = {c: [] for c in RAW_COLUMNS}
    # a count the backend does not report in some frame (pre-NMS boxes, candidate cut) becomes None, never zero
    extra = {"saturated_frames": 0, "max_detections": 0, "n_suppressed": 0, "n_beyond_image": 0,
             "max_overshoot_px": 0.0, "max_candidate_cut": 0.0, "n_raw": 0, "max_raw": 0, "n_degenerate_dropped": 0}
    for i in frames:
        r0 = time.perf_counter()
        img = _read_frame(seq, i)
        decode = (time.perf_counter() - r0) * 1000.0
        height, width = img.shape[:2]
        t0 = time.perf_counter()
        dets, info = backend.detect(img)
        dt = (time.perf_counter() - t0) * 1000.0
        where = f"{seq['sequence_id']} frame {i}"
        dets, n_deg = drop_degenerate(dets)
        extra["n_degenerate_dropped"] += n_deg
        dets = _validate_output(dets, height, width, conf_floor, where)
        raw_dets = _validate_raw(info.get("raw"), dets, conf_floor, person, where)
        n = len(dets)
        for j, d in enumerate(dets):
            for c, v in zip(DETECTION_COLUMNS, (i, j, d[0], d[1], d[2], d[3], d[4], PERSON)):
                det_rows[c].append(v)
        for j, d in enumerate(raw_dets):
            for c, v in zip(RAW_COLUMNS, (i, j, d[0], d[1], d[2], d[3], d[4], int(d[5]))):
                raw_rows[c].append(v)
        extra["n_raw"] += len(raw_dets)
        extra["max_raw"] = max(extra["max_raw"], len(raw_dets))
        hw = info.get("input_hw") or (-1, -1)
        before = info.get("n_before_nms")
        for c, v in zip(INDEX_COLUMNS, (i, _frame_name(seq, i), "ok", height, width, n,
                                        None if before is None else int(before), int(hw[0]), int(hw[1]), dt,
                                        len(raw_dets), decode)):
            idx_rows[c].append(v)
        extra["saturated_frames"] += int(bool(info.get("saturated", False)))
        extra["max_detections"] = max(extra["max_detections"], n)
        if before is None or extra["n_suppressed"] is None:
            extra["n_suppressed"] = None
        else:
            extra["n_suppressed"] += int(before) - int(info.get("n_after_nms", n))
        if n:
            overshoot = np.max(np.column_stack([-dets[:, 0], -dets[:, 1], dets[:, 2] - width, dets[:, 3] - height,
                                                np.zeros(n)]), axis=1)
            extra["n_beyond_image"] += int((overshoot > 0).sum())
            extra["max_overshoot_px"] = max(extra["max_overshoot_px"], float(overshoot.max()))
        cut = info.get("candidate_cut_score")
        extra["max_candidate_cut"] = (None if cut is None or extra["max_candidate_cut"] is None
                                      else max(extra["max_candidate_cut"], float(cut)))
    return (pa.table(det_rows, schema=_DET_SCHEMA), pa.table(idx_rows, schema=_IDX_SCHEMA),
            pa.table(raw_rows, schema=_RAW_SCHEMA), extra)


def _write_table_atomic(tbl: pa.Table, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    pq.write_table(tbl, tmp)
    tmp.replace(dest)


PART_FILES = ("det", "idx", "raw", "res")


def _samples_table(sample_rows: list[dict], part: int) -> pa.Table:
    from .resources import SAMPLE_FIELDS
    cols = {"part": pa.array([part] * len(sample_rows), pa.int32())}
    for c in SAMPLE_FIELDS:
        cols[c] = pa.array([a.get(c) for a in sample_rows], pa.float64())
    return pa.table(cols)


def _part_is_valid(part_dir: Path, k: int, cell_id: str, frames: range) -> bool:
    meta = part_dir / f"part_{k:05d}.json"
    if not meta.is_file():
        return False
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if m.get("cell_id") != cell_id or m.get("frames") != [frames.start, frames.stop - 1]:
        return False
    for nm in PART_FILES:
        p = part_dir / f"part_{k:05d}.{nm}.parquet"
        if nm not in m.get("files", {}):
            return False
        if not p.is_file() or sha256_file(p) != m["files"][nm]:
            return False
    return True


def _check_cell(cell: Path, ident: dict) -> dict:
    m = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
    if m.get("identity") != {k: v for k, v in ident.items() if k != "cell_id"} or m.get("cell_id") != ident["cell_id"]:
        raise InvalidCache(f"{cell}: the existing manifest has another identity; nothing was overwritten")
    _check_files(cell, m)
    return m


def _check_files(cell: Path, m: dict) -> None:
    for nm, meta in m["files"].items():
        p = cell / nm
        if not p.is_file() or sha256_file(p) != meta["sha256"]:
            raise InvalidCache(f"{cell.name}: {nm} missing or tampered with (hash does not match the manifest)")


def detect_sequence(sequence: dict, detector: dict, run: dict) -> dict:
    """`run` holds run_dir, seed, backend (with `detect` and `describe`), part_size and environment."""
    backend, seed_value = run["backend"], int(run["seed"])
    descr = backend.describe()
    ident = detection_cell_identity(detector, sequence, seed_value, descr.get("versions", {}), run.get("environment"))
    cell = cell_dir(run["run_dir"], detector, sequence, seed_value)
    if (cell / "manifest.json").is_file():
        m = _check_cell(cell, ident)
        _event(run["run_dir"], "detect", ident["cell_id"], "skipped")
        return m

    _event(run["run_dir"], "detect", ident["cell_id"], "running")
    try:
        m = _build_cell(sequence, detector, run, backend, descr, ident, cell, seed_value)
    except Exception as e:
        _event(run["run_dir"], "detect", ident["cell_id"], "failed", error=f"{type(e).__name__}: {e}")
        raise
    _event(run["run_dir"], "detect", ident["cell_id"], "complete")
    return m


def _build_cell(sequence, detector, run, backend, descr, ident, cell: Path, seed_value: int) -> dict:
    from . import resources as RS
    _seed_all(seed_value)
    n, part_len = int(sequence["n_frames"]), int(run.get("part_size", 500))
    conf_floor = float(detector["detection"]["conf_floor"])
    person = person_label(descr)
    res_cfg = run.get("resources") or {}
    part_dir = cell / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = [range(a, min(a + part_len, n + 1)) for a in range(1, n + 1, part_len)]
    extras, part_resources = [], []
    t0 = time.perf_counter()
    for k, frames in enumerate(parts):
        if _part_is_valid(part_dir, k, ident["cell_id"], frames):
            meta = json.loads((part_dir / f"part_{k:05d}.json").read_text(encoding="utf-8"))
            extras.append(meta["extra"])
            part_resources.append(meta.get("resources"))
            continue
        with RS.ResourceMonitor(res_cfg.get("device", "cpu"), sample_hz=float(res_cfg.get("sample_hz", 4.0))) as mon:
            det, idx, raw, extra = _process_part(sequence, frames, backend, conf_floor, person)
        part_summary = mon.summary()
        _write_table_atomic(det, part_dir / f"part_{k:05d}.det.parquet")
        _write_table_atomic(idx, part_dir / f"part_{k:05d}.idx.parquet")
        _write_table_atomic(raw, part_dir / f"part_{k:05d}.raw.parquet")
        _write_table_atomic(_samples_table(mon.samples(), k), part_dir / f"part_{k:05d}.res.parquet")
        write_json_atomic(part_dir / f"part_{k:05d}.json", {
            "cell_id": ident["cell_id"], "frames": [frames.start, frames.stop - 1], "extra": extra,
            "resources": part_summary,
            "files": {nm: sha256_file(part_dir / f"part_{k:05d}.{nm}.parquet") for nm in PART_FILES}})
        extras.append(extra)
        part_resources.append(part_summary)
        on_part = run.get("on_part")
        if on_part is not None and len(parts) > 1:
            on_part(k, len(parts), frames, part_summary.get("duration_s"))

    # one row group per part, so the files can be read in bounded chunks
    idx_total = pa.concat_tables([pq.read_table(part_dir / f"part_{k:05d}.idx.parquet") for k in range(len(parts))])
    ids = idx_total.column("frame_id").to_pylist()
    if ids != list(range(1, n + 1)):
        raise InvalidCache(f"{cell.name}: the index does not cover exactly frames 1..{n}")
    det_dest, idx_dest = cell / "detections.parquet", cell / "frames_index.parquet"
    for dest, suffix, arrow_schema in ((det_dest, "det", _DET_SCHEMA),
                                       (cell / "raw_detections.parquet", "raw", _RAW_SCHEMA)):
        tmp = dest.with_name(f".{dest.name}.tmp")
        with pq.ParquetWriter(tmp, arrow_schema) as w:
            for k in range(len(parts)):
                w.write_table(pq.read_table(part_dir / f"part_{k:05d}.{suffix}.parquet"))
        tmp.replace(dest)
    _write_table_atomic(idx_total, idx_dest)
    _write_table_atomic(pa.concat_tables([pq.read_table(part_dir / f"part_{k:05d}.res.parquet")
                                          for k in range(len(parts))]),
                        cell / "resources.parquet")
    cell_resources = RS.combine(part_resources)
    cell_resources.update(RS.per_frame(cell_resources, n))
    nraw = idx_total.column("n_raw_detections").to_pylist()
    shapes = sorted({(h, w) for h, w in zip(idx_total.column("input_height").to_pylist(),
                                            idx_total.column("input_width").to_pylist())})
    ndet = idx_total.column("n_detections").to_pylist()
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "created_utc": _now(),
        "cell_id": ident["cell_id"], "identity": {k: v for k, v in ident.items() if k != "cell_id"},
        "cell_dir": str(cell),
        "detector_variant_id": detector.get("detector_variant_id"), "detector_name": detector["detector"]["name"],
        "resolution_requested": detector["resolution"]["requested"],
        "resolution_semantics": detector["resolution"]["semantics"], "detection_params": dict(detector["detection"]),
        "sequence_key": f"{sequence['dataset']}/{sequence['sequence_id']}",
        "backend": descr,
        "effective_input": {"distinct_input_hw": [list(f) for f in shapes],
                            "expected_input_hw": list(expected_input_hw(
                                detector["detector"]["backend"], detector["resolution"]["requested"],
                                int(sequence["height"]), int(sequence["width"])))
                            if detector["detector"]["backend"] in ("ultralytics", "dfine", "torchvision") else None},
        "counts": {"n_frames": n, "n_detections": int(sum(ndet)), "n_empty_frames": int(sum(1 for x in ndet if x == 0)),
                   # parts written before drop_degenerate have no count, and no degenerate box
                   "n_degenerate_dropped": int(sum(e.get("n_degenerate_dropped", 0) for e in extras)),
                   "max_detections_per_frame": max(e["max_detections"] for e in extras) if extras else 0,
                   "saturated_frames": int(sum(e["saturated_frames"] for e in extras)),
                   "n_suppressed_by_nms": (None if any(e["n_suppressed"] is None for e in extras)
                                           else int(sum(e["n_suppressed"] for e in extras))),
                   "n_beyond_image": (None if any(e.get("n_beyond_image") is None for e in extras)
                                      else int(sum(e["n_beyond_image"] for e in extras))),
                   "max_overshoot_px": (None if not extras or any(e.get("max_overshoot_px") is None for e in extras)
                                        else float(max(e["max_overshoot_px"] for e in extras))),
                   "max_candidate_cut_score": (None if not extras
                                               or any(e.get("max_candidate_cut") is None for e in extras)
                                               else float(max(e["max_candidate_cut"] for e in extras))),
                   "n_raw_detections": int(sum(nraw)), "max_raw_detections_per_frame": int(max(nraw)) if nraw else 0},
        "raw_output": {"file": "raw_detections.parquet", "label_space": descr.get("label_space"),
                       "note": "every box the model returned with the catalog parameters, in all classes, with the "
                               "native checkpoint label; detections.parquet is the person subset with score >= floor"},
        "timing": {"clocks": {"elapsed_ms_cache_pass": "detector: preprocessing, inference and postprocessing, until "
                                                       "the output is on the CPU; no frame reading or writing",
                              "elapsed_ms_decode": "frame reading and decoding"},
                   "detector_ms_mean": float(np.mean(idx_total.column("elapsed_ms_cache_pass").to_numpy())),
                   "decode_ms_mean": float(np.mean(idx_total.column("elapsed_ms_decode").to_numpy()))},
        "resources": cell_resources,
        "research_cost": {"seconds_this_call": round(time.perf_counter() - t0, 3),
                          "note": "duration of the cache generation; not the detector clock"},
        "files": {nm: {"sha256": sha256_file(cell / nm), "size": (cell / nm).stat().st_size}
                  for nm in ("detections.parquet", "frames_index.parquet", "raw_detections.parquet",
                             "resources.parquet")},
    }
    write_json_atomic(cell / "manifest.json", manifest)
    for p in sorted(part_dir.glob("part_*")):
        p.unlink()
    part_dir.rmdir()
    return manifest


def load_detections(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = Path(manifest["cell_dir"])
    _check_files(cell, manifest)
    return (pq.read_table(cell / "frames_index.parquet").to_pandas(),
            pq.read_table(cell / "detections.parquet").to_pandas())


def validate_effective_inputs(manifests: list[dict]) -> None:
    """Two resolution levels of one model must not produce the same input tensor on a sequence."""
    seen: dict[tuple, str] = {}
    for m in manifests:
        key = (m["detector_name"], m["sequence_key"],
               tuple(tuple(x) for x in m["effective_input"]["distinct_input_hw"]))
        if key in seen and seen[key] != m["detector_variant_id"]:
            raise InvalidOutput(f"{m['detector_name']}: {seen[key]} and {m['detector_variant_id']} produce the same "
                                f"effective input on {m['sequence_key']}: {list(key[2])}")
        seen[key] = m["detector_variant_id"]


class _WithVersions:
    """Backend view with the environment's versions, so a cell identity is known before the weights are loaded."""

    def __init__(self, view, versions: dict):
        self.view, self.versions = view, dict(versions)

    def describe(self) -> dict:
        return {**self.view.describe(), "versions": dict(self.versions)}

    def detect(self, frame):
        return self.view.detect(frame)


def _cell_is_complete(cell: Path, ident: dict) -> bool:
    """Raises `InvalidCache` if a manifest exists but does not match."""
    if not (cell / "manifest.json").is_file():
        return False
    _check_cell(cell, ident)
    return True


def run_detection(variants: list[dict], sequences: list[dict], seeds: list[int], run_dir, device: str, *,
                  paths: dict, loader, binder, versions_of, part_size: int = 500, seeds_by_seq: dict | None = None,
                  sample_hz: float = 4.0) -> dict:
    """All variant x sequence x seed cells, resuming; each detector is loaded once, and only if a cell is pending."""
    from .replicates import seeds_for
    run_dir = Path(run_dir)
    expected, completed, failures, manifests = [], [], [], []
    by_detector: dict[str, list[dict]] = {}
    for v in variants:
        by_detector.setdefault(v["detector"]["name"], []).append(v)
    # weights are checked even for detectors whose cells are all complete
    for vs in by_detector.values():
        verify_checkpoint(vs[0]["detector"], vs[0].get("checkpoint"), paths)
    environment = RT.execution_environment(device)
    RT.require_run_environment(run_dir, "detection", environment)
    for nm, vs in by_detector.items():
        versions = versions_of(vs[0]["detector"]["backend"])
        pending = []
        for v in vs:
            for s in sequences:
                for seed_value in seeds_for(s, seeds, seeds_by_seq):
                    key = {"detector_variant_id": v["detector_variant_id"],
                           "sequence": f"{s['dataset']}/{s['sequence_id']}", "seed": int(seed_value)}
                    expected.append(key)
                    ident = detection_cell_identity(v, s, seed_value, versions, environment)
                    cell = cell_dir(run_dir, v, s, seed_value)
                    try:
                        if _cell_is_complete(cell, ident):
                            _event(run_dir, "detect", ident["cell_id"], "skipped")
                            completed.append({**key, "cell_id": ident["cell_id"]})
                            manifests.append(json.loads((cell / "manifest.json").read_text(encoding="utf-8")))
                            continue
                    except InvalidCache as e:
                        failures.append({**key, "cell_id": ident["cell_id"], "error": f"InvalidCache: {e}"})
                        continue
                    pending.append((v, s, int(seed_value), key))
        total_det = len([c for c in expected if c["detector_variant_id"] in {v["detector_variant_id"] for v in vs}])
        prog = PG.Progress(f"detect {nm}", len(pending), sum(int(s["n_frames"]) for _v, s, _seed, _c in pending))
        if not pending:
            PG.say(f"[detect {nm}] {total_det} cells already complete; nothing to do")
            continue
        prog.info(f"{len(pending)} cells to do ({total_det - len(pending)} already complete), "
                  f"{int(prog.work_total)} frames; loading the model on {device}")
        model = loader(vs[0]["detector"], paths, device, vs[0]["detection"]["precision"],
                       checkpoint=vs[0]["checkpoint"])
        try:
            for v, s, seed_value, key in pending:
                view = _WithVersions(binder(model, v), versions)
                where = f"{v['detector_variant_id']} {s['dataset']}/{s['sequence_id']}"

                def report_part(k, n_parts, frames, seconds, where=where, total=int(s["n_frames"])):
                    prog.info(f"  {where}: part {k + 1} of {n_parts} (frames {frames.start} to {frames.stop - 1} of "
                              f"{total}) in {PG.fmt_duration(seconds)}")
                try:
                    m = detect_sequence(s, v, {"run_dir": run_dir, "seed": seed_value, "backend": view,
                                               "part_size": part_size, "environment": environment,
                                               "on_part": report_part,
                                               "resources": {"device": device, "sample_hz": sample_hz}})
                except (InvalidCache, InvalidOutput, UnreadableFrame, OSError, RuntimeError, ValueError) as e:
                    failures.append({**key, "error": f"{type(e).__name__}: {e}"})
                    prog.advance(f"{where}: {type(e).__name__}: {e}", int(s["n_frames"]), failed=True)
                    continue
                completed.append({**key, "cell_id": m["cell_id"]})
                manifests.append(m)
                timing = m.get("timing") or {}
                extra = PG.resources_text(m.get("resources"))
                prog.advance(f"{where}: {m['counts']['n_frames']} frames in "
                             f"{PG.fmt_duration((m.get('resources') or {}).get('duration_s'))} "
                             f"(detector {timing.get('detector_ms_mean', float('nan')):.1f} ms/frame, read "
                             f"{timing.get('decode_ms_mean', float('nan')):.1f} ms; {m['counts']['n_detections']} "
                             f"people, {m['counts']['n_raw_detections']} raw boxes)" + (f" {extra}" if extra else ""),
                             int(s["n_frames"]))
        finally:
            del model
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
                sys.modules["torch"].cuda.empty_cache()
    input_error = None
    try:
        validate_effective_inputs(manifests)
    except InvalidOutput as e:
        input_error = str(e)
    done = {(c["detector_variant_id"], c["sequence"], c["seed"]) for c in completed + failures}
    report = {
        "schema_version": SCHEMA_VERSION, "created_utc": _now(), "stage": "detection", "device": str(device),
        "expected": len(expected), "complete": completed, "failed": failures,
        "missing": [c for c in expected if (c["detector_variant_id"], c["sequence"], c["seed"]) not in done],
        "effective_input_error": input_error,
    }
    write_json_atomic(run_dir / "reports" / "detection_coverage.json", report)
    return report

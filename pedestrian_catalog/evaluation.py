"""Whole-sequence evaluation with the unmodified TrackEval, run in its own Python environment.

The GT keeps all its classes so that TrackEval's preprocessing can forgive predictions on distractors. The cells of
one sequence are evaluated in a single subprocess, each as an independent tracker.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from . import progress as PG
from .corpus import DATASET_PROTOCOLS
from . import resources as RS
from .detection import InvalidCache, _now, _event
from .provenance import sha256_file, write_json_atomic
from .schema import SCHEMA_VERSION, stable_id

METRICS = ("HOTA", "CLEAR", "Identity")
# Only PersonPath22 uses its own TrackEval class, which honors the ignore regions of unannotated frames.
_BENCHMARK = {p["annotation_protocol"]: p["evaluator_benchmark"] for p in DATASET_PROTOCOLS.values()}
_DATASET_CLASS = {p["annotation_protocol"]: p["eval_dataset"] for p in DATASET_PROTOCOLS.values()}
# Per-threshold metrics use the mean over thresholds (___AUC); TrackEval gives ratios in [0, 1], scaled here by 100.
_COLUMNS = {"HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC", "LocA": "LocA___AUC",
            "DetRe": "DetRe___AUC", "DetPr": "DetPr___AUC", "AssRe": "AssRe___AUC", "AssPr": "AssPr___AUC",
            "IDF1": "IDF1", "IDR": "IDR", "IDP": "IDP", "MOTA": "MOTA", "MOTP": "MOTP",
            "IDSW": "IDSW", "Frag": "Frag", "MT": "MT", "ML": "ML", "CLR_TP": "CLR_TP", "CLR_FN": "CLR_FN",
            "CLR_FP": "CLR_FP"}
_COUNTS = frozenset({"IDSW", "Frag", "MT", "ML", "CLR_TP", "CLR_FN", "CLR_FP"})
NORMALIZED = tuple(_COLUMNS)
_RUNNER = Path(__file__).with_name("_trackeval_runner.py")
_SPLIT = "train"


class GTMismatch(RuntimeError):
    """The GT on disk is not the one the inventory recorded."""


class EvaluationError(RuntimeError):
    """TrackEval failed or did not produce what it should."""


def benchmark_of(seq_info: dict) -> str:
    try:
        return _BENCHMARK[seq_info["annotation_protocol"]]
    except KeyError:
        raise ValueError(f"annotation protocol without a declared BENCHMARK: "
                         f"{seq_info.get('annotation_protocol')!r}") from None


def dataset_class_of(seq_info: dict) -> str:
    """Not part of the cell identity: the benchmark already distinguishes it."""
    try:
        return _DATASET_CLASS[seq_info["annotation_protocol"]]
    except KeyError:
        raise ValueError(f"annotation protocol without a declared evaluation class: "
                         f"{seq_info.get('annotation_protocol')!r}") from None


@lru_cache(maxsize=8)
def _evaluator_info(python: str, repo: str) -> tuple:
    r = subprocess.run([python, "-c", "import sys, numpy, scipy; "
                                      "print(sys.version.split()[0], numpy.__version__, scipy.__version__)"],
                       capture_output=True, text=True, timeout=120, cwd=tempfile.gettempdir())
    if r.returncode != 0:
        raise EvaluationError(f"the TrackEval Python does not run: {r.stderr[-300:]}")
    py, np_v, sp_v = r.stdout.split()
    g = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    return (("python", py), ("numpy", np_v), ("scipy", sp_v),
            ("trackeval_commit", g.stdout.strip() if g.returncode == 0 else None))


def evaluation_cell_identity(seq_info: dict, tracks: dict, benchmark: str, do_preproc: bool,
                             evaluator_info: dict) -> dict:
    ident = {"stage": "evaluation", "tracks_cell_id": tracks["cell_id"],
             "tracks_sha256": tracks["files"]["tracks_mot.txt"]["sha256"], "gt_sha256": seq_info["gt_sha256"],
             "gt_rows_after_last_frame": int(seq_info["gt_rows_after_last_frame"]),
             "sequence_key":f"{seq_info['dataset']}/{seq_info['sequence_id']}", "benchmark": benchmark,
             "do_preproc": bool(do_preproc), "metrics": list(METRICS), "evaluator": dict(evaluator_info)}
    return {"cell_id": stable_id(ident), **ident}


def cell_dir(run_dir, tracks: dict, ident: dict) -> Path:
    return Path(run_dir) / "metrics" / tracks["cell_id"] / ident["cell_id"]


def _check_tracks(tracks: dict) -> None:
    p = Path(tracks["cell_dir"]) / "tracks_mot.txt"
    if not p.is_file() or sha256_file(p) != tracks["files"]["tracks_mot.txt"]["sha256"]:
        raise InvalidCache(f"{tracks['cell_id']}: tracks_mot.txt missing or tampered with")


def _check_cell(cell: Path, ident: dict) -> dict:
    m = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
    if m.get("cell_id") != ident["cell_id"]:
        raise InvalidCache(f"{cell}: the existing evaluation has a different identity")
    for entry, meta in m["files"].items():
        if not (cell / entry).is_file() or sha256_file(cell / entry) != meta["sha256"]:
            raise InvalidCache(f"{cell.name}: {entry} missing or tampered with")
    return m


def _number(v: str):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f


def _csv_row(path: Path, seq: str) -> dict:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    id_key = next(iter(rows[0])) if rows else "seq"
    for row in rows:
        if row[id_key] == seq:
            return {k: _number(v) for k, v in row.items() if k != id_key}
    raise EvaluationError(f"{path.name} has no row for sequence {seq}")


def _normalize(native: dict) -> dict:
    out = {}
    for entry, column in _COLUMNS.items():
        if column not in native:
            raise EvaluationError(f"column {column} did not come from TrackEval; nothing is filled in by default")
        out[entry] = int(native[column]) if entry in _COUNTS else 100.0 * float(native[column])
    return out


def _run_trackeval(argv: list[str], cwd: str, sample_hz: float):
    """Resources are measured on the child itself: with parallel batches the parent would sum all of them."""
    from types import SimpleNamespace
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    proc = None
    try:
        import psutil
        proc = psutil.Process(child.pid)
    except Exception:
        pass
    # at least 10 Hz: on Windows a venv python.exe is a launcher and TrackEval runs in a short-lived grandchild
    with RS.ResourceMonitor("cpu", sample_hz=max(float(sample_hz), 10.0), include_children=True,
                            from_process_start=True, proc=proc) as mon:
        output, err = child.communicate()
    return SimpleNamespace(returncode=child.returncode, stdout=output, stderr=err), mon.summary()


def _gt_after_last_frame(gt: Path, n_frames: int) -> list[str]:
    with gt.open(encoding="utf-8", newline="") as fh:
        return [l for l in fh if l.strip() and int(float(l.split(",")[0])) > n_frames]


def _prepare_gt(src: Path, dest: Path, n_frames: int, after: int) -> None:
    """Copies the GT, dropping rows after the last video frame, which TrackEval rejects."""
    if not after:
        shutil.copyfile(src, dest)
        return
    with src.open(encoding="utf-8", newline="") as fh:
        rows = [l for l in fh if not l.strip() or int(float(l.split(",")[0])) <= n_frames]
    dest.write_text("".join(rows), encoding="utf-8", newline="")


def evaluate_batch(sequence: dict, tracks_list: list[dict], run: dict) -> list[dict]:
    """Cells of one sequence; `run` holds run_dir, python, repo and do_preproc. Manifests keep the input order."""
    bench, do_preproc = benchmark_of(sequence), bool(run.get("do_preproc", True))
    seq_dir, seq = Path(sequence["dir"]), sequence["sequence_id"]
    gt = seq_dir / "gt" / "gt.txt"
    if sha256_file(gt) != sequence["gt_sha256"]:
        raise GTMismatch(f"{sequence['dataset']}/{seq}: gt.txt changed after the inventory; "
                         f"run inventory and plan again")
    after = len(_gt_after_last_frame(gt, int(sequence["n_frames"])))
    if after != int(sequence["gt_rows_after_last_frame"]):
        raise GTMismatch(f"{sequence['dataset']}/{seq}: {after} GT rows after the last frame, but the inventory "
                         f"recorded {sequence['gt_rows_after_last_frame']}; run inventory and plan again")
    evaluator_info = dict(_evaluator_info(str(run["python"]), str(run["repo"])))
    output: list = [None] * len(tracks_list)
    pending = []
    for k, t in enumerate(tracks_list):
        _check_tracks(t)
        ident = evaluation_cell_identity(sequence, t, bench, do_preproc, evaluator_info)
        cell = cell_dir(run["run_dir"], t, ident)
        if (cell / "manifest.json").is_file():
            output[k] = _check_cell(cell, ident)
            _event(run["run_dir"], "evaluate", ident["cell_id"], "skipped")
        else:
            pending.append((k, t, ident, cell))
    if not pending:
        return output
    for _k, _t, ident, _c in pending:
        _event(run["run_dir"], "evaluate", ident["cell_id"], "running")
    try:
        _evaluate_pending(sequence, pending, run, bench, do_preproc, evaluator_info, output)
    except Exception as e:
        for _k, _t, ident, cell in pending:
            if not (cell / "manifest.json").is_file():
                _event(run["run_dir"], "evaluate", ident["cell_id"], "failed", error=f"{type(e).__name__}: {e}")
        raise
    return output


def _evaluate_pending(sequence, pending, run, bench, do_preproc, evaluator_info, output) -> None:
    seq_dir, seq = Path(sequence["dir"]), sequence["sequence_id"]
    bench_split = f"{bench}-{_SPLIT}"
    with tempfile.TemporaryDirectory(prefix="pcat_te_") as tmp:
        ws = Path(tmp)
        gt_seq = ws / "gt" / bench_split / seq
        (gt_seq / "gt").mkdir(parents=True)
        _prepare_gt(seq_dir / "gt" / "gt.txt", gt_seq / "gt" / "gt.txt", int(sequence["n_frames"]),
                    int(sequence["gt_rows_after_last_frame"]))
        shutil.copyfile(seq_dir / "seqinfo.ini", gt_seq / "seqinfo.ini")
        (ws / "gt" / "seqmaps").mkdir(parents=True)
        seqmap = ws / "gt" / "seqmaps" / f"{bench_split}.txt"
        seqmap.write_text(f"name\n{seq}\n", encoding="utf-8")
        aliases = []
        for j, (_k, t, _ident, _cell) in enumerate(pending):
            alias = f"t{j:04d}"
            data_dir = ws / "trackers" / bench_split / alias / "data"
            data_dir.mkdir(parents=True)
            shutil.copyfile(Path(t["cell_dir"]) / "tracks_mot.txt", data_dir / f"{seq}.txt")
            aliases.append(alias)
        cfg = {"gt_folder": str(ws / "gt"), "trackers_folder": str(ws / "trackers"), "benchmark": bench,
               "split": _SPLIT, "dataset_class": dataset_class_of(sequence),
               "trackers": aliases, "seqmap": str(seqmap), "do_preproc": do_preproc, "repo": str(run["repo"]),
               "status_out": str(ws / "status.json")}
        (ws / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
        t0 = time.perf_counter()
        r, usage = _run_trackeval([str(run["python"]), str(_RUNNER), str(ws / "cfg.json")], tmp,
                                  float(run.get("sample_hz", 4.0)))
        seconds = time.perf_counter() - t0
        if r.returncode != 0:
            raise EvaluationError(f"TrackEval failed (rc {r.returncode}): {(r.stderr or r.stdout)[-600:]}")
        import configparser
        cp = configparser.ConfigParser()
        cp.read(seq_dir / "seqinfo.ini", encoding="utf-8")
        n_evaluated = int(cp["Sequence"]["seqLength"])
        for (k, t, ident, cell), alias in zip(pending, aliases):
            src = ws / "trackers" / bench_split / alias
            native = _csv_row(src / "pedestrian_detailed.csv", seq)
            cell.mkdir(parents=True, exist_ok=True)
            raw_files = sorted(p.name for p in src.iterdir() if p.is_file())
            if "pedestrian_detailed.csv" not in raw_files:
                raise EvaluationError(f"TrackEval did not write pedestrian_detailed.csv for {alias}")
            for entry in raw_files:
                tmp_dest = cell / f".{entry}.tmp"
                shutil.copyfile(src / entry, tmp_dest)
                tmp_dest.replace(cell / entry)
            metrics = {
                "schema_version": SCHEMA_VERSION, "sequence_key": ident["sequence_key"], "tracks_cell_id": t["cell_id"],
                "config_id": t.get("config_id"), "replicate_seed": t.get("replicate_seed"), "benchmark": bench,
                "do_preproc": do_preproc, "evaluated_frames": n_evaluated, "native": native,
                "native_scale": "ratios as a fraction from 0 to 1 and integer counts, as in the detailed TrackEval CSV",
                "normalized": _normalize(native),
                "units": {n: ("count" if n in _COUNTS else "x100") for n in _COLUMNS}, "evaluator": evaluator_info,
                "resources": {**usage, "batch_cells": len(pending),
                              "note": "resources of the TrackEval subprocess (and its children) for the whole batch, "
                                      "not split across cells"},
                "research_cost": {"batch_seconds": round(seconds, 3), "batch_cells": len(pending),
                                  "seconds_per_cell_frame": (seconds / (len(pending) * n_evaluated) if n_evaluated
                                                             else None),
                                  "note": "duration of the TrackEval subprocess split across the cells of the batch; "
                                          "research cost"},
                "note": "integral HOTA of the complete sequence, computed by TrackEval; not a mean of blocks or of "
                        "components",
            }
            write_json_atomic(cell / "metrics.json", metrics)
            manifest = {"schema_version": SCHEMA_VERSION, "status": "complete", "created_utc": _now(),
                        "cell_id": ident["cell_id"], "identity": {x: v for x, v in ident.items() if x != "cell_id"},
                        "cell_dir": str(cell), "sequence_key": ident["sequence_key"], "tracks_cell_id": t["cell_id"],
                        "config_id": t.get("config_id"), "replicate_seed": t.get("replicate_seed"),
                        "files": {n: {"sha256": sha256_file(cell / n), "size": (cell / n).stat().st_size}
                                  for n in (*raw_files, "metrics.json")}}
            write_json_atomic(cell / "manifest.json", manifest)
            _event(run["run_dir"], "evaluate", ident["cell_id"], "complete")
            output[k] = manifest


def evaluate_sequence(sequence: dict, tracks: dict, evaluator: dict) -> dict:
    return evaluate_batch(sequence, [tracks], evaluator)[0]


def _tracks_manifest_path(run_dir: Path, config: dict, seq_info: dict, seed_value: int) -> Path:
    return (run_dir / "tracks" / config["config_hash"] / f"{seq_info['dataset']}__{seq_info['sequence_id']}"
            / f"seed{int(seed_value)}" / "manifest.json")


def _check_corpus_tracks(run_dir: Path, config: dict, sequence: dict, seed: int, tracks: dict) -> None:
    """Rejects intact tracks made with another corpus, fps or configuration."""
    from .detection import cell_dir as detection_dir

    def check(m: dict, expected: dict, stage: str) -> None:
        identity = m.get("identity") or {}
        if m.get("status") != "complete" or m.get("cell_id") != stable_id(identity):
            raise InvalidCache(f"{stage}: incomplete manifest or inconsistent identity")
        differs = [k for k, v in expected.items() if identity.get(k) != v]
        if differs:
            raise InvalidCache(f"{stage}: from another protocol ({', '.join(differs)})")

    dp = detection_dir(run_dir, config, sequence, seed) / "manifest.json"
    if not dp.is_file():
        raise InvalidCache("detection: no manifest for the current corpus")
    dm = json.loads(dp.read_text(encoding="utf-8"))
    check(dm, {"stage": "detection", "cache_key": config["cache_key"],
               "detector_variant_id": config["detector_variant_id"], "replicate_seed": int(seed),
               "sequence": {"dataset": sequence["dataset"], "sequence_id": sequence["sequence_id"],
                            "frames_manifest_sha256": sequence["frames_manifest_sha256"]}}, "detection")
    check(tracks, {"stage": "tracking", "config_hash": config["config_hash"], "config_id": config["config_id"],
                   "detection_cell_id": dm["cell_id"], "fps": float(sequence["fps"])}, "tracking")
    _check_tracks(tracks)


def run_evaluation(configs: list[dict], sequences: list[dict], seeds: list[int], run_dir, *, evaluator: dict,
                   workers: int = 1, seeds_by_seq: dict | None = None, sample_hz: float = 4.0) -> dict:
    """Evaluates every cell that already has tracks, resuming, one TrackEval batch per sequence.

    `reports/quality_long.csv` covers only the selection of this call."""
    from concurrent.futures import ThreadPoolExecutor

    from .provenance import write_text_atomic
    from .replicates import seeds_for
    run_dir = Path(run_dir)
    expected_cells, no_tracks, failures, completed, batches = [], [], [], [], []
    for s in sequences:
        batch = []
        for c in configs:
            for seed_value in seeds_for(s, seeds, seeds_by_seq):
                id_key = {"config_id": c["config_id"], "sequence": f"{s['dataset']}/{s['sequence_id']}",
                          "seed": int(seed_value)}
                expected_cells.append(id_key)
                p = _tracks_manifest_path(run_dir, c, s, seed_value)
                if not p.is_file():
                    no_tracks.append(id_key)
                    continue
                try:
                    tm = json.loads(p.read_text(encoding="utf-8"))
                    _check_corpus_tracks(run_dir, c, s, seed_value, tm)
                except (InvalidCache, OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                    failures.append({**id_key, "error": f"{type(e).__name__}: {e}"})
                    continue
                batch.append((id_key, tm))
        if batch:
            batches.append((s, batch))

    prog = PG.Progress("evaluate", len(batches), sum(len(l) for _, l in batches))
    prog.info(f"{len(batches)} sequences with tracks, {int(prog.work_total)} cells (those already evaluated are "
              f"checked and skipped), {max(1, int(workers))} in parallel")

    def evaluate(task):
        s, batch = task
        where = f"{s['dataset']}/{s['sequence_id']}"
        t0 = time.perf_counter()
        try:
            ms = evaluate_batch(s, [tm for _, tm in batch], {**evaluator, "run_dir": run_dir, "sample_hz": sample_hz})
            prog.advance(f"{where}: {len(batch)} cells in {PG.fmt_duration(time.perf_counter() - t0)}", len(batch))
            return [("ok", id_key, m) for (id_key, _), m in zip(batch, ms)]
        except Exception as e:                                    # one sequence does not bring down the others
            prog.advance(f"{where}: {type(e).__name__}: {e}", len(batch), failed=True)
            return [("failed", id_key, f"{type(e).__name__}: {e}") for id_key, _ in batch]

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        results = list(ex.map(evaluate, batches))
    rows = []
    for group in results:
        for state, id_key, x in group:
            if state != "ok":
                failures.append({**id_key, "error": x})
                continue
            completed.append({**id_key, "cell_id": x["cell_id"], "tracks_cell_id": x["tracks_cell_id"]})
            ds, seq = id_key["sequence"].split("/", 1)
            rows.append({"config_id": id_key["config_id"], "dataset": ds, "sequence_id": seq, "seed": id_key["seed"],
                         **load_metrics(x)["normalized"], "evaluation_cell_id": x["cell_id"]})
    columns = ["config_id", "dataset", "sequence_id", "seed", *NORMALIZED, "evaluation_cell_id"]
    buf = [",".join(columns)]
    for l in sorted(rows, key=lambda r: (r["config_id"], r["dataset"], r["sequence_id"], r["seed"])):
        buf.append(",".join(str(l[c]) for c in columns))
    write_text_atomic(run_dir / "reports" / "quality_long.csv", "\n".join(buf) + "\n")
    done = {(x["config_id"], x["sequence"], x["seed"]) for x in completed + failures + no_tracks}
    report = {"schema_version": SCHEMA_VERSION, "created_utc": _now(), "stage": "evaluation",
              "expected": len(expected_cells), "complete": completed, "failed": failures, "missing_tracks": no_tracks,
              "missing": [x for x in expected_cells if (x["config_id"], x["sequence"], x["seed"]) not in done]}
    write_json_atomic(run_dir / "reports" / "evaluation_coverage.json", report)
    return report


def collect_metrics(run_dir, configs: list[dict], sequences: list[dict], seeds: list[int], do_preproc: bool
                    ) -> tuple[list[dict], list[str]]:
    """The normalized metrics of every cell, and the list of cells whose evaluation is missing or ambiguous."""
    run_dir = Path(run_dir)
    rows, problems = [], []
    for c in configs:
        for s in sequences:
            for seed_value in seeds:
                where = f"{c['config_id']} {s['dataset']}/{s['sequence_id']} seed {int(seed_value)}"
                p = _tracks_manifest_path(run_dir, c, s, seed_value)
                if not p.is_file():
                    problems.append(f"{where}: no tracks")
                    continue
                try:
                    tm = json.loads(p.read_text(encoding="utf-8"))
                    _check_corpus_tracks(run_dir, c, s, seed_value, tm)
                except (InvalidCache, OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                    problems.append(f"{where}: {e}")
                    continue
                cands = []
                for mp in sorted((run_dir / "metrics" / tm["cell_id"]).glob("*/manifest.json")):
                    ident = json.loads(mp.read_text(encoding="utf-8")).get("identity", {})
                    if (ident.get("do_preproc") == bool(do_preproc) and ident.get("gt_sha256") == s["gt_sha256"]
                            and ident.get("gt_rows_after_last_frame") == s["gt_rows_after_last_frame"]
                            and ident.get("benchmark") == benchmark_of(s)
                            and ident.get("sequence_key") == f"{s['dataset']}/{s['sequence_id']}"
                            and ident.get("tracks_cell_id") == tm["cell_id"]
                            and ident.get("tracks_sha256") == tm["files"]["tracks_mot.txt"]["sha256"]):
                        cands.append(json.loads(mp.read_text(encoding="utf-8")))
                if len(cands) != 1:
                    problems.append(f"{where}: " + ("no evaluation" if not cands else
                                                    f"{len(cands)} evaluations of the same protocol with different "
                                                    f"evaluators"))
                    continue
                try:
                    met = load_metrics(cands[0])
                except InvalidCache as e:
                    problems.append(f"{where}: {e}")
                    continue
                rows.append({"config_id": c["config_id"], "dataset": s["dataset"], "sequence_id": s["sequence_id"],
                             "seed": int(seed_value), **met["normalized"], "evaluation_cell_id": cands[0]["cell_id"]})
    return rows, problems


def load_metrics(manifest: dict) -> dict:
    cell = Path(manifest["cell_dir"])
    for entry, meta in manifest["files"].items():
        if not (cell / entry).is_file() or sha256_file(cell / entry) != meta["sha256"]:
            raise InvalidCache(f"{cell.name}: {entry} missing or tampered with")
    return json.loads((cell / "metrics.json").read_text(encoding="utf-8"))

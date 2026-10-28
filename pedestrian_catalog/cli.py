"""Command line interface.

    python -m pedestrian_catalog doctor         environment, versions and the separate TrackEval environment
    python -m pedestrian_catalog inventory      corpus manifest
    python -m pedestrian_catalog plan           resolved catalog, checkpoints and cell counts
    python -m pedestrian_catalog check-options  every catalog parameter is applied and makes a difference (real weights)
    python -m pedestrian_catalog detect | track | evaluate     the three stages of a run
    python -m pedestrian_catalog run            the three stages in sequence, resuming what is missing
    python -m pedestrian_catalog export         one or more runs to the tables read by the analysis
    python -m pedestrian_catalog scene-properties   the scene properties of each video (needs the datasets)
    python -m pedestrian_catalog analyze        the results computed from the exported tables

Only check-options, detect, track and run load weights, and they use the GPU only with --device cuda.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .catalog import build_catalog, detector_variants
from .corpus import inspect_corpus, read_corpus_csv, write_corpus_csv
from .provenance import hf_snapshot_config, resolve_checkpoint, resolve_weights_file, sha256_file, write_json_atomic
from .replicates import primary_seeds, seeds_by_sequence
from .schema import SCHEMA_VERSION, SpecError

PACKAGES = ("numpy", "pandas", "pyarrow", "PyYAML", "scipy", "opencv-python", "torch",
            "torchvision", "ultralytics", "boxmot", "transformers", "huggingface-hub", "safetensors", "pytest")

_TRACKEVAL_PROBE = r'''
import json, os, sys
repo = sys.argv[1]
sys.path.insert(0, repo)
out = {"python": sys.version.split()[0]}
try:
    import numpy, scipy
    out.update(numpy=numpy.__version__, scipy=scipy.__version__)
    import trackeval
    f = getattr(trackeval, "__file__", None)
    out.update(trackeval_version=getattr(trackeval, "__version__", None), has_evaluator=hasattr(trackeval, "Evaluator"),
               imported_from_repo=bool(f) and os.path.abspath(f).startswith(os.path.abspath(repo)))
    out["status"] = "ok" if out["has_evaluator"] else "failed"
except Exception as e:
    out.update(status="failed", has_evaluator=False, error=type(e).__name__ + ": " + str(e))
print(json.dumps(out))
'''


def _yaml(path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version(dist: str):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _module_version(name: str):
    """Reads __version__ from __init__.py without importing; BoxMOT's module and distribution versions differ."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith("__init__.py"):
        return None
    m = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", Path(spec.origin).read_text(encoding="utf-8", errors="replace"), re.M)
    return m.group(1) if m else None


def _gpu() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"status": "unavailable"}
    r = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return {"status": "failed"}
    cards = [dict(zip(("name", "memory_total"), (x.strip() for x in l.split(","))))
             for l in r.stdout.strip().splitlines() if l.strip()]
    return {"status": "ok", "devices": cards}


def _commit(repo: Path):
    if not (repo / ".git").exists() or not shutil.which("git"):
        return None
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    return r.stdout.strip() if r.returncode == 0 else None


def _probe_trackeval(te: dict) -> dict:
    """Runs in a temporary folder so that `import trackeval` does not pick up the clone as a namespace package."""
    repo, py = Path(te["repo"]), str(te["python"])
    with tempfile.TemporaryDirectory(prefix="pcat_doctor_") as tmp:
        probe = Path(tmp) / "probe.py"
        probe.write_text(_TRACKEVAL_PROBE, encoding="utf-8")
        try:
            r = subprocess.run([py, str(probe), str(repo)], capture_output=True, text=True, cwd=tmp, timeout=120)
        except OSError as e:
            return {"status": "failed", "has_evaluator": False, "error": f"{type(e).__name__}: the TrackEval python does not run"}
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"status": "failed", "has_evaluator": False, "error": "the probe returned no JSON"}
    if "error" in out:
        out["error"] = out["error"].replace(str(repo), "<repo>")
    out["commit"] = _commit(repo)
    out["python_executable_name"] = Path(py).name
    return out


def cmd_doctor(args) -> int:
    paths = _yaml(args.paths)
    env = {
        "schema_version": SCHEMA_VERSION, "created_utc": _now(), "pedestrian_catalog": __version__,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable_name": Path(sys.executable).name},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": {p: _version(p) for p in PACKAGES},
        "module_versions": {m: _module_version(m) for m in ("boxmot", "ultralytics")},
        "gpu": _gpu(),
        "trackeval": _probe_trackeval(paths["trackeval"]),
    }
    write_json_atomic(args.out, env)
    te = env["trackeval"]
    print(f"python {env['python']['version']} | numpy {env['packages']['numpy']} | torch {env['packages']['torch']} | "
          f"boxmot {env['packages']['boxmot']} (module {env['module_versions']['boxmot']}) | "
          f"ultralytics {env['packages']['ultralytics']}")
    print(f"gpu: {env['gpu'].get('devices', env['gpu']['status'])}")
    print(f"trackeval: {te['status']} (python {te.get('python')}, numpy {te.get('numpy')}, commit {te.get('commit')})"
          + (f" error: {te['error']}" if te.get("error") else ""))
    print(f"environment written to {args.out}")
    return 0 if te["status"] == "ok" else 1


def cmd_inventory(args) -> int:
    paths = _yaml(args.paths)
    base = Path(args.paths).parent
    for key in ("camera_motion", "physical_groups"):
        if paths.get(key) and not Path(paths[key]).is_absolute():
            paths[key] = str(base / paths[key])
    if paths.get("selection"):
        paths["selection"] = {d: (c if Path(c).is_absolute() else str(base / c)) for d, c in paths["selection"].items()}
    lines = inspect_corpus(paths)
    write_corpus_csv(lines, args.out)
    eligible = [l for l in lines if l["eligible"]]
    print(f"sequences {len(lines)}, eligible {len(eligible)}")
    if paths.get("camera_motion"):
        print(f"  static camera {sum(l['camera_motion'] == 'fixed' for l in eligible)}, "
              f"moving {sum(l['camera_motion'] == 'moving' for l in eligible)} (eligible)")
    after = [l for l in eligible if l["gt_rows_after_last_frame"]]
    for l in after:
        print(f"  {l['dataset']}/{l['sequence_id']}: {l['gt_rows_after_last_frame']} ground-truth rows after the last "
              "frame, left out of the evaluation")
    by_ds: dict[str, list[int]] = {}
    for l in lines:
        c = by_ds.setdefault(l["dataset"], [0, 0, 0])
        c[0] += 1
        c[1] += int(l["eligible"])
        c[2] += l["n_frames"] if l["eligible"] else 0
    for ds, (n, e, q) in sorted(by_ds.items()):
        print(f"  {ds:<18} {n:>3} sequences, {e:>3} eligible, {q:>7} eligible frames")
    reasons: dict[str, int] = {}
    for l in lines:
        if not l["eligible"]:
            reasons[l["exclusion_reason"]] = reasons.get(l["exclusion_reason"], 0) + 1
    for m, n in sorted(reasons.items()):
        print(f"  excluded by {m}: {n}")
    print(f"manifest written to {args.out}")
    return 0


def _experiment_spec(exp_path: Path) -> tuple[dict, dict, dict, dict]:
    base = exp_path.parent
    exp = _yaml(exp_path)
    cat_p, trk_p = base / exp["catalog"], base / exp["trackers"]
    spec, trk = _yaml(cat_p), _yaml(trk_p)
    spec["trackers"] = trk.get("trackers")
    meta = {"catalog_spec_sha256": sha256_file(cat_p), "trackers_spec_sha256": sha256_file(trk_p),
            "boxmot_distribution": trk.get("boxmot_distribution"), "boxmot_module_version": trk.get("boxmot_module_version")}
    return exp, spec, meta, _yaml(base / exp["paths"])


def _detector_limits(variants: list[dict], paths: dict) -> dict:
    """D-FINE's query count is fixed by its checkpoint, so detection.max_det, shared by all detectors, must equal it."""
    limits = {}
    for det in {v["detector"]["name"]: v["detector"] for v in variants}.values():
        if det["backend"] != "dfine":
            continue
        nq = hf_snapshot_config(det, paths).get("num_queries")
        if isinstance(nq, bool) or not isinstance(nq, int):
            raise SpecError(f"{det['name']}: the config.json of the checkpoint has no num_queries; the fixed limit of "
                            "D-FINE cannot be checked")
        limits[det["name"]] = {"num_queries": nq}
        cap = int(variants[0]["detection"]["max_det"])
        if cap != nq:
            raise SpecError(f"detection.max_det {cap}: the limit common to every detector must be D-FINE's "
                            f"({nq} fixed queries in the checkpoint of {det['name']})")
    return limits


def cmd_plan(args) -> int:
    exp_path = Path(args.experiment)
    base = exp_path.parent
    exp, spec, meta, paths = _experiment_spec(exp_path)
    configs = build_catalog(spec)
    variants = detector_variants(configs)
    by_detector: dict[str, dict] = {}
    for v in variants:
        name = v["detector"]["name"]
        if name not in by_detector:
            by_detector[name] = resolve_checkpoint(v["detector"], paths)
        v["checkpoint"] = by_detector[name]
    limits = _detector_limits(variants, paths)
    reids = sorted({c["tracker"]["profile"]["reid_weights"] for c in configs if c["tracker"]["family"] == "botsort"})
    assets = {"reid_weights": resolve_weights_file(reids[0], paths)} if len(reids) == 1 else \
        {f"reid_weights:{r}": resolve_weights_file(r, paths) for r in reids}
    corpus_p = base / exp["corpus_manifest"]
    corpus = read_corpus_csv(corpus_p)
    eligible = [l for l in corpus if l["eligible"]]
    n = len(eligible)
    by_seq = seeds_by_sequence(corpus, exp["quality"])
    pairs = sum(len(s) for s in by_seq.values())
    by_ds: dict[str, int] = {}
    for l in eligible:
        by_ds[l["dataset"]] = by_ds.get(l["dataset"], 0) + 1
    resolved = {
        "schema_version": SCHEMA_VERSION, "created_utc": _now(), **meta,
        "detector_variants": variants, "configs": configs, "tracker_assets": assets, "detector_limits": limits,
        "corpus": {"manifest_sha256": sha256_file(corpus_p), "n_sequences": len(corpus), "n_eligible": n,
                   "n_frames_eligible": sum(l["n_frames"] for l in eligible), "eligible_by_dataset": by_ds},
        "quality": {"replicate_seeds": primary_seeds(exp["quality"]), "determinism_check": exp["quality"].get("determinism_check"),
                    "determinism_sequences": sorted(k for k, s in by_seq.items() if len(s) > len(primary_seeds(exp["quality"]))),
                    "sequence_seed_pairs": pairs},
        "cells": {"detection": len(variants) * pairs, "tracking": len(configs) * pairs,
                  "evaluation": len(configs) * pairs},
    }
    dest = Path(args.out) if args.out else base / exp["resolved_catalog"]
    write_json_atomic(dest, resolved)
    print(f"{len(variants)} detector x resolution variants, {len(configs)} configurations, {n} eligible sequences "
          f"({resolved['corpus']['n_frames_eligible']} frames)")
    for v in variants:
        ck = v["checkpoint"]
        print(f"  {v['detector_variant_id']:<24} {v['resolution']['semantics']:<20} "
              f"{ck.get('file') or ck.get('hf_repo')}{'@' + ck['revision'] if ck.get('revision') else ''} "
              f"sha256 {ck['sha256'][:12]}")
    c = resolved["cells"]
    print(f"cells: detection {c['detection']}, tracking {c['tracking']}, evaluation {c['evaluation']}")
    print(f"resolved catalog written to {os.path.normpath(dest)}")
    return 0


class OutdatedCatalog(SpecError):
    """The catalog, the tracker profiles or the corpus manifest changed after the last `plan`."""


def _run_context(exp_path: Path) -> dict:
    exp, _spec, meta, paths = _experiment_spec(exp_path)
    base = exp_path.parent
    res_p = base / exp["resolved_catalog"]
    if not res_p.is_file():
        raise FileNotFoundError("resolved catalog missing: run the plan command first")
    res = json.loads(res_p.read_text(encoding="utf-8"))
    for field in ("catalog_spec_sha256", "trackers_spec_sha256"):
        if res.get(field) != meta[field]:
            raise OutdatedCatalog(f"{field} changed since the last plan: run plan again before running")
    corpus_p = base / exp["corpus_manifest"]
    if res["corpus"]["manifest_sha256"] != sha256_file(corpus_p):
        raise OutdatedCatalog("the corpus manifest changed since the last plan: run plan again before running")
    runs_root = Path(paths.get("runs_root") or "runs")
    if not runs_root.is_absolute():
        runs_root = base.parent / runs_root
    return {"exp": exp, "paths": paths, "resolved": res, "corpus": read_corpus_csv(corpus_p), "runs_root": runs_root,
            "base": base}


def _seal_path(ctx: dict) -> Path:
    rel = (ctx["exp"].get("option_check") or {}).get("seal")
    if not rel:
        raise SpecError("experiment.option_check.seal missing: the check-options seal has nowhere to go")
    return ctx["base"] / rel


def _require_seal(ctx: dict) -> dict:
    from . import options_check as OC
    from .backends import detectors as DB
    from .backends import trackers as BT
    res = ctx["resolved"]
    return OC.require_seal(_seal_path(ctx), res, OC.current_versions(res, DB.backend_versions, BT.tracker_versions))


def _select(items, requested: str | None, key, label: str):
    if not requested:
        return list(items)
    wanted = {p.strip() for p in requested.split(",") if p.strip()}
    known = {key(i) for i in items}
    if wanted - known:
        raise SpecError(f"unknown {label}: {', '.join(sorted(wanted - known))}")
    return [i for i in items if key(i) in wanted]


def _seed_design(ctx: dict, requested: str | None):
    """Seeds given on the command line apply to every sequence; otherwise the protocol's per-sequence design is used."""
    if requested:
        return [int(s) for s in requested.split(",")], None
    q = ctx["exp"]["quality"]
    return primary_seeds(q), seeds_by_sequence(ctx["corpus"], q)


def _sample_hz(ctx: dict) -> float:
    return float((ctx["exp"].get("resources") or {}).get("sample_hz", 4.0))


def cmd_detect(args) -> int:
    import contextlib

    from . import detection as D
    from .backends import detectors as DB
    from .corpus import resolve_sequence
    from .runtime import GPU_LOCK_NAME, gpu_lock
    ctx = _run_context(Path(args.experiment))
    _require_seal(ctx)
    variants = _select(ctx["resolved"]["detector_variants"], args.detectors, lambda v: v["detector"]["name"], "detector")
    lines = _select([l for l in ctx["corpus"] if l["eligible"]], args.sequences,
                    lambda l: f"{l['dataset']}/{l['sequence_id']}", "sequence")
    seeds, by_seq = _seed_design(ctx, args.seeds)
    sequences = [resolve_sequence(l, ctx["paths"]) for l in lines]
    run_dir = ctx["runs_root"] / args.run_id
    lock = (gpu_lock(ctx["runs_root"] / GPU_LOCK_NAME, f"detect {args.run_id}") if str(args.device).startswith("cuda")
            else contextlib.nullcontext())
    with lock:
        rep = D.run_detection(variants, sequences, seeds, run_dir, args.device, paths=ctx["paths"],
                              loader=DB.load_model, binder=DB.bind, versions_of=DB.backend_versions,
                              part_size=args.part_size, seeds_by_seq=by_seq, sample_hz=_sample_hz(ctx))
    print(f"detection: {len(rep['complete'])} of {rep['expected']} cells complete, {len(rep['failed'])} failed, "
          f"{len(rep['missing'])} missing")
    for f in rep["failed"][:10]:
        print(f"  failed {f['detector_variant_id']} {f['sequence']} seed {f['seed']}: {f['error']}")
    if rep["effective_input_error"]:
        print(f"  EFFECTIVE INPUT: {rep['effective_input_error']}")
    print(f"coverage in {os.path.normpath(run_dir / 'reports' / 'detection_coverage.json')}")
    return 0 if not rep["failed"] and not rep["missing"] and not rep["effective_input_error"] else 1


def cmd_track(args) -> int:
    import contextlib

    from . import tracking as TR
    from .backends import detectors as DB
    from .backends import trackers as BT
    from .corpus import resolve_sequence
    from .runtime import GPU_LOCK_NAME, gpu_lock
    ctx = _run_context(Path(args.experiment))
    _require_seal(ctx)
    configs = _select(ctx["resolved"]["configs"], args.configs, lambda c: c["config_id"], "configuration")
    lines = _select([l for l in ctx["corpus"] if l["eligible"]], args.sequences,
                    lambda l: f"{l['dataset']}/{l['sequence_id']}", "sequence")
    seeds, by_seq = _seed_design(ctx, args.seeds)
    sequences = [resolve_sequence(l, ctx["paths"]) for l in lines]
    variants = {v["detector_variant_id"]: v for v in ctx["resolved"]["detector_variants"]}
    reid = ctx["resolved"].get("tracker_assets", {}).get("reid_weights")
    reid_path = _checked_reid(ctx)
    cuda = str(args.device).startswith("cuda")
    run_dir = ctx["runs_root"] / args.run_id
    lock = gpu_lock(ctx["runs_root"] / GPU_LOCK_NAME, f"track {args.run_id}") if cuda else contextlib.nullcontext()
    with lock:
        rep = TR.run_tracking(configs, variants, sequences, seeds, run_dir, factory=BT.factory(args.device, reid_path),
                              tracker_versions=BT.tracker_versions(), detection_versions_of=DB.backend_versions,
                              reid_sha256=reid["sha256"] if reid else None, device=args.device, seeds_by_seq=by_seq,
                              sample_hz=_sample_hz(ctx))
    print(f"tracking: {len(rep['complete'])} of {rep['expected']} cells complete, {len(rep['failed'])} failed, "
          f"{len(rep['missing_detection'])} waiting for detection")
    for f in rep["failed"][:10]:
        print(f"  failed {f['config_id']} {f['sequence']} seed {f['seed']}: {f['error']}")
    print(f"coverage in {os.path.normpath(run_dir / 'reports' / 'tracking_coverage.json')}")
    return 0 if not rep["failed"] and not rep["missing_detection"] and not rep["missing"] else 1


def cmd_evaluate(args) -> int:
    from . import evaluation as EV
    from .corpus import resolve_sequence
    ctx = _run_context(Path(args.experiment))
    configs = _select(ctx["resolved"]["configs"], args.configs, lambda c: c["config_id"], "configuration")
    lines = _select([l for l in ctx["corpus"] if l["eligible"]], args.sequences,
                    lambda l: f"{l['dataset']}/{l['sequence_id']}", "sequence")
    seeds, by_seq = _seed_design(ctx, args.seeds)
    te = ctx["paths"]["trackeval"]
    rep = EV.run_evaluation(configs, [resolve_sequence(l, ctx["paths"]) for l in lines], seeds,
                            ctx["runs_root"] / args.run_id, workers=args.workers, seeds_by_seq=by_seq, sample_hz=_sample_hz(ctx),
                            evaluator={"python": te["python"], "repo": te["repo"],
                                       "do_preproc": bool(ctx["exp"]["evaluation"]["do_preproc"])})
    print(f"evaluation: {len(rep['complete'])} of {rep['expected']} cells complete, {len(rep['failed'])} failed, "
          f"{len(rep['missing_tracks'])} waiting for tracks")
    for f in rep["failed"][:10]:
        print(f"  failed {f['config_id']} {f['sequence']} seed {f['seed']}: {f['error']}")
    run_dir = ctx["runs_root"] / args.run_id
    print(f"matrix in {os.path.normpath(run_dir / 'reports' / 'quality_long.csv')}")
    return 0 if not rep["failed"] and not rep["missing_tracks"] and not rep["missing"] else 1


def cmd_export(args) -> int:
    from . import export as EX
    ctx = _run_context(Path(args.experiment))
    seeds = primary_seeds(ctx["exp"]["quality"])
    runs = [ctx["runs_root"] / r for r in args.run_id]
    r = EX.export(runs, ctx["resolved"]["configs"], ctx["corpus"], seeds,
                  bool(ctx["exp"]["evaluation"]["do_preproc"]), Path(args.out))
    print(f"exported {len(runs)} run(s): {r['videos']} videos, {r['configurations']} configurations to "
          f"{os.path.normpath(args.out)}")
    return 0


def cmd_scene_properties(args) -> int:
    from . import scene
    return scene.write_properties(Path(args.paths), Path(args.corpus), Path(args.out))


def cmd_analyze(args) -> int:
    from . import analysis
    return analysis.run(Path(args.results), Path(args.out) if args.out else None, args.fps)


def _run_step(argv: list[str], log_path: Path) -> int:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"], env["PYTHONIOENCODING"] = "1", "utf-8"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-m", "pedestrian_catalog", *argv], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace")
        for line in child.stdout:
            try:
                sys.stdout.write(line)
            except UnicodeEncodeError:
                sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return child.wait()


def _log_step(logs: Path, record: dict) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "run.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def cmd_run(args) -> int:
    """Detection runs one process per detector, so that each process holds a single model."""
    import time
    from datetime import datetime, timezone
    ctx = _run_context(Path(args.experiment))
    _require_seal(ctx)
    detectors = list(dict.fromkeys(v["detector"]["name"] for v in ctx["resolved"]["detector_variants"]))
    base = ["--run-id", args.run_id, "--experiment", args.experiment]
    steps = [["detect", *base, "--device", args.device, "--detectors", name] for name in detectors]
    steps += [["track", *base, "--device", args.device], ["evaluate", *base, "--workers", str(args.workers)]]
    logs = ctx["runs_root"] / args.run_id / "logs"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    now = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")          # noqa: E731
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    print(f"run {args.run_id}: {len(steps)} stages; logs in {os.path.normpath(logs)}", flush=True)
    t_total = time.perf_counter()
    for i, argv in enumerate(steps, 1):
        name = argv[0] + (f"_{argv[argv.index('--detectors') + 1]}" if "--detectors" in argv else "")
        log = logs / f"{stamp}_{i:02d}_{name}.log"
        print(f"== stage {i} of {len(steps)}: {' '.join(argv)}", flush=True)
        record = {"run_started": stamp, "step": i, "of": len(steps), "name": name, "argv": argv, "log": log.name}
        _log_step(logs, {**record, "state": "running", "start_utc": now()})
        t0 = time.perf_counter()
        rc = _run_step(argv, log)
        seconds = round(time.perf_counter() - t0, 1)
        _log_step(logs, {**record, "state": "complete" if rc == 0 else "failed", "end_utc": now(),
                         "seconds": seconds, "returncode": rc})
        print(f"== stage {i} of {len(steps)} ({name}) ended with code {rc} in {_duration(seconds)}; "
              f"total {_duration(time.perf_counter() - t_total)}", flush=True)
        if rc != 0:
            print(f"the run stopped at stage {i} ({name}) with code {rc}; the log is {os.path.normpath(log)}; "
                  "repeating the same command resumes what is missing", file=sys.stderr, flush=True)
            return rc
    print(f"run complete: {len(steps)} stages in {_duration(time.perf_counter() - t_total)}", flush=True)
    return 0


def _duration(seconds) -> str:
    from .progress import fmt_duration
    return fmt_duration(seconds)


def _checked_reid(ctx: dict) -> str | None:
    """Path of the ReID weights, verified against the hash recorded by plan."""
    from .provenance import verify_weights_file
    reid = ctx["resolved"].get("tracker_assets", {}).get("reid_weights")
    return str(verify_weights_file(reid, ctx["paths"])) if reid else None


def cmd_check_options(args) -> int:
    import contextlib

    import numpy as np

    from . import options_check as OC
    from .backends import detectors as DB
    from .backends import trackers as BT
    from .corpus import resolve_sequence
    from .detection import _read_frame
    from .provenance import to_native
    from .runtime import GPU_LOCK_NAME, gpu_lock
    ctx = _run_context(Path(args.experiment))
    cfg, res = ctx["exp"].get("option_check") or {}, ctx["resolved"]
    eligible = [l for l in ctx["corpus"] if l["eligible"]]
    frames = OC.probe_frames(eligible, lambda l: resolve_sequence(l, ctx["paths"]), int(cfg.get("frames_per_geometry", 2)))
    tc = cfg.get("tracker_clip") or {}
    line = next((l for l in eligible if f"{l['dataset']}/{l['sequence_id']}" == tc.get("sequence")), None)
    if line is None:
        raise SpecError(f"option_check.tracker_clip.sequence {tc.get('sequence')!r} is not an eligible sequence of the "
                        "corpus")
    if tc.get("detector_variant") not in {v["detector_variant_id"] for v in res["detector_variants"]}:
        raise SpecError(f"option_check.tracker_clip.detector_variant {tc.get('detector_variant')!r} is not in the catalog")
    seq = resolve_sequence(line, ctx["paths"])
    clip = {"variant_id": tc["detector_variant"], "seq": seq, "frames": min(int(tc.get("frames", 100)), int(seq["n_frames"]))}
    reid_path = _checked_reid(ctx)
    profiles: dict = {}
    for c in res["configs"]:
        items = profiles.setdefault(c["tracker"]["family"], [])
        if c["tracker"]["profile"] not in items:
            items.append(c["tracker"]["profile"])
    cuda = str(args.device).startswith("cuda")
    lock = gpu_lock(ctx["runs_root"] / GPU_LOCK_NAME, "check-options") if cuda else contextlib.nullcontext()
    with lock:
        cks = {v["detector"]["name"]: v["checkpoint"] for v in res["detector_variants"]}
        det = OC.check_detectors(res["detector_variants"], frames,
                                 lambda d: DB.load_model(d, ctx["paths"], args.device, checkpoint=cks[d["name"]]),
                                 DB.bind, clip)
        from boxmot.motion.cmc import available_cmc_methods
        empty = np.zeros((int(seq["height"]), int(seq["width"]), 3), np.uint8)
        trk = OC.check_trackers(
            profiles, det["clip_detections"], lambda f: (lambda i: _read_frame(seq, i)) if BT.needs_image(f) else (lambda i: empty),
            float(seq["fps"]), construct=lambda f, kw: BT.construct(f, kw, args.device, reid_path),
            profile_kwargs=BT.profile_kwargs, verify=BT.verify_profile, update=BT.update, cmc_methods=available_cmc_methods())
    problems = det["problems"] + trk["problems"]
    record = OC.seal_record(res, OC.current_versions(res, DB.backend_versions, BT.tracker_versions), problems, to_native({
        "probe_frames": [{k: f[k] for k in ("sequence", "frame", "width", "height")} for f in frames],
        "tracker_clip": {"sequence": tc["sequence"], "frames": clip["frames"], "detector_variant": clip["variant_id"]},
        "detectors": det["detectors"], "trackers": trk["trackers"]}), args.device)
    dest = _seal_path(ctx)
    OC.write_seal(dest, record)
    print(f"check-options: {len(frames)} probe frames in {len({(f['width'], f['height']) for f in frames})} formats; "
          f"tracker clip {tc['sequence']} with {clip['frames']} frames ({clip['variant_id']})")
    for name, r in det["detectors"].items():
        strong = "; ".join(f"{k} strong changes {x['frames_changed']}/{x['frames']}" for k, x in r["strong"].items()
                           if "frames" in x)
        print(f"  {name}: " + ", ".join(f"{vid} {x['input_hw']}" for vid, x in r["levels"].items()) + f"; {strong}")
    for fam, r in trk["trackers"].items():
        print(f"  {fam}: {len(r['fields'])} fields, each changes {min((x['frames_changed'] for x in r['fields'].values()), default=0)} "
              f"to {max((x['frames_changed'] for x in r['fields'].values()), default=0)} frames; confirmed without effect: "
              f"{', '.join(k for k, x in r['no_effect'].items() if x.get('frames_changed') == 0) or 'none'}")
    print(f"seal {'APPROVED' if not problems else 'REFUSED'} in {os.path.normpath(dest)}")
    for p in problems[:30]:
        print(f"  PROBLEM: {p}")
    return 0 if not problems else 1


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pedestrian_catalog", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    device_help = "cpu or cuda:0; required, so that the GPU is never used by default"
    g = sub.add_parser("check-options", help="checks with the real weights that every parameter is applied and makes a "
                                             "difference; writes the seal that the runs require")
    g.add_argument("--device", required=True, help=device_help)
    g.add_argument("--experiment", default="configs/experiment.yaml")
    g.set_defaults(func=cmd_check_options)
    e = sub.add_parser("evaluate", help="evaluates the tracks with TrackEval in its separate environment (CPU)")
    e.add_argument("--run-id", required=True)
    e.add_argument("--experiment", default="configs/experiment.yaml")
    e.add_argument("--configs", default=None, help="comma-separated config_id")
    e.add_argument("--sequences", default=None, help="comma-separated DATASET/SEQUENCE")
    e.add_argument("--seeds", default=None, help="comma-separated replicate seeds")
    e.add_argument("--workers", type=int, default=1, help="sequences evaluated at the same time")
    e.set_defaults(func=cmd_evaluate)
    r = sub.add_parser("track", help="creates or resumes the tracks on the detection caches (BoT-SORT uses the GPU with "
                                     "--device cuda)")
    r.add_argument("--run-id", required=True)
    r.add_argument("--device", required=True, help=device_help)
    r.add_argument("--experiment", default="configs/experiment.yaml")
    r.add_argument("--configs", default=None, help="comma-separated config_id")
    r.add_argument("--sequences", default=None, help="comma-separated DATASET/SEQUENCE")
    r.add_argument("--seeds", default=None, help="comma-separated replicate seeds")
    r.set_defaults(func=cmd_track)
    d = sub.add_parser("doctor", help="records the environment and checks the separate TrackEval environment")
    d.add_argument("--paths", default="configs/paths.local.yaml")
    d.add_argument("--out", default="manifests/environment.json")
    d.set_defaults(func=cmd_doctor)
    i = sub.add_parser("inventory", help="inventories the corpus and writes the manifest")
    i.add_argument("--paths", default="configs/paths.local.yaml")
    i.add_argument("--out", default="manifests/corpus.csv")
    i.set_defaults(func=cmd_inventory)
    p = sub.add_parser("plan", help="resolves the catalog and counts the cells, without loading weights")
    p.add_argument("--experiment", default="configs/experiment.yaml")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_plan)
    u = sub.add_parser("run", help="a whole run: detect (one process per detector), track and evaluate; stops at the "
                                   "first stage that fails and resumes when repeated")
    u.add_argument("--run-id", required=True)
    u.add_argument("--device", required=True, help=device_help)
    u.add_argument("--experiment", default="configs/experiment.yaml")
    u.add_argument("--workers", type=int, default=4, help="parallel TrackEval evaluations (CPU)")
    u.set_defaults(func=cmd_run)
    x = sub.add_parser("export", help="exports one or more runs to the tables read by the analysis (no GPU)")
    x.add_argument("--run-id", required=True, nargs="+", help="the runs, in the order in which they are numbered")
    x.add_argument("--experiment", default="configs/experiment.yaml")
    x.add_argument("--out", required=True, help="folder of the tables")
    x.set_defaults(func=cmd_export)
    s = sub.add_parser("scene-properties", help="measures the scene properties of each video from its annotations "
                                                "(needs the datasets; no GPU)")
    s.add_argument("--paths", default="configs/paths.local.yaml")
    s.add_argument("--corpus", default="manifests/corpus.csv")
    s.add_argument("--out", required=True, help="the scene_properties.csv to write")
    s.set_defaults(func=cmd_scene_properties)
    a = sub.add_parser("analyze", help="computes the results from the exported tables (no GPU)")
    a.add_argument("--results", default="paper_results", help="folder with the exported tables")
    a.add_argument("--out", default=None, help="folder of the result tables (default: <results>/analysis)")
    a.add_argument("--fps", type=float, nargs="+", default=[30, 25, 20, 14],
                   help="frame rates whose time between frames are the budgets of budget.csv")
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("detect", help="creates or resumes the detection caches (uses the GPU with --device cuda)")
    t.add_argument("--run-id", required=True)
    t.add_argument("--device", required=True, help=device_help)
    t.add_argument("--experiment", default="configs/experiment.yaml")
    t.add_argument("--detectors", default=None, help="comma-separated names")
    t.add_argument("--sequences", default=None, help="comma-separated DATASET/SEQUENCE")
    t.add_argument("--seeds", default=None, help="comma-separated replicate seeds")
    t.add_argument("--part-size", type=int, default=500)
    t.set_defaults(func=cmd_detect)
    return ap


def _quiet_warnings() -> None:
    """torch warns in every process that pynvml was renamed; the warning is irrelevant and hides the progress."""
    import warnings
    warnings.filterwarnings("ignore", message="The pynvml package is deprecated", category=FutureWarning)


def main(argv=None) -> int:
    from .runtime import GpuBusy
    _quiet_warnings()
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (SpecError, FileNotFoundError, KeyError, ValueError, GpuBusy) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

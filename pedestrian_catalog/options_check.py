"""Checks with the real weights that every catalog option is applied and changes the output, and that no two levels
give the same output.

The result is a seal tied to the catalog identity and the library versions; detect and track refuse to start without
an approved seal that matches them.
"""
from __future__ import annotations

import copy
import inspect
import itertools
import json
from pathlib import Path

import numpy as np

from .catalog import RESOLUTION_MULTIPLE, TRACKER_FIXED_CONFIRMATION, TRACKING_FIELDS, level_tag, no_effect_fields
from .detection import _now, _read_frame, expected_input_hw
from .provenance import write_json_atomic
from .schema import SCHEMA_VERSION, SpecError, stable_id

SEAL_VERSION = 1
STRONG_CONF_FLOOR = 0.5
STRONG_MAX_DET = 1
# Two values per field, on both sides of the default: some fields react in one direction only (appearance_thresh).
STRONG_TRACKER = {
    "track_thresh": (0.2, 0.8), "match_thresh": (0.3, 0.95), "track_buffer": (1, 120), "min_conf": (0.02, 0.4),
    "det_thresh": (0.1, 0.8), "max_age": (1, 120), "min_hits": (1, 10), "iou_threshold": (0.05, 0.9),
    "max_obs": (5, 500), "delta_t": (1, 10), "inertia": (0.0, 0.9), "Q_xy_scaling": (0.0001, 1.0),
    "Q_s_scaling": (1e-6, 1.0), "track_high_thresh": (0.2, 0.8), "track_low_thresh": (0.02, 0.4),
    "new_track_thresh": (0.3, 0.9), "proximity_thresh": (0.1, 0.9), "appearance_thresh": (0.05, 0.9),
    "lost_track_seconds": (0.05, 4.0), "high_score_thresh": (0.2, 0.8), "confirm_frames": (1, 6),
}
_NOT_PERTURBABLE = frozenset({"reid_weights"})


def _yes(x) -> bool:
    return x is True or str(x).strip().lower() in ("true", "1")


def _equal(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and not isinstance(a, bool) and abs(float(a) - float(b)) <= 1e-12
    return a == b


def catalog_identity(resolved: dict) -> str:
    return stable_id({"configs": sorted(c["config_hash"] for c in resolved["configs"]),
                      "variants": sorted(v["cache_key"] for v in resolved["detector_variants"]),
                      "detector_limits": resolved.get("detector_limits") or {},
                      "tracker_assets": resolved.get("tracker_assets") or {}})


def current_versions(resolved: dict, backend_versions, tracker_versions) -> dict:
    backends = sorted({v["detector"]["backend"] for v in resolved["detector_variants"]})
    return {"detectors": {b: backend_versions(b) for b in backends}, "trackers": tracker_versions()}


def probe_frames(corpus_lines: list[dict], resolve, per_geometry: int) -> list[dict]:
    """`per_geometry` evenly spaced frames of the first eligible sequence of each frame size."""
    by_shape: dict = {}
    for l in sorted((l for l in corpus_lines if _yes(l["eligible"])), key=lambda l: (l["dataset"], l["sequence_id"])):
        by_shape.setdefault((int(l["width"]), int(l["height"])), l)
    out = []
    for (w, h), l in sorted(by_shape.items()):
        seq = resolve(l)
        n = int(seq["n_frames"])
        for i in sorted({1 + round((n - 1) * k / per_geometry) for k in range(per_geometry)}):
            out.append({"sequence": f"{l['dataset']}/{l['sequence_id']}", "frame": int(i), "width": w, "height": h, "seq": seq})
    return out


def _sorted_dets(d) -> np.ndarray:
    d = np.asarray(d, np.float64).reshape(-1, 6)
    return d[np.lexsort(d.T[::-1])]


def _same(a, b) -> bool:
    a, b = _sorted_dets(a), _sorted_dets(b)
    return a.shape == b.shape and np.array_equal(a, b)


def _with(v: dict, **detection) -> dict:
    w = copy.deepcopy(v)
    w["detection"].update(detection)
    return w


def _model_stride(backend: str, fab: dict):
    """The stride the resolution must be a multiple of (torchvision accepts any size)."""
    if backend == "ultralytics":
        return fab.get("stride")
    if backend == "dfine":
        return max(fab.get("feat_strides") or [0]) or None
    return 1


def _applied(backend: str, settings: dict, fab: dict, v: dict, info: dict, h: int, w: int) -> list[str]:
    """Mismatches between the requested settings and those read back from the live object."""
    det, r = v["detection"], int(v["resolution"]["requested"])
    failures = []
    expected = list(expected_input_hw(backend, r, h, w))
    if list(info.get("input_hw") or []) != expected:
        failures.append(f"input {info.get('input_hw')} instead of {expected}")
    if backend == "ultralytics":
        requested = {"conf": float(det["conf_floor"]), "max_det": int(det["max_det"]), "classes": None,
                     "end2end": fab.get("end2end")}
    elif backend == "torchvision":
        requested = {"score_thresh": float(det["conf_floor"]), "detections_per_img": int(det["max_det"]), "min_size": [r],
                     "max_size": r, "nms_thresh": fab.get("nms_thresh")}
    else:
        requested = {"conf_floor": float(det["conf_floor"]), "size": [r, r], "num_queries": int(det["max_det"]),
                     "n_candidates": int(det["max_det"])}
    for k, value in requested.items():
        if not _equal(settings.get(k), value):
            failures.append(f"{k} read back as {settings.get(k)!r} instead of {value!r}")
    return failures


def check_detectors(variants: list[dict], frames: list[dict], load, bind, clip: dict | None = None) -> dict:
    """Runs every level of each detector on the probe frames; also returns the detections of the tracker clip."""
    report, problems, clip_dets = {}, [], None
    by_det: dict = {}
    for v in variants:
        by_det.setdefault(v["detector"]["name"], []).append(v)
    for name, vs in sorted(by_det.items()):
        vs = sorted(vs, key=lambda v: int(v["resolution"]["requested"]))
        backend = vs[0]["detector"]["backend"]
        model = load(vs[0]["detector"])
        fab = dict(getattr(model, "factory_settings", None) or {})
        r_det = {"backend": backend, "factory_settings": fab, "levels": {}, "strong": {}}
        stride = _model_stride(backend, fab)
        if not stride or RESOLUTION_MULTIPLE % int(stride):
            problems.append(f"{name}: the model stride {stride} does not divide the required resolution multiple "
                            f"{RESOLUTION_MULTIPLE}")
        outputs: dict = {}
        for v in vs:
            vid, b = v["detector_variant_id"], bind(model, v)
            level = {"input_hw": [], "saturated_frames": 0, "detections": 0}
            for f in frames:
                img = _read_frame(f["seq"], f["frame"])
                try:
                    dets, info = b.detect(img)
                except Exception as e:                       # noqa: BLE001
                    problems.append(f"{vid}: failed on {f['sequence']} frame {f['frame']}: {type(e).__name__}: {e}")
                    outputs[vid] = None
                    break
                problems += [f"{vid} ({f['sequence']} frame {f['frame']}): {x}"
                             for x in _applied(backend, b.settings(), fab, v, info, *img.shape[:2])]
                if list(info.get("input_hw") or []) not in level["input_hw"]:
                    level["input_hw"].append(list(info.get("input_hw") or []))
                level["saturated_frames"] += int(bool(info.get("saturated")))
                level["detections"] += int(len(dets))
                outputs.setdefault(vid, []).append(_sorted_dets(dets))
            r_det["levels"][vid] = {**level, "frames": len(frames)}
        valid = [v["detector_variant_id"] for v in vs if outputs.get(v["detector_variant_id"]) is not None]
        for i, a in enumerate(valid):
            for b_id in valid[i + 1:]:
                if all(_same(x, y) for x, y in zip(outputs[a], outputs[b_id])):
                    problems.append(f"{a} and {b_id} give the same output on the {len(frames)} probe frames: one level "
                                    "repeats the other")
        v0 = vs[0]
        if outputs.get(v0["detector_variant_id"]) is not None:
            tests = {"conf_floor": {"conf_floor": STRONG_CONF_FLOOR}}
            if backend != "dfine":
                tests["max_det"] = {"max_det": STRONG_MAX_DET}
            for field, change in tests.items():
                b = bind(model, _with(v0, **change))
                changed = sum(int(not _same(base, b.detect(_read_frame(f["seq"], f["frame"]))[0]))
                              for f, base in zip(frames, outputs[v0["detector_variant_id"]]))
                r_det["strong"][field] = {"value": change[field], "frames_changed": changed, "frames": len(frames)}
                if not changed:
                    problems.append(f"{name}: {field} does not change the output on any of the {len(frames)} frames "
                                    "under a strong perturbation")
            if backend == "dfine":
                r_det["strong"]["max_det"] = {"note": "limit fixed by the queries of the checkpoint; checked that it cuts there"}
        if clip and clip["variant_id"] in {v["detector_variant_id"] for v in vs}:
            b = bind(model, next(v for v in vs if v["detector_variant_id"] == clip["variant_id"]))
            clip_dets = [np.asarray(b.detect(_read_frame(clip["seq"], i))[0], np.float32).reshape(-1, 6)
                         for i in range(1, int(clip["frames"]) + 1)]
        del model
        report[name] = r_det
    return {"detectors": report, "clip_detections": clip_dets, "problems": problems}


def installed_defaults(family: str) -> dict:
    """Constructor defaults of the installed BoxMOT family class and its base class."""
    from boxmot import BotSort, ByteTrack, OcSort
    from boxmot.trackers.basetracker import BaseTracker
    defaults = {}
    for c in (BaseTracker, {"bytetrack": ByteTrack, "ocsort": OcSort, "botsort": BotSort}[family]):
        for k, p in inspect.signature(c.__init__).parameters.items():
            if p.default is not inspect.Parameter.empty and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                defaults[k] = p.default
    return defaults


def _strong(field: str, value, cmc_available=()) -> list:
    if isinstance(value, bool):
        return [not value]
    if field == "cmc_method":
        return [m for m in cmc_available if m != value][:1]
    return [x for x in STRONG_TRACKER.get(field, ()) if not _equal(x, value)]


def _tracks(outputs: list) -> np.ndarray:
    """(frame, x1, y1, x2, y2, id, conf, cls, det) with the identities renumbered by order of appearance."""
    rows = [(i, *l) for i, s in enumerate(outputs, start=1) for l in np.asarray(s, np.float64).reshape(-1, 8)]
    a = np.array(rows, np.float64).reshape(-1, 9)
    if len(a):
        a = a[np.lexsort((a[:, 8], a[:, 0]))]
        ids: dict = {}
        for j in range(len(a)):
            a[j, 5] = ids.setdefault(int(a[j, 5]), len(ids) + 1)
    return a


def _frames_changed(a: np.ndarray, b: np.ndarray, n: int) -> int:
    return sum(int(not np.array_equal(a[a[:, 0] == i], b[b[:, 0] == i])) for i in range(1, n + 1))


def check_trackers(profiles: dict, clip_dets: list, frame_reader, fps: float, *, construct, profile_kwargs, verify,
                   update, defaults_of=installed_defaults, no_effect=None, cmc_methods=(), common_fields=TRACKING_FIELDS,
                   fixed_common=None) -> dict:
    """Each family on the clip: defaults, read-back, the effect of each field and the fields declared without effect."""
    # a confirmation hard-coded in a family cannot be perturbed there
    fixed_common = ({f: {"confirm_frames"} for f in TRACKER_FIXED_CONFIRMATION} if fixed_common is None else fixed_common)
    report, problems, n = {}, [], len(clip_dets)

    def run(family, kw, reader):
        t = construct(family, kw)
        return t, _tracks([update(t, clip_dets[i - 1], reader(i)) for i in range(1, n + 1)])
    for family, declared in sorted(profiles.items()):
        # one profile, or a list with one profile per level; the first is used for the own fields
        levels = list(declared) if isinstance(declared, list) else [declared]
        profile = levels[0]
        reader = frame_reader(family)
        r: dict = {"defaults": {}, "common": {}, "fields": {}, "fixed": {}, "no_effect": {}}
        defaults = defaults_of(family)
        without_effect = no_effect_fields(family) if no_effect is None else no_effect.get(family, {})
        for k, v in profile.items():
            if k in ("frame_rate", "reid_weights"):
                continue
            if k in common_fields:
                r["common"][k] = v
                continue
            if k not in defaults:
                problems.append(f"{family}: {k} has no default in the constructor of the installed BoxMOT")
            elif not _equal(defaults[k], v):
                problems.append(f"{family}: {k} = {v!r} is not the default of the installed BoxMOT ({defaults[k]!r}); "
                                "the parameters of a family are not common to the trackers")
            r["defaults"][k] = defaults.get(k)
        base_kw = profile_kwargs(family, profile, fps)
        t, base = run(family, base_kw, reader)
        try:
            r["resolved"] = verify(t, family, profile, fps)
        except ValueError as e:
            problems.append(f"{family}: {e}")
        r["base_rows"] = int(len(base))
        for k, v in profile.items():
            if k in _NOT_PERTURBABLE:
                continue
            if k in fixed_common.get(family, ()):
                r["fixed"][k] = {"value": v, "status": "fixed in the code of the family"}
                continue
            if k == "frame_rate":
                requests = [("rate x 0.2", profile_kwargs(family, profile, fps * 0.2))]
            else:
                requests = [(x, profile_kwargs(family, {**profile, k: x}, fps)) for x in _strong(k, v, cmc_methods)]
                if not requests:
                    problems.append(f"{family}: no strong perturbation declared for {k}")
                    continue
            tries = []
            for value, kw in requests:
                try:
                    changed = _frames_changed(base, run(family, kw, reader)[1], n)
                except Exception as e:                       # noqa: BLE001
                    problems.append(f"{family}: {k} = {value!r} failed: {type(e).__name__}: {e}")
                    break
                tries.append({"value": value, "frames_changed": changed})
                if changed:
                    break
            if not tries:
                continue
            r["fields"][k] = {"tries": tries, "frames_changed": max(t["frames_changed"] for t in tries), "frames": n}
            if not r["fields"][k]["frames_changed"]:
                problems.append(f"{family}: {k} did not change the tracks on any of the {n} frames of the clip under a "
                                f"strong perturbation ({', '.join(repr(t['value']) for t in tries)})")
        for k in sorted(without_effect):
            if k not in defaults:
                r["no_effect"][k] = {"status": "not a constructor parameter; BoxMOT derives it or does not receive it"}
                continue
            tries = [{"value": x, "frames_changed": _frames_changed(base, run(family, {**base_kw, k: x}, reader)[1], n)}
                     for x in _strong(k, defaults[k], cmc_methods)]
            changed = max((t["frames_changed"] for t in tries), default=0)
            r["no_effect"][k] = {"tries": tries, "frames_changed": changed, "frames": n}
            if changed:
                problems.append(f"{family}: {k} now changes the tracks in the installed version ({changed} frames); "
                                "review catalog.TRACKER_NO_EFFECT")
        if len(levels) > 1:
            varying = [k for k in common_fields if len({p.get(k) for p in levels}) > 1]
            if not varying:
                problems.append(f"{family}: {len(levels)} profiles declared without any different common parameter")
            label = {id(p): ", ".join(f"{k}={level_tag(p[k])}" for k in varying) for p in levels}
            outputs = {label[id(levels[0])]: base}
            for p in levels[1:]:
                outputs[label[id(p)]] = run(family, profile_kwargs(family, p, fps), reader)[1]
            pairs = []
            for a, b in itertools.combinations(sorted(outputs), 2):
                changed = _frames_changed(outputs[a], outputs[b], n)
                pairs.append({"a": a, "b": b, "frames_changed": changed})
                if not changed:
                    problems.append(f"{family}: the levels {a} and {b} give the same tracks on the {n} frames of the "
                                    "clip: one level repeats the other")
            r["levels"] = {"fields": varying, "rows": {k: int(len(v)) for k, v in outputs.items()}, "pairs": pairs}
        report[family] = r
    return {"trackers": report, "problems": problems}


def seal_record(resolved: dict, versions: dict, problems: list[str], detail: dict, device: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "seal_version": SEAL_VERSION, "created_utc": _now(),
            "status": "ok" if not problems else "refused", "catalog_identity": catalog_identity(resolved),
            "versions": versions, "device": str(device), "problems": list(problems), **detail}


def write_seal(path, record: dict) -> Path:
    return write_json_atomic(path, record)


def require_seal(path, resolved: dict, versions: dict) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SpecError("no check-options seal: run check-options before the experiment")
    s = json.loads(p.read_text(encoding="utf-8"))
    if s.get("status") != "ok":
        first = (s.get("problems") or ["?"])[0]
        raise SpecError(f"the last check-options refused the catalog ({len(s.get('problems') or [])} problems; the first: "
                        f"{first}); fix it and run check-options again")
    if s.get("catalog_identity") != catalog_identity(resolved):
        raise SpecError("the catalog changed after the last check-options: run check-options again")
    if s.get("versions") != versions:
        raise SpecError("the library versions changed after the last check-options: run check-options again")
    return s

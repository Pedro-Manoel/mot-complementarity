"""The three BoxMOT trackers, built through the constructor from a catalog profile and verified by reading it back.

Only BoT-SORT reads pixels (ReID and camera motion); ByteTrack and OC-SORT only need a frame of the right shape.
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np

from ..catalog import TRACKER_FIXED_CONFIRMATION, TRACKING_FIELDS, own_fields

FAMILIES = ("bytetrack", "ocsort", "botsort")
_WITH_PIXELS = frozenset({"botsort"})
# Profile fields that are not instance attributes; they are checked through what they produce.
_SPECIAL = frozenset({"frame_rate", "reid_weights", *TRACKING_FIELDS})
# Consumed by the constructor without being stored (BotSort keeps only the CMC instance).
_DERIVED = {"botsort": frozenset({"cmc_method"})}


def needs_image(family: str) -> bool:
    if family not in FAMILIES:
        raise ValueError(f"unknown tracker family: {family!r}")
    return family in _WITH_PIXELS


def _literal_version(module: str):
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        return None
    m = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", Path(spec.origin).read_text(encoding="utf-8", errors="replace"), re.M)
    return m.group(1) if m else None


def tracker_versions() -> dict:
    """Versions that determine the tracking output, read without importing BoxMOT."""
    out = {}
    for p in ("boxmot", "torch"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    out["boxmot_module"] = _literal_version("boxmot")
    return out


def _fps(profile: dict, fps: float) -> int:
    fr = profile.get("frame_rate", "from_sequence")
    return int(round(float(fps) if fr == "from_sequence" else float(fr)))


def lost_track_frames(profile: dict, fps: float) -> int:
    """Frames without a detection that the identity of a lost track survives."""
    return max(1, int(round(_fps(profile, fps) * float(profile["lost_track_seconds"]))))


def _buffer_and_frame_rate(profile: dict, fps: float) -> tuple[int, int]:
    """track_buffer and frame_rate for max_time_lost = frames - 1, as these families keep an identity one frame past it.
    When the real rate cannot reach that value, frame_rate 30 is passed: BoxMOT uses it only for max_time_lost."""
    fr, target = _fps(profile, fps), lost_track_frames(profile, fps) - 1
    tb = math.ceil(target * 30.0 / fr)
    if int(fr / 30.0 * tb) == target:
        return int(tb), fr
    return int(target), 30


def create(family: str, profile: dict, fps: float, device: str, reid_path):
    if family not in FAMILIES:
        raise ValueError(f"unknown tracker family: {family!r}")
    extra = sorted(set(profile) - set(own_fields(family)) - set(TRACKING_FIELDS))
    if extra:
        raise ValueError(f"{family}: fields outside the profile of the family: {', '.join(extra)}")
    missing = [k for k in TRACKING_FIELDS if k not in profile]
    if missing:
        raise ValueError(f"{family}: missing common parameters {', '.join(missing)}")
    return construct(family, profile_kwargs(family, profile, fps), device, reid_path)


def _retention(p: dict, fps: float) -> dict:
    tb, fr = _buffer_and_frame_rate(p, fps)
    return {"track_buffer": tb, "frame_rate": fr}


def profile_kwargs(family: str, profile: dict, fps: float) -> dict:
    """BoxMOT constructor arguments: the common parameters translated for the family, plus its own fields."""
    p, high = dict(profile), float(profile["high_score_thresh"])
    if family == "bytetrack":
        return {"min_conf": p["min_conf"], "track_thresh": high, "match_thresh": p["match_thresh"],
                **_retention(p, fps), "per_class": p["per_class"]}
    if family == "ocsort":
        # OC-SORT does not count the frame where a track starts, so the same confirmation is confirm_frames - 1
        return {"det_thresh": high, "max_age": lost_track_frames(p, fps), "min_hits": int(p["confirm_frames"]) - 1,
                **{k: p[k] for k in ("iou_threshold", "delta_t", "inertia", "use_byte", "Q_xy_scaling", "Q_s_scaling")}}
    if family == "botsort":
        ret = _retention(p, fps)
        return {"track_high_thresh": high, "track_low_thresh": p["track_low_thresh"], "new_track_thresh": high,
                "track_buffer": ret["track_buffer"], "match_thresh": p["match_thresh"],
                "proximity_thresh": p["proximity_thresh"], "appearance_thresh": p["appearance_thresh"],
                "cmc_method": p["cmc_method"], "frame_rate": ret["frame_rate"],
                "fuse_first_associate": p["fuse_first_associate"],
                "with_reid": p["with_reid"]}
    raise ValueError(f"unknown tracker family: {family!r}")


_BOXMOT_QUIET = False


def _quiet_boxmot() -> None:
    """Hides the log line BoxMOT prints for every tracker it creates; must run after importing boxmot."""
    global _BOXMOT_QUIET
    if _BOXMOT_QUIET:
        return
    try:
        import boxmot  # noqa: F401
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING", format="<level>{level: <8}</level> | <level>{message}</level>")
        _BOXMOT_QUIET = True
    except Exception:
        pass


def construct(family: str, kwargs: dict, device: str, reid_path):
    _quiet_boxmot()
    if family == "bytetrack":
        from boxmot import ByteTrack
        return ByteTrack(**kwargs)
    if family == "ocsort":
        from boxmot import OcSort
        return OcSort(**kwargs)
    if family == "botsort":
        import torch
        from boxmot import BotSort
        if not reid_path:
            raise ValueError("BoT-SORT needs the path of the ReID weights")
        # half has no default in BotSort; the ReID runs in fp32, like the detectors
        return BotSort(reid_weights=Path(reid_path), device=torch.device(device), half=False, **kwargs)
    raise ValueError(f"unknown tracker family: {family!r}")


def _equal(actual, v) -> bool:
    if isinstance(v, float) and not isinstance(v, bool):
        return actual is not None and abs(float(actual) - float(v)) <= 1e-12
    return actual == v


def verify_profile(t, family: str, profile: dict, fps: float) -> dict:
    """Reads every profile field back from the instance; raises on a mismatch and returns the values read."""
    resolved = {}
    derived = _DERIVED.get(family, frozenset())
    for k, v in profile.items():
        if k in _SPECIAL or k in derived:
            continue
        if not hasattr(t, k):
            raise ValueError(f"{family}: the profile field {k} does not exist in the installed BoxMOT version")
        actual = getattr(t, k)
        if not _equal(actual, v):
            raise ValueError(f"{family}: {k} resolved to {actual!r}, the profile asks for {v!r}")
        resolved[k] = actual
    high, frames = float(profile["high_score_thresh"]), lost_track_frames(profile, fps)
    # fields set from the common parameters, and the asso_threshold OC-SORT derives from iou_threshold
    if family == "bytetrack":
        expected = {"track_thresh": high, "det_thresh": high, "max_time_lost": frames - 1}
    elif family == "ocsort":
        expected = {"det_thresh": high, "max_age": frames, "min_hits": int(profile["confirm_frames"]) - 1,
                    "asso_threshold": profile["iou_threshold"]}
    else:
        expected = {"track_high_thresh": high, "new_track_thresh": high, "max_time_lost": frames - 1}
    for k, v in expected.items():
        actual = getattr(t, k, None)
        if not _equal(actual, v):
            raise ValueError(f"{family}: {k} {actual!r} differs from what the common parameters ask for ({v!r})")
        resolved[k] = actual
    fixed = TRACKER_FIXED_CONFIRMATION.get(family)
    if fixed is not None and int(profile["confirm_frames"]) != fixed:
        raise ValueError(f"{family}: BoxMOT confirms a track after {fixed} frames; the profile asks for "
                         f"{profile['confirm_frames']}")
    if "cmc_method" in derived:
        from boxmot.motion.cmc import get_cmc_method
        cls, actual = get_cmc_method(profile["cmc_method"]), getattr(t, "cmc", None)
        if (cls is None) != (actual is None) or (cls is not None and not isinstance(actual, cls)):
            raise ValueError(f"{family}: cmc {type(actual).__name__} differs from the profile {profile['cmc_method']!r}")
        resolved["cmc_class"] = None if actual is None else type(actual).__name__
    if family == "botsort" and profile.get("with_reid"):
        if getattr(t, "model", None) is None:
            raise ValueError("botsort: with_reid true without the ReID model loaded")
        half = getattr(t.model, "half", None)
        if half is not False:
            raise ValueError(f"botsort: ReID with half {half!r}; the protocol uses fp32 (catalog.TRACKER_FIXED)")
        resolved["reid_half"] = half
    resolved["lost_track_frames"] = frames
    resolved["frame_rate_used"] = None if family == "ocsort" else _retention(profile, fps)["frame_rate"]
    return resolved


def update(t, dets, frame) -> np.ndarray:
    """One frame. `dets` (N, 6) [x1, y1, x2, y2, score, class]; returns the (M, 8) array of BoxMOT."""
    d = np.asarray(dets, np.float32).reshape(-1, 6)
    out = np.asarray(t.update(d, frame), dtype=np.float64)
    return out.reshape(-1, 8) if out.size else np.empty((0, 8))


class _Adapter:
    def __init__(self, t, resolved: dict):
        self.t, self.resolved = t, resolved

    def update(self, dets, frame):
        return update(self.t, dets, frame)


def factory(device: str, reid_path):
    """Returns `make(config, fps)` for `tracking.track_sequence`, which builds the tracker and verifies its profile."""
    def make(config: dict, fps: float):
        family, profile = config["tracker"]["family"], config["tracker"]["profile"]
        t = create(family, profile, fps, device, reid_path)
        return _Adapter(t, verify_profile(t, family, profile, fps))
    return make

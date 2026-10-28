"""The declarative catalog: detector x resolution x tracker family x tracker parameter levels.

`build_catalog` expands catalog.yaml and the profiles of trackers.yaml into the full product of configurations. Only
parameters common to every detector, or to every tracker, are accepted. Configurations of the same detector variant
share a `cache_key`, and therefore the detection cache.
"""
from __future__ import annotations

import itertools

from .schema import SCHEMA_VERSION, SpecError, is_unresolved, require_fields, stable_id

# The resolution r is the longer side of the network input for every detector.
RESOLUTION_SEMANTICS = "long_side"
RESOLUTION_MECHANISM = {
    "ultralytics": "imgsz=r: letterbox with longer side r",
    "dfine": "processor with size r x r: square, the preprocessing of the checkpoint (longer side r)",
    "torchvision": "transform with min_size = max_size = r: longer side r",
}
# YOLO rounds a non-multiple up (1290 and 1300 both give 1312) and D-FINE fails in the FPN.
RESOLUTION_MULTIPLE = 32

DETECTION_FIELDS = ("conf_floor", "max_det", "precision")
PRECISIONS = ("fp32",)
_REFUSED_DETECTION = {
    "nms_iou": "not common to every detector: D-FINE and YOLO26 have no NMS as distributed, and Faster R-CNN keeps its "
               "default NMS",
    "classes": "the module detects persons only, selected from the output of each model",
    "batch": "every detector processes one frame at a time",
}

# Common to the three trackers; backends.trackers.profile_kwargs translates them into the fields of each family.
TRACKING_FIELDS = ("lost_track_seconds", "high_score_thresh", "confirm_frames")
# Suffix of a varying field in the profile name; a field with a single level gets no suffix.
TRACKING_TAGS = {"lost_track_seconds": "s", "high_score_thresh": "h", "confirm_frames": "c"}
# Family fields set from a common parameter; they cannot be declared in a profile.
TRACKER_COMMON_MAP = {
    "bytetrack": {"track_thresh": "high_score_thresh", "track_buffer": "lost_track_seconds"},
    "ocsort": {"det_thresh": "high_score_thresh", "max_age": "lost_track_seconds", "min_hits": "confirm_frames"},
    "botsort": {"track_high_thresh": "high_score_thresh", "new_track_thresh": "high_score_thresh",
                "track_buffer": "lost_track_seconds"},
}
# Confirmation hard-coded by BoxMOT in these families; confirm_frames must equal it.
TRACKER_FIXED_CONFIRMATION = {"bytetrack": 2, "botsort": 2}
# Low threshold of the second association; must stay below high_score_thresh.
_LOW_THRESHOLD = {"bytetrack": "min_conf", "botsort": "track_low_thresh"}

# Own fields of each family that change the tracks; their values must be the BoxMOT defaults.
TRACKER_FIELDS = {
    "bytetrack": ("match_thresh", "min_conf", "per_class", "frame_rate"),
    "ocsort": ("iou_threshold", "delta_t", "inertia", "use_byte", "Q_xy_scaling", "Q_s_scaling"),
    "botsort": ("track_low_thresh", "match_thresh", "proximity_thresh", "appearance_thresh", "cmc_method", "with_reid",
                "fuse_first_associate", "reid_weights", "frame_rate"),
}
# Accepted by BoxMOT 17.0.0 but without effect on the tracks; declaring them is refused.
TRACKER_NO_EFFECT = {
    "bytetrack": {"max_age": "no effect (measured)", "min_hits": "no effect (measured)",
                  "iou_threshold": "no effect (measured)", "max_obs": "no effect (measured)",
                  "det_thresh": "BoxMOT replaces it with track_thresh"},
    "ocsort": {"asso_threshold": "BoxMOT derives it from iou_threshold", "min_conf": "only acts with use_byte, which is false",
               "max_obs": "no effect (measured)", "per_class": "no effect (measured)",
               "frame_rate": "OC-SORT does not use the frame rate"},
    "botsort": {"det_thresh": "no effect (measured)", "max_age": "no effect (measured)", "min_hits": "no effect (measured)"},
}
TRACKER_FIXED = {"botsort": {"half": "the ReID precision is fp32 (half false), as for the detectors; BotSort has no default"}}
FRAME_RATE_VALUE = "from_sequence"


def own_fields(family: str) -> tuple:
    return tuple(TRACKER_FIELDS[family])


def no_effect_fields(family: str) -> dict:
    return dict(TRACKER_NO_EFFECT[family])


def lowest_score_field(family: str) -> str:
    """Field with the lowest score the family consumes; the cache floor must stay below it."""
    return _LOW_THRESHOLD.get(family) or "high_score_thresh"


def lowest_consumed_threshold(profiles: dict) -> float:
    values = []
    for name, profile in profiles.items():
        family = profile.get("family", name)
        values.append(float(profile[lowest_score_field(family)]))
    return min(values)


def _check_detection(block) -> dict:
    if not isinstance(block, dict):
        raise SpecError("detection: block missing")
    for field in sorted(set(block) - set(DETECTION_FIELDS)):
        raise SpecError(f"detection.{field}: {_REFUSED_DETECTION.get(field, 'unknown field')}")
    require_fields(block, DETECTION_FIELDS, "detection")
    if not 0.0 < float(block["conf_floor"]) < 1.0:
        raise SpecError(f"detection.conf_floor outside (0, 1): {block['conf_floor']!r}")
    if isinstance(block["max_det"], bool) or not isinstance(block["max_det"], int) or block["max_det"] <= 0:
        raise SpecError(f"detection.max_det must be a positive integer: {block['max_det']!r}")
    if block["precision"] not in PRECISIONS:
        raise SpecError(f"detection.precision {block['precision']!r}: only {', '.join(PRECISIONS)} is applied by every "
                        "detector")
    return dict(block)


def _check_detectors(items) -> list[dict]:
    if not isinstance(items, list) or not items:
        raise SpecError("detectors: empty list")
    out, names, identities = [], set(), set()
    for i, d in enumerate(items):
        require_fields(d, ("name", "backend", "checkpoint"), f"detectors[{i}]")
        unknown = sorted(set(d) - {"name", "backend", "checkpoint", "options"})
        if unknown:
            raise SpecError(f"detectors[{i}]: unknown fields {', '.join(unknown)}")
        if d["backend"] not in RESOLUTION_MECHANISM:
            raise SpecError(f"detectors[{i}]: unknown backend {d['backend']!r}; known: {sorted(RESOLUTION_MECHANISM)}")
        if d.get("options"):
            raise SpecError(f"detectors[{i}] ({d['name']}): model-specific options are not allowed "
                            f"({', '.join(sorted(d['options']))}); only parameters common to every detector, the rest "
                            "stays at the library default")
        if not isinstance(d["checkpoint"], dict) or not d["checkpoint"]:
            raise SpecError(f"detectors[{i}]: checkpoint must be a non-empty mapping")
        ident = {"name": str(d["name"]), "backend": d["backend"], "checkpoint": dict(d["checkpoint"])}
        if ident["name"] in names:
            raise SpecError(f"detectors: duplicate name {ident['name']!r}")
        key = stable_id({k: v for k, v in ident.items() if k != "name"})
        if key in identities:
            raise SpecError(f"detectors: {ident['name']!r} repeats the content of another detector")
        names.add(ident["name"])
        identities.add(key)
        out.append(ident)
    return out


def _check_resolutions(items) -> list[int]:
    if not isinstance(items, list) or not items:
        raise SpecError("resolutions: empty list")
    if any(not isinstance(r, int) or isinstance(r, bool) or r <= 0 for r in items):
        raise SpecError(f"resolutions: positive integers only, got {items!r}")
    if len(set(items)) != len(items):
        raise SpecError(f"resolutions: repeated values in {items!r}")
    bad = [r for r in items if r % RESOLUTION_MULTIPLE]
    if bad:
        raise SpecError(f"resolutions: {bad} are not multiples of {RESOLUTION_MULTIPLE}; with longer side = r, YOLO "
                        "would round up (1290 and 1300 both become 1312, identical outputs) and D-FINE breaks")
    return list(items)


def _check_profiles(profiles) -> list[tuple[str, str, dict]]:
    if not isinstance(profiles, dict) or not profiles:
        raise SpecError("trackers: no profile declared")
    out, contents, families = [], set(), {}
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise SpecError(f"trackers.{name}: a profile must be a mapping")
        family = profile.get("family", name)
        if family not in TRACKER_FIELDS:
            raise SpecError(f"trackers.{name}: unknown family {family!r}; known: {sorted(TRACKER_FIELDS)}")
        if family in families:
            raise SpecError(f"trackers.{name}: {family} already has the profile {families[family]!r}; the parameters of "
                            "a family are not common to the trackers, so only the family varies")
        families[family] = str(name)
        fields = {k: v for k, v in profile.items() if k != "family"}
        for field in sorted(set(fields) & set(TRACKING_FIELDS)):
            raise SpecError(f"trackers.{name}: {field} is a parameter common to the trackers; declare it once in the "
                            "tracking block")
        for field in sorted(set(fields) & set(TRACKER_COMMON_MAP[family])):
            raise SpecError(f"trackers.{name}: {field} comes from the common parameter "
                            f"tracking.{TRACKER_COMMON_MAP[family][field]} (the same idea in the three families); "
                            "remove it from the profile")
        no_effect = no_effect_fields(family)
        for field in sorted(set(fields) & set(no_effect)):
            raise SpecError(f"trackers.{name}: {field} does not change the tracks of {family} in BoxMOT 17.0.0 "
                            f"({no_effect[field]}); remove it from the profile")
        for field in sorted(set(fields) & set(TRACKER_FIXED.get(family, {}))):
            raise SpecError(f"trackers.{name}: {field} is not a catalog parameter: {TRACKER_FIXED[family][field]}")
        require_fields(fields, own_fields(family), f"trackers.{name}")
        unknown = sorted(set(fields) - set(own_fields(family)))
        if unknown:
            raise SpecError(f"trackers.{name}: fields that the family {family} does not have: {', '.join(unknown)}")
        if "frame_rate" in fields and fields["frame_rate"] != FRAME_RATE_VALUE:
            raise SpecError(f"trackers.{name}: frame_rate must be {FRAME_RATE_VALUE!r} (the video's rate, not a setting)")
        key = stable_id({"family": family, "profile": fields})
        if key in contents:
            raise SpecError(f"trackers.{name}: repeats the content of another profile")
        contents.add(key)
        out.append((str(name), family, fields))
    return out


def level_tag(v) -> str:
    return f"{v:g}"


def _check_level(field: str, v) -> None:
    if field == "lost_track_seconds":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(round(30 * float(v))) < 1:
            raise SpecError(f"tracking.lost_track_seconds must be positive and give at least one frame at 30 fps: {v!r}")
    elif field == "high_score_thresh":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < float(v) < 1.0:
            raise SpecError(f"tracking.high_score_thresh outside (0, 1): {v!r}")
    elif field == "confirm_frames":
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise SpecError(f"tracking.confirm_frames must be an integer >= 1: {v!r}")


def _check_tracking(block) -> dict:
    if not isinstance(block, dict):
        raise SpecError("tracking: block missing (the parameters common to the trackers)")
    for field in sorted(set(block) - set(TRACKING_FIELDS)):
        raise SpecError(f"tracking.{field}: unknown field; the common ones are {', '.join(TRACKING_FIELDS)}")
    require_fields(block, TRACKING_FIELDS, "tracking")
    levels = {}
    for field in TRACKING_FIELDS:
        v = block[field]
        items = list(v) if isinstance(v, list) else [v]
        if not items:
            raise SpecError(f"tracking.{field}: empty list; declare the value or at least one level")
        for x in items:
            if is_unresolved(x):
                raise SpecError(f"tracking.{field}: missing or unresolved level in {items!r}")
            _check_level(field, x)
        if len(set(items)) != len(items):
            raise SpecError(f"tracking.{field}: repeated values in {items!r}")
        levels[field] = items
    return levels


def build_catalog(spec: dict) -> list[dict]:
    """The full product detector x resolution x profile; raises `SpecError`."""
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise SpecError(f"schema_version {spec.get('schema_version')!r} not supported (expected {SCHEMA_VERSION})")
    detection = _check_detection(spec.get("detection"))
    detectors = _check_detectors(spec.get("detectors"))
    resolutions = _check_resolutions(spec.get("resolutions"))
    profiles = _check_profiles(spec.get("trackers"))
    levels = _check_tracking(spec.get("tracking"))
    varying = [k for k in TRACKING_FIELDS if len(levels[k]) > 1]
    combinations = [dict(zip(TRACKING_FIELDS, vs)) for vs in itertools.product(*(levels[k] for k in TRACKING_FIELDS))]
    for common in combinations:
        for name, family, fields in profiles:
            fixed = TRACKER_FIXED_CONFIRMATION.get(family)
            if fixed is not None and int(common["confirm_frames"]) != fixed:
                raise SpecError(f"tracking.confirm_frames {common['confirm_frames']}: {family} confirms a track after "
                                f"{fixed} frames, fixed in the BoxMOT code; the common value must be {fixed} while it is "
                                "in the catalog")
            low = _LOW_THRESHOLD.get(family)
            if low and not float(fields[low]) < float(common["high_score_thresh"]):
                raise SpecError(f"tracking.high_score_thresh {common['high_score_thresh']}: must be above the low "
                                f"threshold of {family} ({low} {fields[low]})")
    # one profile per combination of levels, named with the suffixes of the varying fields
    profiles = [(name + "".join(f"_{TRACKING_TAGS[k]}{level_tag(common[k])}" for k in varying), family,
                 {**fields, **common}, {k: common[k] for k in varying})
                for common in combinations for name, family, fields in profiles]

    floor = float(detection["conf_floor"])
    lowest = lowest_consumed_threshold({n: {"family": f, **c} for n, f, c, _ in profiles})
    if floor > lowest:
        raise SpecError(f"incompatible cache: the confidence floor {floor} is above the lowest threshold consumed by "
                        f"the trackers ({lowest}); detections they would use would be left out of the cache")

    configs = []
    for det in detectors:
        for r in resolutions:
            resolution = {"requested": r, "semantics": RESOLUTION_SEMANTICS}
            cache_key = stable_id({"detector": {k: v for k, v in det.items() if k != "name"},
                                   "resolution": resolution, "detection": detection})
            variant = f"{det['name']}_r{r}"
            for name, family, fields, level in profiles:
                config_id = f"{variant}__{name}"
                profile_hash = stable_id({"family": family, "profile": fields})
                configs.append({
                    "schema_version": SCHEMA_VERSION,
                    "config_id": config_id,
                    "config_hash": stable_id({"cache_key": cache_key, "tracker": {"family": family, "profile": fields}}),
                    "detector_variant_id": variant,
                    "cache_key": cache_key,
                    "detector": dict(det),
                    "resolution": dict(resolution),
                    "detection": dict(detection),
                    "tracker": {"name": name, "family": family, "profile": dict(fields), "profile_hash": profile_hash,
                                "levels": dict(level)},
                })
    return configs


def detector_variants(configs: list[dict]) -> list[dict]:
    """One entry per detector x resolution variant (`cache_key`), sorted by id."""
    by_id: dict[str, dict] = {}
    for c in configs:
        by_id.setdefault(c["detector_variant_id"], {
            "detector_variant_id": c["detector_variant_id"], "cache_key": c["cache_key"],
            "detector": dict(c["detector"]), "resolution": dict(c["resolution"]), "detection": dict(c["detection"]),
        })
    return [by_id[k] for k in sorted(by_id)]

# -*- coding: utf-8 -*-
"""Declarative catalog: the product, identities, cache keys and refused specifications."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from pedestrian_catalog import catalog as C
from pedestrian_catalog.schema import SpecError

ROOT = Path(__file__).resolve().parents[1]


def _profiles() -> dict:
    """Own fields of each family, at the BoxMOT 17.0.0 defaults."""
    return {
        "bytetrack": {"match_thresh": 0.8, "min_conf": 0.1, "per_class": False, "frame_rate": "from_sequence"},
        "ocsort": {"iou_threshold": 0.3, "delta_t": 3, "inertia": 0.2, "use_byte": False,
                   "Q_xy_scaling": 0.01, "Q_s_scaling": 0.0001},
        "botsort": {"track_low_thresh": 0.1, "match_thresh": 0.8, "proximity_thresh": 0.5, "appearance_thresh": 0.25,
                    "cmc_method": "ecc", "with_reid": True, "fuse_first_associate": False,
                    "reid_weights": "osnet_x0_25_msmt17", "frame_rate": "from_sequence"},
    }


def _common() -> dict:
    """Parameters common to the three trackers (the median of the BoxMOT defaults)."""
    return {"lost_track_seconds": 1.0, "high_score_thresh": 0.45, "confirm_frames": 2}


def _spec(**overrides) -> dict:
    s = {
        "schema_version": 1,
        "detection": {"conf_floor": 0.01, "max_det": 300, "precision": "fp32"},
        "tracking": _common(),
        "detectors": [
            {"name": "yolo26m", "backend": "ultralytics", "checkpoint": {"file": "yolo26m.pt"}},
            {"name": "dfine_l", "backend": "dfine", "checkpoint": {"hf_repo": "ustc-community/dfine-large-coco"}},
            {"name": "frcnn_r50_fpn_v2", "backend": "torchvision",
             "checkpoint": {"torchvision_weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
                            "file": "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"}},
        ],
        "resolutions": [640, 800, 1280],
        "trackers": _profiles(),
    }
    s.update(overrides)
    return s


def _spec_from(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec["trackers"] = yaml.safe_load((ROOT / "configs" / "trackers.yaml").read_text(encoding="utf-8"))["trackers"]
    return spec


def test_frozen_catalog_of_the_integration_tests_is_the_one_of_this_test():
    assert C.build_catalog(_spec_from(ROOT / "tests" / "test_configs" / "catalog.yaml")) == C.build_catalog(_spec())


def test_catalog_of_the_paper():
    spec = _spec_from(ROOT / "configs" / "catalog.yaml")
    expected = _spec(detectors=[{"name": "yolo26n", "backend": "ultralytics", "checkpoint": {"file": "yolo26n.pt"}},
                                *_spec()["detectors"]],
                     resolutions=[640, 960, 1280], tracking={**_common(), "high_score_thresh": [0.3, 0.45, 0.6]})
    cfgs = C.build_catalog(spec)
    assert cfgs == C.build_catalog(expected)
    assert len(cfgs) == 108 and len({c["detector_variant_id"] for c in cfgs}) == 12
    assert {c["detector"]["name"] for c in cfgs} == {"yolo26n", "yolo26m", "dfine_l", "frcnn_r50_fpn_v2"}
    assert {c["resolution"]["requested"] for c in cfgs} == {640, 960, 1280}
    assert {c["tracker"]["levels"]["high_score_thresh"] for c in cfgs} == {0.3, 0.45, 0.6}


def test_catalog_of_the_paper_has_the_identities_of_the_released_resolved_catalog():
    released = json.loads((ROOT / "manifests" / "catalog.resolved.json").read_text(encoding="utf-8"))["configs"]
    now = C.build_catalog(_spec_from(ROOT / "configs" / "catalog.yaml"))
    assert {c["config_id"]: c["config_hash"] for c in now} == {c["config_id"]: c["config_hash"] for c in released}
    assert {c["config_id"]: c["cache_key"] for c in now} == {c["config_id"]: c["cache_key"] for c in released}
    assert len(now) == 108


def test_full_3x3x3_product_without_duplicates():
    cfgs = C.build_catalog(_spec())
    assert len(cfgs) == 27
    assert len({c["config_id"] for c in cfgs}) == 27 and len({c["config_hash"] for c in cfgs}) == 27
    assert len({c["detector_variant_id"] for c in cfgs}) == 9
    assert len({c["cache_key"] for c in cfgs}) == 9
    assert {c["tracker"]["family"] for c in cfgs} == {"bytetrack", "ocsort", "botsort"}
    assert {c["resolution"]["requested"] for c in cfgs} == {640, 800, 1280}
    assert all(c["schema_version"] == 1 for c in cfgs)


def test_configuration_profile_joins_the_common_parameters_and_the_own_fields_of_the_family():
    cfgs = {c["config_id"]: c for c in C.build_catalog(_spec())}
    p = cfgs["yolo26m_r640__ocsort"]["tracker"]["profile"]
    assert p == {**_profiles()["ocsort"], **_common()}
    base = {i: (c["cache_key"], c["config_hash"]) for i, c in cfgs.items()}
    other = {c["config_id"]: (c["cache_key"], c["config_hash"])
             for c in C.build_catalog(_spec(tracking={**_common(), "lost_track_seconds": 2.0}))}
    assert all(other[i][0] == base[i][0] and other[i][1] != base[i][1] for i in base)


def test_declaring_the_nano_detector_gives_36_without_code_changes():
    s = _spec()
    s["detectors"].append({"name": "yolo26n", "backend": "ultralytics", "checkpoint": {"file": "yolo26n.pt"}})
    cfgs = C.build_catalog(s)
    assert len(cfgs) == 36 and len({c["cache_key"] for c in cfgs}) == 12


def test_adding_to_the_catalog_keeps_the_existing_identities():
    base = {c["config_id"]: (c["config_hash"], c["cache_key"]) for c in C.build_catalog(_spec())}
    s = _spec(resolutions=[640, 800, 1280, 960])
    s["detectors"].append({"name": "yolo26n", "backend": "ultralytics", "checkpoint": {"file": "yolo26n.pt"}})
    larger = {c["config_id"]: (c["config_hash"], c["cache_key"]) for c in C.build_catalog(s)}
    assert set(base) < set(larger) and len(larger) == 4 * 4 * 3
    assert all(larger[i] == base[i] for i in base)


def test_resolution_is_the_longer_side_for_every_detector():
    cfgs = C.build_catalog(_spec())
    assert {c["resolution"]["semantics"] for c in cfgs} == {"long_side"}
    assert set(C.RESOLUTION_MECHANISM) == {c["detector"]["backend"] for c in cfgs}


def test_resolution_not_multiple_of_32_is_refused():
    with pytest.raises(SpecError, match="not multiples of 32"):
        C.build_catalog(_spec(resolutions=[640, 1290, 1300]))
    cfgs = C.build_catalog(_spec(resolutions=[640, 800, 1312]))
    assert "yolo26m_r1312__bytetrack" in {c["config_id"] for c in cfgs}


def test_order_of_keys_and_lists_does_not_change_the_identity():
    base = C.build_catalog(_spec())
    s = _spec()
    s["detectors"] = [dict(reversed(list(d.items()))) for d in reversed(s["detectors"])]
    s["trackers"] = {f: dict(reversed(list(p.items()))) for f, p in reversed(list(s["trackers"].items()))}
    s["detection"] = dict(reversed(list(s["detection"].items())))
    s["tracking"] = dict(reversed(list(s["tracking"].items())))
    other = C.build_catalog(s)
    key = lambda cs: {(c["config_id"], c["config_hash"], c["cache_key"]) for c in cs}   # noqa: E731
    assert key(base) == key(other)


def test_cache_depends_on_what_changes_boxes_and_not_on_the_tracker():
    base = {c["config_id"]: c for c in C.build_catalog(_spec())}
    by_variant: dict = {}
    for c in base.values():
        by_variant.setdefault(c["detector_variant_id"], set()).add(c["cache_key"])
    assert all(len(k) == 1 for k in by_variant.values())
    s = _spec()
    s["trackers"]["bytetrack"]["match_thresh"] = 0.7
    tracker_changed = {c["config_id"]: c for c in C.build_catalog(s)}
    assert all(tracker_changed[i]["cache_key"] == base[i]["cache_key"] for i in base)
    assert tracker_changed["yolo26m_r640__bytetrack"]["config_hash"] != base["yolo26m_r640__bytetrack"]["config_hash"]
    assert tracker_changed["yolo26m_r640__ocsort"]["config_hash"] == base["yolo26m_r640__ocsort"]["config_hash"]
    for field, value in (("conf_floor", 0.05), ("max_det", 100)):
        s = _spec()
        s["detection"][field] = value
        other = {c["config_id"]: c for c in C.build_catalog(s)}
        assert all(other[i]["cache_key"] != base[i]["cache_key"] for i in base), field
    s = _spec()
    s["detectors"][0]["checkpoint"] = {"file": "yolo26m_other.pt"}
    other = {c["config_id"]: c for c in C.build_catalog(s)}
    assert other["yolo26m_r640__ocsort"]["cache_key"] != base["yolo26m_r640__ocsort"]["cache_key"]
    assert other["dfine_l_r640__ocsort"]["cache_key"] == base["dfine_l_r640__ocsort"]["cache_key"]


@pytest.mark.parametrize("field,reason", [("nms_iou", "NMS"), ("classes", "persons"), ("batch", "one frame")])
def test_detection_parameter_that_is_not_common_is_refused_with_the_reason(field, reason):
    s = _spec()
    s["detection"][field] = {"nms_iou": 0.7, "classes": ["person"], "batch": 1}[field]
    with pytest.raises(SpecError, match=f"detection.{field}: .*{reason}"):
        C.build_catalog(s)


def test_only_fp32_is_accepted():
    with pytest.raises(SpecError, match="fp32"):
        C.build_catalog(_spec(detection={**_spec()["detection"], "precision": "fp16"}))


def test_model_specific_option_is_refused():
    s = _spec()
    s["detectors"][0]["options"] = {"end2end": False}
    with pytest.raises(SpecError, match="model-specific options.*end2end"):
        C.build_catalog(s)
    s["detectors"][0]["options"] = {}
    assert len(C.build_catalog(s)) == 27
    s["detectors"][0]["resolution"] = 640
    with pytest.raises(SpecError, match="unknown fields resolution"):
        C.build_catalog(s)


def test_second_profile_of_the_same_family_is_refused():
    s = _spec()
    s["trackers"]["bytetrack_slow"] = dict(_profiles()["bytetrack"], match_thresh=0.7, family="bytetrack")
    with pytest.raises(SpecError, match="only the family varies"):
        C.build_catalog(s)


@pytest.mark.parametrize("family,field,common", [
    ("bytetrack", "track_thresh", "high_score_thresh"), ("bytetrack", "track_buffer", "lost_track_seconds"),
    ("ocsort", "det_thresh", "high_score_thresh"), ("ocsort", "max_age", "lost_track_seconds"),
    ("ocsort", "min_hits", "confirm_frames"), ("botsort", "track_high_thresh", "high_score_thresh"),
    ("botsort", "new_track_thresh", "high_score_thresh"), ("botsort", "track_buffer", "lost_track_seconds")])
def test_family_field_that_comes_from_a_common_parameter_is_refused(family, field, common):
    s = _spec()
    s["trackers"][family][field] = 1
    with pytest.raises(SpecError, match=f"{field} comes from the common parameter tracking.{common}"):
        C.build_catalog(s)


def test_common_parameter_declared_in_the_profile_is_refused():
    s = _spec()
    s["trackers"]["ocsort"]["high_score_thresh"] = 0.5
    with pytest.raises(SpecError, match="is a parameter common to the trackers"):
        C.build_catalog(s)


@pytest.mark.parametrize("change,error", [
    (lambda t: t.pop("confirm_frames"), "confirm_frames"),
    (lambda t: t.update(other=1), "tracking.other"),
    (lambda t: t.update(lost_track_seconds=0), "lost_track_seconds"),
    (lambda t: t.update(lost_track_seconds=0.01), "lost_track_seconds"),
    (lambda t: t.update(high_score_thresh=1.2), "high_score_thresh"),
    (lambda t: t.update(confirm_frames=0), "confirm_frames"),
    (lambda t: t.update(confirm_frames=3), "fixed in the BoxMOT code"),  # ByteTrack and BoT-SORT confirm after 2
    (lambda t: t.update(high_score_thresh=0.1), "above the low threshold"),  # equal to the min_conf of ByteTrack
])
def test_block_of_the_common_parameters_is_checked(change, error):
    s = _spec()
    change(s["tracking"])
    with pytest.raises(SpecError, match=error):
        C.build_catalog(s)
    with pytest.raises(SpecError, match="tracking"):
        C.build_catalog({k: v for k, v in _spec().items() if k != "tracking"})


@pytest.mark.parametrize("field", ["lost_track_seconds", "high_score_thresh", "confirm_frames"])
def test_text_value_of_a_common_parameter_is_refused(field):
    with pytest.raises(SpecError, match=field):
        C.build_catalog(_spec(tracking={**_common(), field: "default"}))


def test_only_with_ocsort_alone_can_the_confirmation_be_another():
    s = _spec(trackers={"ocsort": _profiles()["ocsort"]}, tracking={**_common(), "confirm_frames": 3})
    assert len(C.build_catalog(s)) == 9


@pytest.mark.parametrize("family,field,value", [
    ("bytetrack", "max_age", 30), ("bytetrack", "min_hits", 3), ("bytetrack", "iou_threshold", 0.3),
    ("bytetrack", "max_obs", 50), ("bytetrack", "det_thresh", 0.45), ("ocsort", "asso_threshold", 0.3),
    ("ocsort", "min_conf", 0.1), ("ocsort", "max_obs", 50), ("ocsort", "per_class", False),
    ("ocsort", "frame_rate", "from_sequence"), ("botsort", "det_thresh", 0.3), ("botsort", "max_age", 30),
    ("botsort", "min_hits", 3)])
def test_field_that_does_not_change_the_tracks_is_refused_with_the_reason(family, field, value):
    s = _spec()
    s["trackers"][family][field] = value
    with pytest.raises(SpecError, match=f"{field} does not change the tracks of {family}"):
        C.build_catalog(s)


def test_min_conf_is_not_a_field_of_botsort():
    s = _spec(trackers={"botsort": {**_profiles()["botsort"], "min_conf": 0.1}})
    with pytest.raises(SpecError, match="fields that the family botsort does not have: min_conf"):
        C.build_catalog(s)


def test_own_and_no_effect_fields_of_ocsort():
    """min_conf only acts with use_byte, which is false by default."""
    assert "use_byte" in C.own_fields("ocsort")
    assert "min_conf" not in C.own_fields("ocsort")
    assert "min_conf" in C.no_effect_fields("ocsort")
    assert C.own_fields("bytetrack") == C.TRACKER_FIELDS["bytetrack"]


def test_reid_precision_is_fixed_and_is_not_a_catalog_parameter():
    s = _spec()
    s["trackers"]["botsort"]["half"] = False
    with pytest.raises(SpecError, match="half is not a catalog parameter.*fp32"):
        C.build_catalog(s)


def test_frame_rate_is_the_rate_of_the_video():
    s = _spec()
    s["trackers"]["bytetrack"]["frame_rate"] = 30
    with pytest.raises(SpecError, match="from_sequence"):
        C.build_catalog(s)


@pytest.mark.parametrize("family,field", [("bytetrack", "min_conf"), ("ocsort", "inertia"),
                                          ("botsort", "appearance_thresh"), ("botsort", "frame_rate")])
@pytest.mark.parametrize("mode", ["missing", "null", "pending"])
def test_incomplete_profile_is_refused(family, field, mode):
    s = _spec()
    if mode == "missing":
        del s["trackers"][family][field]
    else:
        s["trackers"][family][field] = None if mode == "null" else "TBD"
    with pytest.raises(SpecError, match=field):
        C.build_catalog(s)


def test_unknown_field_in_the_profile_is_refused():
    s = _spec()
    s["trackers"]["ocsort"]["iou_treshold"] = 0.3  # deliberate typo
    with pytest.raises(SpecError, match="iou_treshold"):
        C.build_catalog(s)


def test_cache_floor_above_the_lowest_consumed_threshold_is_refused():
    with pytest.raises(SpecError, match="incompatible"):
        C.build_catalog(_spec(detection={**_spec()["detection"], "conf_floor": 0.2}))
    s = _spec()
    s["trackers"]["botsort"]["track_low_thresh"] = 0.005
    with pytest.raises(SpecError, match="incompatible"):
        C.build_catalog(s)


def test_in_ocsort_the_lowest_consumed_threshold_is_the_common_high_threshold():
    assert C.lowest_consumed_threshold({"ocsort": {**_profiles()["ocsort"], **_common()}}) == pytest.approx(0.45)
    ocsort_only = _spec(trackers={"ocsort": _profiles()["ocsort"]},
                        detection={**_spec()["detection"], "conf_floor": 0.2})
    assert len(C.build_catalog(ocsort_only)) == 9


def test_lowest_score_field_of_each_family():
    assert C.lowest_score_field("bytetrack") == "min_conf"
    assert C.lowest_score_field("botsort") == "track_low_thresh"
    assert C.lowest_score_field("ocsort") == "high_score_thresh"


def test_repeated_resolutions_are_refused_in_the_list_itself():
    with pytest.raises(SpecError, match="resolutions: repeated values"):
        C.build_catalog(_spec(resolutions=[640, 800, 640]))


def test_lowest_consumed_threshold():
    assert C.lowest_consumed_threshold({f: {**p, **_common()} for f, p in _profiles().items()}) == pytest.approx(0.1)


@pytest.mark.parametrize("change", [
    lambda s: s["detectors"][1].update(backend="mmdet"),
    lambda s: s["detectors"].append(copy.deepcopy(s["detectors"][0])),
    lambda s: s["trackers"].update(deepsort=_profiles()["bytetrack"]),
    lambda s: s.update(resolutions=[]),
    lambda s: s.update(resolutions=[640, 640]),
    lambda s: s.update(resolutions=[0, 800]),
    lambda s: s.update(schema_version=2),
    lambda s: s["detection"].pop("max_det"),
    lambda s: s["detection"].update(max_det=0),
    lambda s: s["detection"].update(max_det=True),
    lambda s: s.pop("tracking"),
    lambda s: s.update(detectors=[]),
    lambda s: s["detectors"].append(dict(copy.deepcopy(s["detectors"][0]), name="yolo26m_copy")),  # same content
    lambda s: s["trackers"].update(bytetrack_copy=dict(_profiles()["bytetrack"], family="bytetrack")),
])
def test_invalid_specification_is_refused(change):
    s = _spec()
    change(s)
    with pytest.raises(SpecError):
        C.build_catalog(s)


def test_levels_of_a_common_parameter_multiply_the_configurations():
    cfgs = C.build_catalog(_spec(tracking={**_common(), "high_score_thresh": [0.3, 0.45, 0.6]}))
    assert len(cfgs) == 81 and len({c["config_id"] for c in cfgs}) == 81
    assert len({c["config_hash"] for c in cfgs}) == 81
    assert len({c["cache_key"] for c in cfgs}) == 9
    assert {c["config_id"] for c in cfgs if c["detector_variant_id"] == "yolo26m_r640"} == {
        f"yolo26m_r640__{f}_h{v}" for f in ("bytetrack", "ocsort", "botsort") for v in ("0.3", "0.45", "0.6")}
    one = next(c for c in cfgs if c["config_id"] == "yolo26m_r640__ocsort_h0.3")
    assert one["tracker"]["family"] == "ocsort" and one["tracker"]["levels"] == {"high_score_thresh": 0.3}
    assert one["tracker"]["profile"]["high_score_thresh"] == 0.3 and one["tracker"]["profile"]["confirm_frames"] == 2


def test_a_single_level_is_identical_to_the_single_value():
    assert C.build_catalog(_spec(tracking={**_common(), "high_score_thresh": [0.45]})) == C.build_catalog(_spec())


def test_levels_of_the_lost_track_time_and_of_two_parameters_at_once():
    cfgs = C.build_catalog(_spec(tracking={**_common(), "lost_track_seconds": [0.5, 1.0]}))
    assert len(cfgs) == 54
    one = next(c for c in cfgs if c["config_id"] == "dfine_l_r800__bytetrack_s0.5")
    assert one["tracker"]["profile"]["lost_track_seconds"] == 0.5
    assert one["tracker"]["levels"] == {"lost_track_seconds": 0.5}
    two = C.build_catalog(_spec(tracking={**_common(), "lost_track_seconds": [0.5, 1.0],
                                          "high_score_thresh": [0.3, 0.45]}))
    assert len(two) == 108
    assert {c["config_id"] for c in two
            if c["detector_variant_id"] == "dfine_l_r800" and c["tracker"]["family"] == "ocsort"} == {
        f"dfine_l_r800__ocsort_s{s}_h{h}" for s in ("0.5", "1") for h in ("0.3", "0.45")}


@pytest.mark.parametrize("block,error", [
    ({"high_score_thresh": [0.45, 0.45]}, "repeated values"),
    ({"high_score_thresh": []}, "empty list"),
    ({"high_score_thresh": [0.45, 1.2]}, "high_score_thresh"),
    ({"high_score_thresh": [0.05, 0.45]}, "above the low threshold"),  # below the min_conf of ByteTrack
    ({"confirm_frames": [2, 3]}, "fixed in the BoxMOT code"),  # ByteTrack and BoT-SORT confirm after 2
    ({"lost_track_seconds": [1.0, 0.01]}, "lost_track_seconds"),
])
def test_each_level_of_a_common_parameter_is_checked(block, error):
    with pytest.raises(SpecError, match=error):
        C.build_catalog(_spec(tracking={**_common(), **block}))


def test_cache_floor_is_checked_against_the_lowest_level():
    ocsort_only = _spec(trackers={"ocsort": _profiles()["ocsort"]},
                        detection={**_spec()["detection"], "conf_floor": 0.2},
                        tracking={**_common(), "high_score_thresh": [0.1, 0.45]})
    with pytest.raises(SpecError, match="incompatible"):
        C.build_catalog(ocsort_only)


def test_detector_variants_derive_from_the_catalog():
    cfgs = C.build_catalog(_spec())
    vs = C.detector_variants(cfgs)
    assert len(vs) == 9 and [v["detector_variant_id"] for v in vs] == sorted(v["detector_variant_id"] for v in vs)
    assert all(set(v) >= {"detector_variant_id", "cache_key", "detector", "resolution", "detection"} for v in vs)
    assert all(set(v["detection"]) == set(C.DETECTION_FIELDS) and "options" not in v["detector"] for v in vs)

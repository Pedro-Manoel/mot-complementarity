# -*- coding: utf-8 -*-
"""The options gate and its seal, with fake detectors and trackers that may ignore each parameter."""
from __future__ import annotations

import numpy as np
import pytest

from pedestrian_catalog import options_check as OC
from pedestrian_catalog.detection import expected_input_hw
from pedestrian_catalog.schema import SpecError


def _seq(tmp, name, w, h, n=6) -> dict:
    import cv2
    d = tmp / name / "img1"
    d.mkdir(parents=True)
    for i in range(1, n + 1):
        cv2.imwrite(str(d / f"{i:06d}.png"), np.full((h, w, 3), i, np.uint8))
    return {"dataset": "MOT20", "sequence_id": name, "dir": str(tmp / name), "im_dir": "img1", "im_ext": ".png",
            "index_width": 6, "n_frames": n, "width": w, "height": h, "fps": 25.0, "frames_manifest_sha256": name}


def test_probe_frames_are_spaced_in_the_first_sequence_of_each_format(tmp_path):
    lines = [{"dataset": "MOT20", "sequence_id": "B", "eligible": True, "width": 64, "height": 48},
             {"dataset": "MOT20", "sequence_id": "A", "eligible": True, "width": 64, "height": 48},
             {"dataset": "SOMPT22", "sequence_id": "C", "eligible": "true", "width": 32, "height": 24},
             {"dataset": "MOT17", "sequence_id": "D", "eligible": False, "width": 16, "height": 16}]
    seqs = {"A": _seq(tmp_path, "A", 64, 48, n=9), "C": _seq(tmp_path, "C", 32, 24, n=5)}
    fr = OC.probe_frames(lines, lambda l: seqs[l["sequence_id"]], 2)
    assert [(f["sequence"], f["frame"]) for f in fr] == [("SOMPT22/C", 1), ("SOMPT22/C", 3),
                                                         ("MOT20/A", 1), ("MOT20/A", 5)]


class _Model:
    def __init__(self, backend="ultralytics", fab=None, ignores=(), collapses=False, fails_at=None, lies=None,
                 input_hw=None):
        self.backend, self.ignores, self.collapses, self.fails_at, self.lies, self.input_hw = \
            backend, set(ignores), collapses, fails_at, lies or {}, input_hw
        self.factory_settings = fab if fab is not None else {"stride": 32, "end2end": True}
        self.name = "fake"


class _View:
    """12 boxes, scores 0.95 to 0.02, shifted by the frame and (unless collapsing) by the resolution."""

    def __init__(self, m, v):
        self.m, self.v, self._s = m, v, {}

    def detect(self, img):
        r, det = int(self.v["resolution"]["requested"]), self.v["detection"]
        if self.m.fails_at == r:
            raise RuntimeError("Sizes of tensors must match")
        h, w = img.shape[:2]
        val, shift = int(img[0, 0, 0]), 0 if self.m.collapses else r / 100.0
        floor = 0.01 if "conf_floor" in self.m.ignores else float(det["conf_floor"])
        cap = 10 ** 6 if "max_det" in self.m.ignores else int(det["max_det"])
        boxes = [[val + j + shift, 1, val + j + 5 + shift, 9, s, 0] for j, s in enumerate(np.linspace(0.95, 0.02, 12))]
        d = np.array([b for b in boxes if b[4] >= floor][:cap], np.float32).reshape(-1, 6)
        if self.m.backend == "ultralytics":
            self._s = {"conf": float(det["conf_floor"]), "max_det": int(det["max_det"]), "classes": None,
                       "end2end": True}
        else:
            self._s = {"conf_floor": float(det["conf_floor"]), "size": [r, r], "num_queries": 300, "n_candidates": 300}
        self._s.update(self.m.lies)
        hw = self.m.input_hw or expected_input_hw(self.m.backend, r, h, w)
        return d, {"input_hw": hw, "saturated": len(d) >= cap}

    def settings(self):
        return dict(self._s)


def _variants(backend="ultralytics", rs=(640, 800)):
    return [{"detector_variant_id": f"fake_r{r}", "detector": {"name": "fake", "backend": backend},
             "resolution": {"requested": r, "semantics": "long_side"},
             "detection": {"conf_floor": 0.01, "max_det": 300, "precision": "fp32"}} for r in rs]


def _check(tmp_path, model, backend="ultralytics", clip=None):
    seq = _seq(tmp_path, "S", 64, 48)
    frames = [{"sequence": "MOT20/S", "frame": i, "width": 64, "height": 48, "seq": seq} for i in (1, 4)]
    return OC.check_detectors(_variants(backend), frames, lambda d: model, _View, clip)


def test_detector_that_applies_everything_passes(tmp_path):
    r = _check(tmp_path, _Model())
    assert r["problems"] == []
    d = r["detectors"]["fake"]
    assert d["levels"]["fake_r640"]["input_hw"] == [[480, 640]] and d["levels"]["fake_r800"]["input_hw"] == [[608, 800]]
    assert d["strong"]["conf_floor"]["frames_changed"] == 2 and d["strong"]["max_det"]["frames_changed"] == 2


def test_level_that_repeats_another_is_reported(tmp_path):
    assert any("one level repeats the other" in p for p in _check(tmp_path, _Model(collapses=True))["problems"])


@pytest.mark.parametrize("field", ["conf_floor", "max_det"])
def test_common_parameter_that_does_not_change_the_output_is_reported(tmp_path, field):
    problems = _check(tmp_path, _Model(ignores=[field]))["problems"]
    assert any(f"{field} does not change the output" in p for p in problems)


def test_read_back_different_from_the_request_is_reported(tmp_path):
    assert any("classes read back as [0]" in p for p in _check(tmp_path, _Model(lies={"classes": [0]}))["problems"])


def test_input_different_from_the_expected_one_is_reported(tmp_path):
    assert any("input (48, 64) instead of" in p for p in _check(tmp_path, _Model(input_hw=(48, 64)))["problems"])


def test_level_that_breaks_is_reported_and_does_not_stop_the_rest(tmp_path):
    r = _check(tmp_path, _Model(fails_at=800))
    assert any(p.startswith("fake_r800: failed") and "Sizes of tensors" in p for p in r["problems"])
    assert r["detectors"]["fake"]["strong"]["conf_floor"]["frames_changed"] == 2


def test_model_stride_that_does_not_divide_the_resolution_multiple_is_reported(tmp_path):
    problems = _check(tmp_path, _Model(fab={"stride": 64, "end2end": True}))["problems"]
    assert any("model stride 64" in p for p in problems)


def test_dfine_checks_the_fixed_cap_and_does_not_perturb_it(tmp_path):
    r = _check(tmp_path, _Model(backend="dfine", fab={"feat_strides": [8, 16, 32]}), backend="dfine")
    assert r["problems"] == [] and "note" in r["detectors"]["fake"]["strong"]["max_det"]
    r = _check(tmp_path / "b", _Model(backend="dfine", fab={"feat_strides": [8, 16, 32]}, lies={"n_candidates": 200}),
               backend="dfine")
    assert any("n_candidates" in p for p in r["problems"])


def test_tracker_clip_comes_from_the_given_detector(tmp_path):
    seq = _seq(tmp_path / "c", "T", 64, 48, n=5)
    r = _check(tmp_path, _Model(), clip={"variant_id": "fake_r640", "seq": seq, "frames": 5})
    assert len(r["clip_detections"]) == 5 and all(d.shape[1] == 6 for d in r["clip_detections"])


class _FakeTracker:
    def __init__(self, kw, uses):
        self.kw, self.uses = kw, uses


def _update(t, dets, frame):
    out = []
    for j, d in enumerate(np.asarray(dets).reshape(-1, 6)):
        step = sum(float(t.kw[k]) for k in t.uses if k in t.kw) + \
            (float(t.kw.get("frame_rate", 0)) if "frame_rate" in t.uses else 0)
        out.append([d[0] + step, d[1], d[2] + step, d[3], j + 1, d[4], d[5], j])
    return np.array(out, np.float64).reshape(-1, 8)


def _check_tracker(profile, uses, defaults=None, verify=None):
    dets = [np.array([[10 + i, 5, 30 + i, 40, 0.9, 0], [50, 5, 70, 40, 0.6, 0]], np.float32) for i in range(6)]
    return OC.check_trackers(
        {"bytetrack": profile}, dets, lambda f: (lambda i: np.zeros((48, 64, 3), np.uint8)), 25.0,
        construct=lambda f, kw: _FakeTracker(kw, uses),
        profile_kwargs=lambda f, p, fps: {**{k: v for k, v in p.items() if k != "frame_rate"},
                                          "frame_rate": int(round(fps))},
        verify=verify or (lambda t, f, p, fps: {"read": True}), update=_update,
        defaults_of=lambda f: defaults or {"match_thresh": 0.8, "per_class": False, "max_age": 30},
        no_effect={"bytetrack": {"max_age": "no effect", "asso_threshold": "derived"}})


PROFILE = {"match_thresh": 0.8, "per_class": False, "frame_rate": "from_sequence"}


def test_tracker_that_applies_everything_passes():
    r = _check_tracker(PROFILE, uses={"match_thresh", "per_class", "frame_rate"})
    assert r["problems"] == []
    b = r["trackers"]["bytetrack"]
    assert set(b["fields"]) == {"match_thresh", "per_class", "frame_rate"}
    assert all(x["frames_changed"] == 6 for x in b["fields"].values())
    assert b["no_effect"]["max_age"]["frames_changed"] == 0
    assert b["no_effect"]["asso_threshold"]["status"].startswith("not a constructor parameter")


def test_field_that_only_acts_in_one_direction_passes_and_the_tries_are_recorded():
    def update(t, dets, frame):
        t2 = _FakeTracker({**t.kw, "match_thresh": t.kw["match_thresh"] if t.kw["match_thresh"] > 0.9 else 0.8}, t.uses)
        return _update(t2, dets, frame)
    dets = [np.array([[10 + i, 5, 30 + i, 40, 0.9, 0]], np.float32) for i in range(6)]
    r = OC.check_trackers(
        {"bytetrack": {"match_thresh": 0.8}}, dets, lambda f: (lambda i: np.zeros((48, 64, 3), np.uint8)), 25.0,
        construct=lambda f, kw: _FakeTracker(kw, {"match_thresh"}), profile_kwargs=lambda f, p, fps: dict(p),
        verify=lambda t, f, p, fps: {}, update=update, defaults_of=lambda f: {"match_thresh": 0.8}, no_effect={})
    assert r["problems"] == []
    assert r["trackers"]["bytetrack"]["fields"]["match_thresh"]["tries"] == [{"value": 0.3, "frames_changed": 0},
                                                                              {"value": 0.95, "frames_changed": 6}]


def test_profile_field_that_does_not_change_the_tracks_is_reported():
    r = _check_tracker(PROFILE, uses={"match_thresh", "frame_rate"})
    assert any("per_class did not change the tracks" in p for p in r["problems"])


def test_no_effect_field_that_now_has_an_effect_is_reported():
    r = _check_tracker(PROFILE, uses={"match_thresh", "per_class", "frame_rate", "max_age"})
    assert any("max_age now changes the tracks" in p for p in r["problems"])


def test_no_effect_is_checked_in_both_directions():
    def update(t, dets, frame):
        kw = {k: v for k, v in t.kw.items() if not (k == "max_age" and v <= 100)}
        return _update(_FakeTracker(kw, t.uses), dets, frame)
    dets = [np.array([[10 + i, 5, 30 + i, 40, 0.9, 0]], np.float32) for i in range(6)]
    r = OC.check_trackers(
        {"bytetrack": {"match_thresh": 0.8}}, dets, lambda f: (lambda i: np.zeros((48, 64, 3), np.uint8)), 25.0,
        construct=lambda f, kw: _FakeTracker(kw, {"match_thresh", "max_age"}), profile_kwargs=lambda f, p, fps: dict(p),
        verify=lambda t, f, p, fps: {}, update=update, defaults_of=lambda f: {"match_thresh": 0.8, "max_age": 30},
        no_effect={"bytetrack": {"max_age": "no effect"}})
    assert any("max_age now changes the tracks" in p for p in r["problems"])
    assert [t["value"] for t in r["trackers"]["bytetrack"]["no_effect"]["max_age"]["tries"]] == [1, 120]


def test_ocsort_min_conf_is_checked_as_a_no_effect_field_by_default():
    class _Fake:
        def __init__(self, kw):
            self.kw = kw

    def update(t, dets, frame):
        step = float(t.kw["iou_threshold"]) + (1.0 if t.kw.get("use_byte") else 0.0) + float(t.kw.get("min_conf", 0.0))
        return np.array([[d[0] + step, d[1], d[2], d[3], j + 1, d[4], d[5], j]
                         for j, d in enumerate(np.asarray(dets).reshape(-1, 6))], np.float64).reshape(-1, 8)
    dets = [np.array([[10 + i, 5, 30 + i, 40, 0.9, 0]], np.float32) for i in range(6)]
    r = OC.check_trackers(
        {"ocsort": {"iou_threshold": 0.3, "use_byte": False}}, dets,
        lambda f: (lambda i: np.zeros((48, 64, 3), np.uint8)), 25.0,
        construct=lambda f, kw: _Fake(kw), profile_kwargs=lambda f, p, fps: dict(p), verify=lambda t, f, p, fps: {},
        update=update, defaults_of=lambda f: {"iou_threshold": 0.3, "use_byte": False, "min_conf": 0.1, "max_obs": 50})
    assert "min_conf" in r["trackers"]["ocsort"]["no_effect"]
    assert any("min_conf now changes the tracks" in p for p in r["problems"])  # the fake always uses min_conf


def test_common_parameters_are_not_compared_with_the_default_and_the_fixed_confirmation_is_not_perturbed():
    r = _check_tracker(dict(PROFILE, high_score_thresh=0.45, confirm_frames=2),
                       uses={"match_thresh", "per_class", "frame_rate", "high_score_thresh"})
    b = r["trackers"]["bytetrack"]
    assert r["problems"] == [] and b["common"] == {"high_score_thresh": 0.45, "confirm_frames": 2}
    assert b["fixed"]["confirm_frames"]["status"] == "fixed in the code of the family"
    assert "confirm_frames" not in b["fields"]
    assert b["fields"]["high_score_thresh"]["frames_changed"] == 6


def test_levels_of_a_common_parameter_that_repeat_another_are_reported():
    levels = [dict(PROFILE, high_score_thresh=h, confirm_frames=2) for h in (0.3, 0.45, 0.6)]
    r = _check_tracker(levels, uses={"match_thresh", "per_class", "frame_rate", "high_score_thresh"})
    assert r["problems"] == []
    lv = r["trackers"]["bytetrack"]["levels"]
    assert lv["fields"] == ["high_score_thresh"] and len(lv["pairs"]) == 3
    assert all(p["frames_changed"] == 6 for p in lv["pairs"])
    same = _check_tracker(levels, uses={"match_thresh", "per_class", "frame_rate"})  # the fake ignores the threshold
    assert sum("give the same tracks" in p for p in same["problems"]) == 3
    assert any("high_score_thresh=0.3" in p and "high_score_thresh=0.45" in p for p in same["problems"])


def test_a_single_level_does_not_change_the_gate_report():
    one = _check_tracker(dict(PROFILE, high_score_thresh=0.45, confirm_frames=2),
                         uses={"match_thresh", "per_class", "frame_rate", "high_score_thresh"})
    listed = _check_tracker([dict(PROFILE, high_score_thresh=0.45, confirm_frames=2)],
                            uses={"match_thresh", "per_class", "frame_rate", "high_score_thresh"})
    assert one == listed and "levels" not in one["trackers"]["bytetrack"]


def test_profile_outside_the_default_of_the_installed_version_is_reported():
    r = _check_tracker(dict(PROFILE, match_thresh=0.7), uses={"match_thresh", "per_class", "frame_rate"})
    assert any("match_thresh = 0.7 is not the default" in p for p in r["problems"])


def test_field_without_a_declared_strong_perturbation_is_reported():
    r = _check_tracker(dict(PROFILE, new_field=3), uses={"match_thresh", "per_class", "frame_rate"},
                       defaults={"match_thresh": 0.8, "per_class": False, "new_field": 3, "max_age": 30})
    assert any("no strong perturbation declared for new_field" in p for p in r["problems"])


def test_read_back_that_fails_is_reported():
    def verify(t, f, p, fps):
        raise ValueError("bytetrack: det_thresh 0.3 differs from track_thresh 0.45")
    r = _check_tracker(PROFILE, uses={"match_thresh", "per_class", "frame_rate"}, verify=verify)
    assert any("det_thresh 0.3 differs" in p for p in r["problems"])


def test_defaults_of_the_installed_boxmot_are_the_ones_of_the_profiles():
    from test_catalog import _profiles
    for family, profile in _profiles().items():
        default = OC.installed_defaults(family)
        assert {k: default[k] for k in profile if k not in ("frame_rate", "reid_weights")} == \
            {k: v for k, v in profile.items() if k not in ("frame_rate", "reid_weights")}, family


def _resolved():
    return {"configs": [{"config_hash": "a" * 16}],
            "detector_variants": [{"cache_key": "k" * 16, "detector": {"backend": "dfine"}}],
            "detector_limits": {"dfine_l": {"num_queries": 300}}, "tracker_assets": {}}


def test_approved_seal_is_only_valid_for_the_catalog_and_versions_it_was_made_with(tmp_path):
    res, versions = _resolved(), {"detectors": {"dfine": {"torch": "2.8"}}, "trackers": {"boxmot": "17.0.0"}}
    p = OC.write_seal(tmp_path / "seal.json", OC.seal_record(res, versions, [], {}, "cpu"))
    assert OC.require_seal(p, res, versions)["status"] == "ok"
    with pytest.raises(SpecError, match="the catalog changed"):
        OC.require_seal(p, {**res, "configs": [{"config_hash": "b" * 16}]}, versions)
    with pytest.raises(SpecError, match="library versions changed"):
        OC.require_seal(p, res, {**versions, "trackers": {"boxmot": "18.0.0"}})
    OC.write_seal(p, OC.seal_record(res, versions, ["dfine_l_r800 and dfine_l_r1280 give the same output"], {}, "cpu"))
    with pytest.raises(SpecError, match="refused the catalog.*same output"):
        OC.require_seal(p, res, versions)
    with pytest.raises(SpecError, match="no check-options seal"):
        OC.require_seal(tmp_path / "missing.json", res, versions)


def test_catalog_identity_changes_with_the_checkpoint_limit():
    res = _resolved()
    other = {**res, "detector_limits": {"dfine_l": {"num_queries": 100}}}
    assert OC.catalog_identity(res) != OC.catalog_identity(other)


def test_use_byte_of_ocsort_is_perturbed_like_the_other_own_fields():
    """Uses the real BoxMOT on CPU."""
    from pedestrian_catalog.backends import trackers as BT
    from test_catalog import _common, _profiles
    profile = {**_profiles()["ocsort"], **_common()}
    assert profile["use_byte"] is False and "min_conf" not in profile
    # the score drops to 0.3, between min_conf 0.1 and the 0.45 threshold: only BYTE keeps the track
    dets = [np.array([[100 + 2 * i, 100, 150 + 2 * i, 250, 0.9 if i < 10 else 0.3, 0]], np.float32) for i in range(30)]
    empty = np.zeros((480, 640, 3), np.uint8)
    r = OC.check_trackers({"ocsort": profile}, dets, lambda f: (lambda i: empty), 25.0,
                          construct=lambda f, kw: BT.construct(f, kw, "cpu", None), profile_kwargs=BT.profile_kwargs,
                          verify=BT.verify_profile, update=BT.update)
    o = r["trackers"]["ocsort"]
    assert o["fields"]["use_byte"]["tries"][0]["value"] is True
    assert not [p for p in r["problems"] if "use_byte" in p]

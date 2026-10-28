# -*- coding: utf-8 -*-
"""Output contract of the detector backends, checked with fake models (no weights needed)."""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from pedestrian_catalog import detection as D
from pedestrian_catalog.backends import detectors as DB

FRAME = np.zeros((1080, 1920, 3), np.uint8)


def test_validation_accepts_a_box_past_the_border_with_its_center_inside():
    d = np.array([[-30, 5, 40, 60, 0.9, 0], [1900, 1000, 1960, 1100, 0.5, 0]], np.float32)
    assert D._validate_output(d, 1080, 1920, 0.01, "q").shape == (2, 6)


def test_validation_refuses_a_center_outside_the_image_and_a_degenerate_box():
    with pytest.raises(D.InvalidOutput, match="center of 1 boxes outside the original image"):
        D._validate_output(np.array([[1925, 10, 1990, 80, 0.9, 0]], np.float32), 1080, 1920, 0.01, "q")
    with pytest.raises(D.InvalidOutput, match="degenerate boxes"):
        D._validate_output(np.array([[10, 10, 10, 80, 0.9, 0]], np.float32), 1080, 1920, 0.01, "q")
    # a box in the coordinates of an input larger than the frame
    with pytest.raises(D.InvalidOutput, match="outside the original image"):
        D._validate_output(np.array([[2400, 1300, 2500, 1500, 0.9, 0]], np.float32), 1080, 1920, 0.01, "q")


def test_adapter_contract_enters_the_cache_identity():
    v = {"detector_variant_id": "dfine_l_r800", "cache_key": "k" * 16, "checkpoint": {"sha256": "c" * 64}}
    s = {"dataset": "MOT20", "sequence_id": "S", "frames_manifest_sha256": "f" * 64}
    assert {b: DB.backend_versions(b)["adapter_contract"] for b in ("ultralytics", "dfine", "torchvision")} == \
        {"ultralytics": "3", "dfine": "4", "torchvision": "4"}
    versions = DB.backend_versions("torchvision")
    old = {**versions, "adapter_contract": "2"}
    env = {"kind": "cpu", "name": "test cpu"}
    assert D.detection_cell_identity(v, s, 42, versions, env)["cell_id"] != D.detection_cell_identity(v, s, 42, old, env)["cell_id"]


def test_expected_input_has_the_longest_side_r_in_all_three():
    for backend, (h, w, r), expected in (("torchvision", (1080, 1920, 640), (384, 640)),
                                         ("torchvision", (480, 640, 1280), (960, 1280)),
                                         ("torchvision", (1080, 1920, 1280), (736, 1280)),
                                         ("ultralytics", (1080, 1920, 640), (384, 640)),
                                         ("dfine", (1080, 1920, 800), (800, 800))):
        assert D.expected_input_hw(backend, r, h, w) == expected, backend


class _T:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)

    def __len__(self):
        return len(self.cls.a)


def test_yolo26_receives_only_the_common_parameters_and_person_is_separated_afterwards():
    a = object.__new__(DB._Ultralytics)
    calls = []

    class _Yolo:
        def predict(self, frame, **kw):
            calls.append(kw)
            a._input_hw = (384, 640)
            return [types.SimpleNamespace(boxes=_Boxes([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 9, 9]], [0.9, 0.8, 0.02], [0, 2, 0]))]
    a.model, a.device, a._input_hw, a.name = _Yolo(), "cpu", None, "yolo26m"
    v = {"detection": {"conf_floor": 0.01, "max_det": 3, "precision": "fp32"}, "resolution": {"requested": 640}}
    dets, info = a.detect(FRAME, v)
    assert set(calls[0]) == {"conf", "imgsz", "max_det", "verbose", "half", "device"}
    assert (calls[0]["conf"], calls[0]["imgsz"], calls[0]["max_det"]) == (0.01, 640, 3)
    assert dets.shape == (2, 6) and (dets[:, 5] == 0).all() and sorted(dets[:, 4].tolist()) == pytest.approx([0.02, 0.9])
    assert info["saturated"] is True and info["input_hw"] == (384, 640)  # max_det counts all classes


def test_faster_rcnn_receives_longest_side_floor_and_cap_and_keeps_the_factory_nms():
    a = object.__new__(DB._Torchvision)
    rh = types.SimpleNamespace(nms_thresh=0.5, score_thresh=0.05, detections_per_img=100)
    tr = types.SimpleNamespace(min_size=(800,), max_size=1333)

    class _Frcnn:
        roi_heads, transform, seen = rh, tr, None

        def parameters(self):
            return iter([torch.zeros(1)])

        def __call__(self, imgs):
            self.seen = (tr.min_size, tr.max_size, rh.nms_thresh, rh.score_thresh, rh.detections_per_img)
            return [{"boxes": torch.tensor([[0., 0., 10., 10.], [5., 5., 20., 20.]]), "scores": torch.tensor([0.9, 0.5]),
                     "labels": torch.tensor([1, 3])}]
    a.model, a._input_hw, a._resized = _Frcnn(), None, None
    v = {"detection": {"conf_floor": 0.01, "max_det": 300, "precision": "fp32"}, "resolution": {"requested": 640}}
    dets, info = a.detect(FRAME, v)
    assert a.model.seen == ((640,), 640, 0.5, 0.01, 300)
    assert dets.shape == (1, 6) and info["saturated"] is False
    assert a.settings()["nms_thresh"] == 0.5 and a.settings()["max_size"] == 640


def test_dfine_receives_the_r_square_without_a_module_cap_and_the_common_cap_must_be_the_queries():
    a = object.__new__(DB._DFine)
    seen = {}

    class _Inputs(dict):
        def to(self, _d):
            return self

    class _Proc:
        def __call__(self, images, size, return_tensors):
            seen["size"] = size
            return _Inputs(pixel_values=torch.zeros(1, 3, size["height"], size["width"]))

        def post_process_object_detection(self, outputs, threshold, target_sizes):
            seen["threshold"] = threshold
            return [{"scores": torch.tensor([0.9, 0.8, 0.7, 0.2]), "boxes": torch.tensor([[0., 0., 5., 5.]] * 4),
                     "labels": torch.tensor([0, 0, 0, 2])}]

    class _Model:
        device = "cpu"

        def __call__(self, **kw):
            return None
    a.processor, a.model, a.n_queries, a.name, a._last = _Proc(), _Model(), 4, "dfine_l", {}
    v = {"detection": {"conf_floor": 0.01, "max_det": 4, "precision": "fp32"}, "resolution": {"requested": 800}}
    dets, info = a.detect(FRAME, v)
    assert seen == {"size": {"height": 800, "width": 800}, "threshold": 0.0}
    assert len(dets) == 3 and info["input_hw"] == (800, 800) and info["saturated"] is True
    with pytest.raises(ValueError, match="fixed D-FINE queries"):
        a.detect(FRAME, {**v, "detection": {**v["detection"], "max_det": 1000}})


class _Fake:
    def __init__(self, cut):
        self.cut = cut

    def describe(self):
        return {"backend": "fake", "device": "cpu", "precision": "fp32", "versions": {"fake": "1"}, "label_space": {"name": "fake", "person_label": 0, "labels": {"0": "person"}}}

    def detect(self, frame):
        info = {"input_hw": (32, 64), "n_before_nms": None, "n_after_nms": 1,
                "raw": np.array([[-3, 1, 10, 20, 0.9, 0]], np.float32)}
        if self.cut is not None:
            info["candidate_cut_score"] = self.cut
        return np.array([[-3, 1, 10, 20, 0.9, 0]], np.float32), info  # 3 px past the left border


def test_manifest_measures_the_border_in_any_backend_and_the_cut_only_where_it_exists(tmp_path):
    import cv2
    d = tmp_path / "seq" / "img1"
    d.mkdir(parents=True)
    for i in range(1, 6):
        cv2.imwrite(str(d / f"{i:06d}.png"), np.full((24, 32, 3), i, np.uint8))
    seq = {"dataset": "MOT20", "sequence_id": "S", "dir": str(tmp_path / "seq"), "im_dir": "img1", "im_ext": ".png",
           "index_width": 6, "n_frames": 5, "width": 32, "height": 24, "frames_manifest_sha256": "f" * 64}
    for cut, cache in ((0.12, "a" * 16), (None, "b" * 16)):
        v = {"detector_variant_id": f"fake_{cache[0]}", "cache_key": cache, "checkpoint": {"sha256": "c" * 64},
             "detector": {"name": "fake", "backend": "fake"}, "resolution": {"requested": 640, "semantics": "long_side"},
             "detection": {"conf_floor": 0.01, "max_det": 300, "precision": "fp32"}}
        c = D.detect_sequence(seq, v, {"run_dir": tmp_path / "run", "seed": 42, "backend": _Fake(cut), "part_size": 2,
                                       "environment": {"kind": "cpu", "name": "test cpu"}})["counts"]
        assert c["n_beyond_image"] == 5 and c["max_overshoot_px"] == pytest.approx(3.0)
        if cut is None:
            assert c["max_candidate_cut_score"] is None and c["n_suppressed_by_nms"] is None
        else:
            assert c["max_candidate_cut_score"] == pytest.approx(0.12)

# -*- coding: utf-8 -*-
"""Real backends on CPU against corpus frames (`pytest -m weights`; skipped without configs/paths.local.yaml)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pedestrian_catalog import catalog as C
from pedestrian_catalog import detection as D
from pedestrian_catalog.backends import detectors as B

pytestmark = pytest.mark.weights

ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "configs" / "paths.local.yaml"
FRAMES = [("MOT17/train/MOT17-02-FRCNN", 1), ("MOT17/train/MOT17-05-FRCNN", 1)]


def _yaml(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _paper_spec() -> dict:
    spec = _yaml(ROOT / "configs" / "catalog.yaml")
    spec["trackers"] = _yaml(ROOT / "configs" / "trackers.yaml")["trackers"]
    return spec


def _catalog_variant_ids() -> list[str]:
    return [v["detector_variant_id"] for v in C.detector_variants(C.build_catalog(_paper_spec()))]


@pytest.fixture(scope="module")
def paths():
    if not PATHS.is_file():
        pytest.skip("configs/paths.local.yaml missing")
    return _yaml(PATHS)


@pytest.fixture(scope="module")
def variants():
    return C.detector_variants(C.build_catalog(_paper_spec()))


@pytest.fixture(scope="module")
def models(paths, variants):
    from pedestrian_catalog.provenance import resolve_checkpoint
    return {v["detector"]["name"]: B.load_model(v["detector"], paths, "cpu", checkpoint=resolve_checkpoint(v["detector"], paths))
            for v in {v["detector"]["name"]: v for v in variants}.values()}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])   # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def _gt(seq_dir: Path, frame_id: int) -> np.ndarray:
    g = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",")
    g = g[(g[:, 0] == frame_id) & (g[:, 6] == 1) & (g[:, 7] == 1) & (g[:, 8] >= 0.5)]
    return np.column_stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]])


def test_real_botsort_with_reid_on_cpu(paths):
    import cv2
    from pedestrian_catalog.backends import trackers as BT
    config = next(c for c in C.build_catalog(_paper_spec()) if c["tracker"]["family"] == "botsort")
    profile = dict(config["tracker"]["profile"])
    reid = Path(paths["weights_dir"]) / f"{profile['reid_weights']}.pt"
    t = BT.create("botsort", profile, fps=30.0, device="cpu", reid_path=str(reid))
    r = BT.verify_profile(t, "botsort", profile, fps=30.0)
    assert r["cmc_class"] == "ECC" and r["lost_track_frames"] == 30 and r["with_reid"] is True
    assert r["reid_half"] is False
    frame = cv2.imread(str(Path(paths["data_root"]) / FRAMES[0][0] / "img1" / "000001.jpg"))
    dets = np.array([[100, 100, 160, 260, 0.9, 0], [400, 120, 450, 250, 0.8, 0]], np.float32)
    for _ in range(3):
        out = BT.update(t, dets, frame)
    assert out.shape[1] == 8


@pytest.mark.parametrize("seq_id,frame_id", [("track0002", 129), ("MOT20-01", 173)])
def test_dfine_returns_boxes_past_the_border_and_the_cache_accepts_them(paths, variants, models, seq_id, frame_id):
    from pedestrian_catalog.corpus import read_corpus_csv, resolve_sequence
    v = next(x for x in variants if x["detector_variant_id"] == "dfine_l_r1280")
    seq = resolve_sequence(next(r for r in read_corpus_csv(ROOT / "manifests" / "corpus.csv") if r["sequence_id"] == seq_id), paths)
    frame = D._read_frame(seq, frame_id)
    h, w = frame.shape[:2]
    import torch
    dets, info = B.bind(models["dfine_l"], v).detect(frame)
    assert len(dets) and ((dets[:, 0] < 0) | (dets[:, 1] < 0) | (dets[:, 2] > w) | (dets[:, 3] > h)).any()
    assert info["n_before_nms"] is None and info["n_after_nms"] == len(dets)
    m = models["dfine_l"]  # the native output, minus only the person and floor filters
    inputs = m.processor(images=np.ascontiguousarray(frame[:, :, ::-1]), size={"height": 1280, "width": 1280}, return_tensors="pt")
    with torch.inference_mode():
        native = m.processor.post_process_object_detection(m.model(**inputs.to(m.model.device)), threshold=0.0, target_sizes=[(h, w)])[0]
    people = (native["labels"] == 0) & (native["scores"] >= float(v["detection"]["conf_floor"]))
    assert len(dets) == int(people.sum())
    np.testing.assert_allclose(np.sort(dets[:, 4]), np.sort(native["scores"][people].numpy()), rtol=1e-6)
    assert info["saturated"] == (info["candidate_cut_score"] > float(v["detection"]["conf_floor"]))
    D._validate_output(dets, h, w, float(v["detection"]["conf_floor"]), f"{seq_id} frame {frame_id}")


def test_faster_rcnn_with_factory_nms_and_only_the_common_parameters(paths, variants, models):
    import cv2
    v = next(x for x in variants if x["detector_variant_id"] == "frcnn_r50_fpn_v2_r960")
    frame = cv2.imread(str(Path(paths["data_root"]) / FRAMES[0][0] / "img1" / f"{FRAMES[0][1]:06d}.jpg"))
    m = models["frcnn_r50_fpn_v2"]
    dets, info = B.bind(m, v).detect(frame)
    assert m.factory_settings["nms_thresh"] == 0.5 and m.factory_settings["detections_per_img"] == 100
    s = m.settings()
    assert (s["nms_thresh"], s["score_thresh"], s["detections_per_img"], s["min_size"], s["max_size"]) == (0.5, 0.01, 300, [960], 960)
    assert max(info["resized_hw"]) == 960 and info["n_before_nms"] is None and len(dets) > 0


def test_yolo26_in_factory_end2end_mode_without_an_internal_class_filter(paths, variants, models):
    import cv2
    v = next(x for x in variants if x["detector_variant_id"] == "yolo26m_r960")
    frame = cv2.imread(str(Path(paths["data_root"]) / FRAMES[0][0] / "img1" / f"{FRAMES[0][1]:06d}.jpg"))
    m = models["yolo26m"]
    dets, _ = B.bind(m, v).detect(frame)
    assert m.factory_settings["end2end"] is True and m.factory_settings["stride"] == 32
    s = m.settings()
    assert (s["conf"], s["max_det"], s["classes"], s["end2end"]) == (0.01, 300, None, True) and len(dets) > 0


@pytest.mark.parametrize("rel,frame_id", FRAMES)
@pytest.mark.parametrize("vid", _catalog_variant_ids())
def test_real_backend_on_cpu(paths, variants, models, rel, frame_id, vid):
    import cv2
    v = next(x for x in variants if x["detector_variant_id"] == vid)
    seq_dir = Path(paths["data_root"]) / rel
    frame = cv2.imread(str(seq_dir / "img1" / f"{frame_id:06d}.jpg"))
    h, w = frame.shape[:2]
    dets, info = B.bind(models[v["detector"]["name"]], v).detect(frame)
    assert tuple(info["input_hw"]) == D.expected_input_hw(v["detector"]["backend"], v["resolution"]["requested"], h, w)
    dets = D._validate_output(dets, h, w, float(v["detection"]["conf_floor"]), vid)
    label_space = models[v["detector"]["name"]].describe_model()["label_space"]
    raw = D._validate_raw(info["raw"], dets, float(v["detection"]["conf_floor"]), D.person_label({"label_space": label_space}), vid)
    assert len(raw) >= len(dets) and label_space["labels"][str(label_space["person_label"])] == "person"
    gt = _gt(seq_dir, frame_id)
    strong = dets[dets[:, 4] >= 0.5]
    matched = (_iou(gt, strong[:, :4]).max(axis=1) >= 0.5).mean() if len(strong) and len(gt) else 0.0
    # yolo26n at 640 matches 3 of 11 on frame 1 of MOT17-02; boxes left at the input scale would match 0
    assert matched >= 0.25, f"{vid}: only {matched:.0%} of the visible pedestrians match a strong detection"

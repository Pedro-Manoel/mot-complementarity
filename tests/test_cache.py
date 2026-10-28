# -*- coding: utf-8 -*-
"""Persistent detection cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pedestrian_catalog import detection as D

H, W, N = 48, 64, 5
ENV_CPU = {"kind": "cpu", "name": "test cpu"}
ENV_GPU = {"kind": "cuda", "name": "test gpu", "cuda": "0.0", "cudnn": 0}


class Fake:
    """Reads the frame number i from the pixels; the box depends on i and frame 3 is empty."""

    def __init__(self, fail_at_frame=None, offset=0.0, score=None, cls=0.0, input_hw=(384, 640)):
        self.calls: list[int] = []
        self.fail_at_frame, self.offset, self.score, self.cls = fail_at_frame, offset, score, cls
        self.input_hw = input_hw

    def describe(self) -> dict:
        return {"backend": "fake", "device": "cpu", "precision": "fp32", "versions": {"fake": "1"},
                "label_space": {"name": "fake", "person_label": 0, "labels": {"0": "person"}}}

    def detect(self, frame: np.ndarray):
        i = int(frame[0, 0, 0])
        self.calls.append(i)
        if self.fail_at_frame == i:
            raise RuntimeError(f"simulated crash at frame {i}")
        info = {"input_hw": self.input_hw, "resized_hw": None, "n_before_nms": 2, "n_after_nms": 1}
        if i == 3:
            return np.empty((0, 6), np.float32), dict(info, n_before_nms=0, n_after_nms=0, raw=np.empty((0, 6), np.float32))
        sc = 0.5 + 0.01 * i if self.score is None else self.score
        box = [2.0 * i + self.offset, 2.0, 2.0 * i + 10.0 + self.offset, 30.0, sc, self.cls]
        other = [1.0, 1.0, 5.0, 5.0, 0.7, 2.0]  # another class, only in the raw output
        return np.array([box], np.float32), dict(info, raw=np.array([box, other], np.float32))


def _sequence(tmp: Path, corrupted=None) -> dict:
    import cv2
    d = tmp / "data" / "SEQ-01" / "img1"
    d.mkdir(parents=True)
    for i in range(1, N + 1):
        cv2.imwrite(str(d / f"{i:06d}.png"), np.full((H, W, 3), i, np.uint8))
    if corrupted:
        (d / f"{corrupted:06d}.png").write_bytes(b"this is not an image")
    return {"dataset": "SOMPT22", "sequence_id": "SEQ-01", "dir": str(tmp / "data" / "SEQ-01"), "im_dir": "img1",
            "im_ext": ".png", "index_width": 6, "n_frames": N, "width": W, "height": H, "fps": 30.0,
            "frames_manifest_sha256": "f" * 64}


def _variant(**det) -> dict:
    detection = {"conf_floor": 0.01, "max_det": 300, "precision": "fp32"}
    detection.update(det)
    return {"detector_variant_id": "fake_r640", "cache_key": "c" * 16,
            "detector": {"name": "fake", "backend": "ultralytics", "checkpoint": {"file": "f.pt"}},
            "resolution": {"requested": 640, "semantics": "long_side"}, "detection": detection,
            "checkpoint": {"kind": "file", "file": "f.pt", "sha256": "a" * 64}}


def _run(tmp: Path, backend, **extra) -> dict:
    return {"run_dir": tmp / "run", "seed": 42, "backend": backend, "part_size": 2, "environment": ENV_CPU, **extra}


def test_cell_identity_depends_on_what_changes_the_cache_and_not_on_the_tracker():
    seq, v = {"dataset": "A", "sequence_id": "S", "frames_manifest_sha256": "f" * 64}, _variant()
    a = ENV_CPU
    base = D.detection_cell_identity(v, seq, 42, {"fake": "1"}, a)
    assert D.detection_cell_identity(dict(v, tracker={"family": "botsort"}), seq, 42, {"fake": "1"}, a) == base
    others = [
        D.detection_cell_identity(dict(v, cache_key="d" * 16), seq, 42, {"fake": "1"}, a),
        D.detection_cell_identity(dict(v, checkpoint=dict(v["checkpoint"], sha256="b" * 64)), seq, 42, {"fake": "1"}, a),
        D.detection_cell_identity(v, dict(seq, frames_manifest_sha256="e" * 64), 42, {"fake": "1"}, a),
        D.detection_cell_identity(v, seq, 43, {"fake": "1"}, a),
        D.detection_cell_identity(v, seq, 42, {"fake": "2"}, a),
        D.detection_cell_identity(v, seq, 42, {"fake": "1"}, ENV_GPU),
    ]
    assert all(o["cell_id"] != base["cell_id"] for o in others) and len({o["cell_id"] for o in others}) == 6
    with pytest.raises(ValueError, match="execution environment"):
        D.detection_cell_identity(v, seq, 42, {"fake": "1"}, None)


def test_complete_cache_with_an_index_of_every_frame_including_empty_ones(tmp_path):
    m = D.detect_sequence(_sequence(tmp_path), _variant(), _run(tmp_path, Fake()))
    assert m["status"] == "complete" and m["counts"]["n_frames"] == N and m["counts"]["n_empty_frames"] == 1
    idx, dets = D.load_detections(m)
    assert list(idx["frame_id"]) == list(range(1, N + 1))
    assert idx.set_index("frame_id").loc[3, "n_detections"] == 0 and (idx["read_status"] == "ok").all()
    assert 3 not in set(dets["frame_id"]) and len(dets) == N - 1
    assert list(dets.columns) == D.DETECTION_COLUMNS
    assert dets["class_id"].eq(0).all() and (dets["score"] >= 0.01).all()
    assert m["effective_input"]["distinct_input_hw"] == [[384, 640]]
    for name, meta in m["files"].items():
        assert D.sha256_of(Path(m["cell_dir"]) / name) == meta["sha256"]


def test_complete_cell_is_not_recomputed(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    first = D.detect_sequence(seq, v, _run(tmp_path, Fake()))
    b = Fake()
    second = D.detect_sequence(seq, v, _run(tmp_path, b))
    assert b.calls == [] and second["cell_id"] == first["cell_id"] and second["files"] == first["files"]
    states = [json.loads(l)["state"] for l in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert states == ["running", "complete", "skipped"]


def test_interruption_resumes_by_part_and_matches_the_continuous_run(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    with pytest.raises(RuntimeError, match="simulated crash"):
        D.detect_sequence(seq, v, _run(tmp_path, Fake(fail_at_frame=4)))
    cell = D.cell_dir(tmp_path / "run", v, seq, 42)
    assert not (cell / "manifest.json").exists()
    resumed = Fake()
    m = D.detect_sequence(seq, v, _run(tmp_path, resumed))
    assert resumed.calls == [3, 4, 5]
    continuous = D.detect_sequence(seq, v, {"run_dir": tmp_path / "other", "seed": 42, "backend": Fake(), "part_size": 2,
                                            "environment": ENV_CPU})
    a, b = D.load_detections(m), D.load_detections(continuous)
    pd.testing.assert_frame_equal(a[1], b[1])
    clocks = ["elapsed_ms_cache_pass", "elapsed_ms_decode"]
    pd.testing.assert_frame_equal(a[0].drop(columns=clocks), b[0].drop(columns=clocks))
    assert m["files"]["detections.parquet"]["sha256"] == continuous["files"]["detections.parquet"]["sha256"]
    states = [json.loads(l)["state"] for l in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert states == ["running", "failed", "running", "complete"]


def test_tampered_part_is_redone_on_resumption(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    with pytest.raises(RuntimeError):
        D.detect_sequence(seq, v, _run(tmp_path, Fake(fail_at_frame=5)))
    part = D.cell_dir(tmp_path / "run", v, seq, 42) / "parts" / "part_00000.det.parquet"
    part.write_bytes(part.read_bytes()[:-7] + b"junk!!!")
    b = Fake()
    D.detect_sequence(seq, v, _run(tmp_path, b))
    assert b.calls == [1, 2, 5]  # part 0 fails its hash; part 1 (frames 3 and 4) is reused


def test_part_with_a_consistent_hash_but_a_missing_frame_is_refused(tmp_path):
    import pyarrow.parquet as pq
    seq, v = _sequence(tmp_path), _variant()
    with pytest.raises(RuntimeError):
        D.detect_sequence(seq, v, _run(tmp_path, Fake(fail_at_frame=5)))
    parts = D.cell_dir(tmp_path / "run", v, seq, 42) / "parts"
    idx = parts / "part_00000.idx.parquet"
    tbl = pq.read_table(idx)
    pq.write_table(tbl.filter(pq.ParquetFile(idx).read().column("frame_id").to_numpy() != 2), idx)
    meta = json.loads((parts / "part_00000.json").read_text(encoding="utf-8"))
    meta["files"]["idx"] = D.sha256_of(idx)
    (parts / "part_00000.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(D.InvalidCache, match="does not cover exactly"):
        D.detect_sequence(seq, v, _run(tmp_path, Fake()))


def test_count_before_nms_that_is_not_measured_stays_null_not_zero(tmp_path):
    class NoCount(Fake):
        def detect(self, frame):
            dets, info = super().detect(frame)
            return dets, dict(info, n_before_nms=None)
    m = D.detect_sequence(_sequence(tmp_path), _variant(), _run(tmp_path, NoCount()))
    idx, _ = D.load_detections(m)
    assert idx["n_before_nms"].isna().all()
    assert m["counts"]["n_suppressed_by_nms"] is None


def test_manifest_with_a_tampered_file_is_refused(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    m = D.detect_sequence(seq, v, _run(tmp_path, Fake()))
    target = Path(m["cell_dir"]) / "detections.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(D.InvalidCache, match="detections.parquet"):
        D.detect_sequence(seq, v, _run(tmp_path, Fake()))
    with pytest.raises(D.InvalidCache):
        D.load_detections(m)


def test_same_directory_with_another_identity_is_refused(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    m = D.detect_sequence(seq, v, _run(tmp_path, Fake()))
    mf = Path(m["cell_dir"]) / "manifest.json"
    data = json.loads(mf.read_text(encoding="utf-8"))
    data["identity"]["checkpoint_sha256"] = "b" * 64
    mf.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(D.InvalidCache, match="another identity"):
        D.detect_sequence(seq, v, _run(tmp_path, Fake()))


@pytest.mark.parametrize("backend,fragment", [
    (Fake(offset=200.0), "outside the original image"),
    (Fake(score=0.001), "below the floor"),
    (Fake(cls=2.0), "is not person"),
])
def test_invalid_backend_outputs_fail_the_cell(tmp_path, backend, fragment):
    seq, v = _sequence(tmp_path), _variant()
    with pytest.raises(D.InvalidOutput, match=fragment):
        D.detect_sequence(seq, v, _run(tmp_path, backend))
    assert not (D.cell_dir(tmp_path / "run", v, seq, 42) / "manifest.json").exists()


def test_unreadable_frame_fails_naming_the_frame(tmp_path):
    seq, v = _sequence(tmp_path, corrupted=2), _variant()
    with pytest.raises(D.UnreadableFrame, match="000002.png"):
        D.detect_sequence(seq, v, _run(tmp_path, Fake()))


def test_same_effective_input_at_two_levels_of_the_same_model_is_refused():
    def m(det, r, hw):
        return {"detector_variant_id": f"{det}_r{r}", "detector_name": det, "resolution_requested": r,
                "sequence_key": "SOMPT22/SEQ-01", "effective_input": {"distinct_input_hw": [hw]}}
    ok = [m("yolo", 640, [384, 640]), m("yolo", 800, [480, 800]), m("dfine", 640, [640, 640])]
    D.validate_effective_inputs(ok)
    with pytest.raises(D.InvalidOutput, match="yolo"):
        D.validate_effective_inputs(ok + [m("yolo", 1280, [480, 800])])


@pytest.mark.parametrize("backend,requested,hw,expected", [
    ("ultralytics", 640, (1080, 1920), (384, 640)), ("ultralytics", 800, (1080, 1920), (480, 800)),
    ("ultralytics", 1280, (1080, 1920), (736, 1280)), ("ultralytics", 640, (480, 640), (480, 640)),
    ("dfine", 1280, (1080, 1920), (1280, 1280)), ("dfine", 640, (480, 640), (640, 640)),
    ("torchvision", 1280, (1080, 1920), (736, 1280)), ("torchvision", 640, (480, 640), (480, 640)),
    ("torchvision", 800, (1440, 2560), (480, 800)),
])
def test_expected_input_geometry(backend, requested, hw, expected):
    assert D.expected_input_hw(backend, requested, *hw) == expected


def test_resumption_in_another_environment_reuses_neither_parts_nor_cell(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    with pytest.raises(RuntimeError, match="simulated crash"):
        D.detect_sequence(seq, v, _run(tmp_path, Fake(fail_at_frame=4)))
    gpu = Fake()
    m = D.detect_sequence(seq, v, _run(tmp_path, gpu, environment=ENV_GPU))
    assert gpu.calls == [1, 2, 3, 4, 5] and m["identity"]["environment"] == ENV_GPU
    with pytest.raises(D.InvalidCache, match="another identity"):
        D.detect_sequence(seq, v, _run(tmp_path, Fake()))  # the CPU run does not reuse the GPU cell


def test_cell_stores_the_whole_model_output_and_the_person_cache_is_its_subset(tmp_path):
    m = D.detect_sequence(_sequence(tmp_path), _variant(), _run(tmp_path, Fake()))
    cell = Path(m["cell_dir"])
    raw = pd.read_parquet(cell / "raw_detections.parquet")
    idx, dets = D.load_detections(m)
    assert list(raw.columns) == D.RAW_COLUMNS and len(raw) == 2 * (N - 1)
    assert sorted(raw["label"].unique().tolist()) == [0, 2]
    people = raw[(raw["label"] == 0) & (raw["score"] >= 0.01)].reset_index(drop=True)
    np.testing.assert_array_equal(people[["x1", "y1", "x2", "y2", "score"]].to_numpy(),
                                  dets[["x1", "y1", "x2", "y2", "score"]].to_numpy())
    assert idx["n_raw_detections"].tolist() == [2, 2, 0, 2, 2] and (idx["elapsed_ms_decode"] >= 0).all()
    assert m["counts"]["n_raw_detections"] == 8 and m["counts"]["max_raw_detections_per_frame"] == 2
    assert m["raw_output"]["label_space"]["person_label"] == 0
    assert {"raw_detections.parquet", "resources.parquet"} <= set(m["files"])


class _NoRaw(Fake):
    def detect(self, frame):
        dets, info = super().detect(frame)
        info.pop("raw")
        return dets, info


class _InconsistentRaw(Fake):
    def detect(self, frame):
        dets, info = super().detect(frame)
        return dets, dict(info, raw=np.empty((0, 6), np.float32))


class _NoLabels(Fake):
    def describe(self):
        d = super().describe()
        d.pop("label_space")
        return d


@pytest.mark.parametrize("backend,fragment", [(_NoRaw(), "did not return the raw model output"),
                                             (_InconsistentRaw(), "are not the"), (_NoLabels(), "label space")])
def test_missing_or_inconsistent_raw_output_fails_the_cell(tmp_path, backend, fragment):
    with pytest.raises(D.InvalidOutput, match=fragment):
        D.detect_sequence(_sequence(tmp_path), _variant(), _run(tmp_path, backend))


def test_resources_per_part_are_summed_in_the_cell_and_the_samples_written(tmp_path):
    m = D.detect_sequence(_sequence(tmp_path), _variant(), _run(tmp_path, Fake(), resources={"device": "cpu", "sample_hz": 0}))
    r = m["resources"]
    assert r["n_parts"] == 3 and r["cpu_process_time_s"] is not None and r["duration_s"] > 0  # 5 frames in parts of 2
    assert r["cpu_process_ms_per_frame"] == pytest.approx(1000.0 * r["cpu_process_time_s"] / N)
    assert r["gpu_energy_j"] is None and "TDP" in r["cpu_energy_via"]
    samples = pd.read_parquet(Path(m["cell_dir"]) / "resources.parquet")
    assert sorted(samples["part"].unique().tolist()) == [0, 1, 2] and "gpu_power_w" in samples.columns
    assert m["timing"]["decode_ms_mean"] >= 0 and m["timing"]["detector_ms_mean"] >= 0


def test_resumption_sums_the_resources_of_the_parts_already_done(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    with pytest.raises(RuntimeError):
        D.detect_sequence(seq, v, _run(tmp_path, Fake(fail_at_frame=4), resources={"device": "cpu", "sample_hz": 0}))
    m = D.detect_sequence(seq, v, _run(tmp_path, Fake(), resources={"device": "cpu", "sample_hz": 0}))
    assert m["resources"]["n_parts"] == m["resources"]["n_parts_measured"] == 3

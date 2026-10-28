# -*- coding: utf-8 -*-
"""Tracking over the detection cache, with a fake tracker and with the real BoxMOT on CPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pedestrian_catalog import detection as D
from pedestrian_catalog import tracking as TR
from pedestrian_catalog.backends import trackers as BT

CPU_ENV = {"kind": "cpu", "name": "test cpu"}
GPU_ENV = {"kind": "cuda", "name": "test gpu", "cuda": "0.0", "cudnn": 0}

H, W, N = 48, 64, 5


class _Det:
    """Reads the frame number from the pixels; one box moving 2 px per frame, none on frame 3."""

    def describe(self):
        return {"backend": "fake", "device": "cpu", "precision": "fp32", "versions": {"fake": "1"}, "label_space": {"name": "fake", "person_label": 0, "labels": {"0": "person"}}}

    def detect(self, frame):
        i = int(frame[0, 0, 0])
        if i == 3:
            return np.empty((0, 6), np.float32), {"input_hw": (384, 640), "n_before_nms": 0, "n_after_nms": 0,
                                                  "raw": np.empty((0, 6), np.float32)}
        d = np.array([[4.0 + 2 * i, 4.0, 24.0 + 2 * i, 40.0, 0.9, 0]], np.float32)
        return d, {"input_hw": (384, 640), "n_before_nms": 1, "n_after_nms": 1, "raw": d.copy()}


def _sequence(tmp: Path, name="SEQ-01") -> dict:
    import cv2
    d = tmp / "data" / name / "img1"
    d.mkdir(parents=True)
    for i in range(1, N + 1):
        cv2.imwrite(str(d / f"{i:06d}.png"), np.full((H, W, 3), i, np.uint8))
    return {"dataset": "SOMPT22", "sequence_id": name, "dir": str(tmp / "data" / name), "im_dir": "img1", "im_ext": ".png",
            "index_width": 6, "n_frames": N, "width": W, "height": H, "fps": 25.0, "frames_manifest_sha256": name * 8}


def _variant():
    return {"detector_variant_id": "fake_r640", "cache_key": "c" * 16,
            "detector": {"name": "fake", "backend": "ultralytics", "checkpoint": {"file": "f.pt"}},
            "resolution": {"requested": 640, "semantics": "long_side"},
            "detection": {"conf_floor": 0.01, "max_det": 300, "precision": "fp32"},
            "checkpoint": {"kind": "file", "file": "f.pt", "sha256": "a" * 64}}


def _profile(family: str) -> dict:
    from test_catalog import _common, _profiles
    return {**_profiles()[family], **_common()}


def _config(family="bytetrack", **profile):
    p = dict(_profile(family), **profile)
    ph = str(abs(hash(tuple(sorted((k, str(v)) for k, v in p.items())))))[:8]
    return {"config_id": f"fake_r640__{family}", "config_hash": (family + "_" * 8)[:8] + ph.rjust(8, "0"),
            "cache_key": "c" * 16, "detector_variant_id": "fake_r640", "detection": _variant()["detection"],
            "tracker": {"name": family, "family": family, "profile": p, "profile_hash": ph}}


def _cache(tmp: Path, seq=None, run="run") -> dict:
    seq = seq or _sequence(tmp)
    return D.detect_sequence(seq, _variant(), {"run_dir": tmp / run, "seed": 42, "backend": _Det(), "part_size": 2,
                                               "environment": CPU_ENV})


class _FakeTracker:
    """One track per detection, always with raw identity 7."""

    def __init__(self, log: list, fail_at_frame=None):
        self.log, self.fail_at_frame, self.frame_count = log, fail_at_frame, 0

    def update(self, dets, frame):
        self.frame_count += 1
        self.log.append((self.frame_count, int(frame[0, 0, 0]), frame.shape, len(dets)))
        if self.fail_at_frame == self.frame_count:
            raise RuntimeError(f"simulated crash at frame {self.frame_count}")
        return np.array([[d[0], d[1], d[2], d[3], 7, d[4], d[5], j] for j, d in enumerate(dets)], np.float32).reshape(-1, 8)


def _run(tmp, log, fail=None, instances=None, **extra):
    def make(config, fps):
        if instances is not None:
            instances.append((config["config_id"], fps))
        return _FakeTracker(log, fail)
    return {"run_dir": tmp / "run", "factory": make, "versions": {"fake": "1"}, "environment": CPU_ENV, **extra}


def test_test_configs_have_distinct_hashes():
    assert len({_config(f)["config_hash"] for f in ("bytetrack", "ocsort", "botsort")}) == 3
    assert _config("bytetrack")["config_hash"] != _config("bytetrack", track_buffer=60)["config_hash"]


def test_updates_every_frame_including_frames_without_detections(tmp_path):
    log = []
    m = TR.track_sequence(_sequence(tmp_path, "A"), _config(), _cache(tmp_path, _sequence(tmp_path)), _run(tmp_path, log))
    assert [r[0] for r in log] == list(range(1, N + 1)) and [r[3] for r in log] == [1, 1, 0, 1, 1]
    tracks, frames = TR.load_tracks(m)
    assert list(frames["frame_id"]) == list(range(1, N + 1)) and frames.set_index("frame_id").loc[3, "n_tracks"] == 0
    assert list(tracks.columns) == TR.TRACK_COLUMNS


def test_bytetrack_gets_an_empty_frame_of_the_right_size_and_botsort_the_pixels(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    log = []
    TR.track_sequence(seq, _config("bytetrack"), cache, _run(tmp_path, log))
    assert all(r[1] == 0 and r[2] == (H, W, 3) for r in log)
    log.clear()
    TR.track_sequence(seq, _config("botsort"), cache, _run(tmp_path, log, reid_sha256="r" * 64, environment=GPU_ENV))
    assert [r[1] for r in log] == list(range(1, N + 1))


def test_identities_are_renumbered_by_order_of_appearance_and_raw_ones_are_kept(tmp_path):
    seq = _sequence(tmp_path)
    m = TR.track_sequence(seq, _config(), _cache(tmp_path, seq), _run(tmp_path, []))
    tracks, _ = TR.load_tracks(m)
    assert set(tracks["track_id"]) == {1} and set(tracks["raw_track_id"]) == {7}
    lines = (Path(m["cell_dir"]) / "tracks_mot.txt").read_text(encoding="utf-8").splitlines()
    first = lines[0].split(",")
    assert len(lines) == N - 1 and first[:2] == ["1", "1"] and first[7:] == ["-1", "-1", "-1"]
    assert float(first[4]) == pytest.approx(20.0) and float(first[5]) == pytest.approx(36.0)  # w = x2 - x1, h = y2 - y1


def test_fresh_state_on_each_sequence(tmp_path):
    inst = []
    for name in ("A", "B"):
        seq = _sequence(tmp_path, name)
        TR.track_sequence(seq, _config(), _cache(tmp_path, seq), _run(tmp_path, [], instances=inst))
    assert inst == [("fake_r640__bytetrack", 25.0), ("fake_r640__bytetrack", 25.0)]


def test_failure_restarts_from_the_first_frame_and_equals_the_continuous_run(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    log = []
    with pytest.raises(RuntimeError, match="simulated crash"):
        TR.track_sequence(seq, _config(), cache, _run(tmp_path, log, fail=4))
    assert not (TR.cell_dir(tmp_path / "run", _config(), cache) / "manifest.json").exists()
    log.clear()
    m = TR.track_sequence(seq, _config(), cache, _run(tmp_path, log))
    assert [r[0] for r in log] == list(range(1, N + 1))
    other = TR.track_sequence(seq, _config(), cache, {**_run(tmp_path, []), "run_dir": tmp_path / "other"})
    pd.testing.assert_frame_equal(TR.load_tracks(m)[0], TR.load_tracks(other)[0])
    states = [json.loads(l)["state"] for l in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
              if json.loads(l)["stage"] == "track"]
    assert states == ["running", "failed", "running", "complete"]


def test_complete_cell_is_not_redone(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    TR.track_sequence(seq, _config(), cache, _run(tmp_path, []))
    log = []
    TR.track_sequence(seq, _config(), cache, _run(tmp_path, log))
    assert log == []


def test_same_directory_with_another_identity_is_refused(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    TR.track_sequence(seq, _config(), cache, _run(tmp_path, []))
    with pytest.raises(D.InvalidCache, match="different identity"):
        TR.track_sequence(seq, _config(), cache, {**_run(tmp_path, []), "versions": {"fake": "2"}})


def test_tampered_cache_is_refused(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    target = Path(cache["cell_dir"]) / "detections.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(D.InvalidCache):
        TR.track_sequence(seq, _config(), cache, _run(tmp_path, []))


def test_tracker_threshold_below_the_cache_floor_is_refused(tmp_path):
    seq = _sequence(tmp_path)
    with pytest.raises(ValueError, match="floor"):
        TR.track_sequence(seq, _config(min_conf=0.001), _cache(tmp_path, seq), _run(tmp_path, []))


def test_cell_identity(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    v, cpu, gpu = {"boxmot": "17.0.0"}, CPU_ENV, GPU_ENV
    base = TR.tracking_cell_identity(_config(), cache, 25.0, v, environment=cpu)
    assert TR.tracking_cell_identity(_config(track_buffer=60), cache, 25.0, v, environment=cpu)["cell_id"] != base["cell_id"]
    assert TR.tracking_cell_identity(_config(), dict(cache, cell_id="x" * 16), 25.0, v, environment=cpu)["cell_id"] != base["cell_id"]
    assert TR.tracking_cell_identity(_config(), cache, 30.0, v, environment=cpu)["cell_id"] != base["cell_id"]
    assert TR.tracking_cell_identity(_config(), cache, 25.0, {"boxmot": "18"}, environment=cpu)["cell_id"] != base["cell_id"]
    # the timings depend on the machine, so the environment counts for the CPU trackers too
    assert TR.tracking_cell_identity(_config(), cache, 25.0, v, environment=dict(cpu, name="other"))["cell_id"] != base["cell_id"]
    b = TR.tracking_cell_identity(_config("botsort"), cache, 25.0, v, reid_sha256="r" * 64, environment=gpu)
    assert TR.tracking_cell_identity(_config("botsort"), cache, 25.0, v, reid_sha256="s" * 64,
                                     environment=gpu)["cell_id"] != b["cell_id"]
    assert TR.tracking_cell_identity(_config("botsort"), cache, 25.0, v, reid_sha256="r" * 64,
                                     environment=cpu)["cell_id"] != b["cell_id"]
    with pytest.raises(ValueError, match="execution environment"):
        TR.tracking_cell_identity(_config(), cache, 25.0, v)
    with pytest.raises(ValueError, match="ReID"):
        TR.tracking_cell_identity(_config("botsort"), cache, 25.0, {"boxmot": "17.0.0"}, environment=CPU_ENV)


def _renumber(outputs: list[np.ndarray]) -> list[np.ndarray]:
    """Renumbers the raw identities by order of appearance, as `track_sequence` does."""
    new_id_of: dict[int, int] = {}
    out = []
    for s in outputs:
        s = s.copy()
        for row in s:
            row[4] = new_id_of.setdefault(int(row[4]), len(new_id_of) + 1)
        out.append(s)
    return out


@pytest.mark.parametrize("family", ["bytetrack", "ocsort", "botsort"])
def test_profile_resolved_in_the_installed_version_is_verified(family):
    # BoT-SORT without ReID here; with the real ReID it is covered by the weights tests
    p = dict(_profile(family), with_reid=False) if family == "botsort" else _profile(family)
    t = BT.create(family, p, fps=25.0, device="cpu", reid_path="no_reid.pt")
    r = BT.verify_profile(t, family, p, fps=25.0)
    assert r["lost_track_frames"] == 25
    with pytest.raises(ValueError, match="max_time_lost|max_age"):
        BT.verify_profile(t, family, dict(p, lost_track_seconds=2.0), fps=25.0)
    if family == "botsort":
        assert r["cmc_class"] == "ECC"
        with pytest.raises(ValueError, match="cmc"):
            BT.verify_profile(t, family, dict(p, cmc_method=None), fps=25.0)
    field = {"bytetrack": "match_thresh", "ocsort": "inertia", "botsort": "match_thresh"}[family]
    with pytest.raises(ValueError, match=field):
        BT.verify_profile(t, family, dict(p, **{field: p[field] + 1}), fps=25.0)


@pytest.mark.parametrize("family", ["bytetrack", "ocsort"])
def test_bytetrack_and_ocsort_do_not_read_the_pixels(family):
    rng = np.random.default_rng(0)
    dets = [np.array([[10 + 3 * i, 10, 60 + 3 * i, 120, 0.9, 0], [200, 50 + i, 240, 150 + i, 0.8, 0]], np.float32) for i in range(12)]
    outputs = []
    for mode in ("real", "empty"):
        t = BT.create(family, _profile(family), fps=25.0, device="cpu", reid_path=None)
        seq_out = []
        for d in dets:
            frame = (rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) if mode == "real" else np.zeros((480, 640, 3), np.uint8))
            seq_out.append(np.asarray(BT.update(t, d, frame)))
        outputs.append(_renumber(seq_out))
    for a, b in zip(*outputs):
        np.testing.assert_array_equal(a, b)
    assert not BT.needs_image(family) and BT.needs_image("botsort")


def test_boxmot_has_a_global_per_process_identity_counter():
    d = np.array([[10, 10, 60, 120, 0.9, 0]], np.float32)
    raw_ids = []
    for _ in range(2):
        t = BT.create("bytetrack", _profile("bytetrack"), fps=30.0, device="cpu", reid_path=None)
        out = []
        for _ in range(3):
            out = BT.update(t, d, np.zeros((480, 640, 3), np.uint8))
        raw_ids.append(int(out[0, 4]))
    assert raw_ids[1] > raw_ids[0]


def test_real_bytetrack_keeps_the_identity_across_parts_and_between_calls(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)  # parts of 2 frames
    run = {"run_dir": tmp_path / "run", "factory": BT.factory(device="cpu", reid_path=None), "versions": BT.tracker_versions(),
           "environment": CPU_ENV}
    m1 = TR.track_sequence(seq, _config("bytetrack"), cache, run)
    m2 = TR.track_sequence(seq, _config("bytetrack"), cache, {**run, "run_dir": tmp_path / "other"})
    t1, t2 = TR.load_tracks(m1)[0], TR.load_tracks(m2)[0]
    assert set(t1["track_id"]) == {1} and set(t1["frame_id"]) >= {1, 2, 4, 5}
    pd.testing.assert_frame_equal(t1.drop(columns=["raw_track_id"]), t2.drop(columns=["raw_track_id"]))


def test_everything_goes_through_the_constructor_and_the_derived_field_is_read_back():
    p = dict(_profile("ocsort"), iou_threshold=0.5)
    t = BT.create("ocsort", p, fps=25.0, device="cpu", reid_path=None)
    assert t.asso_threshold == 0.5 and BT.verify_profile(t, "ocsort", p, fps=25.0)["asso_threshold"] == 0.5
    with pytest.raises(ValueError, match="asso_threshold"):
        BT.create("ocsort", dict(_profile("ocsort"), asso_threshold=0.5), fps=25.0, device="cpu", reid_path=None)
    t.asso_threshold = 0.4
    with pytest.raises(ValueError, match="asso_threshold"):
        BT.verify_profile(t, "ocsort", p, fps=25.0)


def test_bytetrack_uses_track_thresh_as_det_thresh_and_the_verification_reads_it():
    t = BT.create("bytetrack", _profile("bytetrack"), fps=25.0, device="cpu", reid_path=None)
    assert BT.verify_profile(t, "bytetrack", _profile("bytetrack"), fps=25.0)["det_thresh"] == 0.45
    t.det_thresh = 0.3
    with pytest.raises(ValueError, match="det_thresh"):
        BT.verify_profile(t, "bytetrack", _profile("bytetrack"), fps=25.0)


@pytest.mark.parametrize("family,field", [("bytetrack", "max_age"), ("ocsort", "max_obs"), ("botsort", "det_thresh")])
def test_field_without_effect_does_not_reach_the_constructor(family, field):
    with pytest.raises(ValueError, match=field):
        BT.create(family, dict(_profile(family), **{field: 1}), fps=25.0, device="cpu", reid_path="no_reid.pt")


def test_buffer_depends_on_the_sequence_fps():
    t = BT.create("bytetrack", _profile("bytetrack"), fps=25.0, device="cpu", reid_path=None)
    assert BT.verify_profile(t, "bytetrack", _profile("bytetrack"), fps=25.0)["lost_track_frames"] == 25
    with pytest.raises(ValueError, match="max_time_lost"):
        BT.verify_profile(t, "bytetrack", _profile("bytetrack"), fps=30.0)


def test_cache_with_consistent_hash_but_a_missing_frame_is_refused(tmp_path):
    import pyarrow.parquet as pq
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    idx = Path(cache["cell_dir"]) / "frames_index.parquet"
    tbl = pq.read_table(idx)
    pq.write_table(tbl.filter(tbl.column("frame_id").to_numpy() != 2), idx)
    cache = dict(cache, files=dict(cache["files"], **{"frames_index.parquet": {"sha256": D.sha256_of(idx), "size": idx.stat().st_size}}))
    with pytest.raises(D.InvalidCache, match="does not cover"):
        TR.track_sequence(seq, _config(), cache, _run(tmp_path, []))


def test_manifest_keeps_the_profile_and_the_resolved_values(tmp_path):
    seq = _sequence(tmp_path)
    cache = _cache(tmp_path, seq)
    run = {"run_dir": tmp_path / "run", "factory": BT.factory(device="cpu", reid_path=None), "versions": BT.tracker_versions(),
           "environment": CPU_ENV}
    m = TR.track_sequence(seq, _config("ocsort"), cache, run)
    assert m["tracker_profile"]["delta_t"] == 3 and m["tracker_resolved"]["frame_rate_used"] is None
    assert m["tracker_resolved"]["asso_threshold"] == 0.3
    mb = TR.track_sequence(seq, _config("bytetrack"), cache, run)
    assert mb["tracker_resolved"]["frame_rate_used"] == 25 and mb["tracker_resolved"]["det_thresh"] == 0.45


def test_tracking_cell_keeps_the_resources_of_the_loop_and_the_mean_timings(tmp_path):
    seq = _sequence(tmp_path, "A")
    m = TR.track_sequence(seq, _config(), _cache(tmp_path, seq),
                          _run(tmp_path, [], resources={"device": "cpu", "sample_hz": 0}))
    r = m["resources"]
    assert r["cpu_process_time_s"] is not None and r["duration_s"] > 0 and r["gpu_energy_j"] is None
    assert r["cpu_process_ms_per_frame"] == pytest.approx(1000.0 * r["cpu_process_time_s"] / N)
    assert "resources.parquet" in m["files"] and (Path(m["cell_dir"]) / "resources.parquet").is_file()
    assert m["timing"]["update_ms_mean"] >= 0 and m["timing"]["read_ms_mean"] >= 0


def test_boxmot_does_not_print_a_line_per_tracker_created():
    import os
    import subprocess
    import sys
    # in a subprocess: BoxMOT logs from a thread that is only flushed when the process ends
    code = ("import json, sys\n"
            "from pedestrian_catalog.backends import trackers as BT\n"
            "profile = json.loads(sys.argv[1])\n"
            "for _ in range(2):\n"
            "    BT.create('bytetrack', profile, fps=30.0, device='cpu', reid_path=None)\n"
            "print('created')\n")
    root = str(Path(__file__).resolve().parents[1])
    r = subprocess.run([sys.executable, "-c", code, json.dumps(_profile("bytetrack"))], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": root}, encoding="utf-8", errors="replace")
    assert r.returncode == 0 and "created" in r.stdout, r.stderr[-500:]
    assert "SUCCESS" not in r.stderr and "SUCCESS" not in r.stdout

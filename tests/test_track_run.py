# -*- coding: utf-8 -*-
"""The track command, on the world of the detect tests with a fake tracker factory."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from pedestrian_catalog import cli
from pedestrian_catalog.backends import trackers as BT
from test_detect_run import _Model, _View, world  # noqa: F401


class _FakeTracker:
    def __init__(self, counts):
        self.counts = counts

    def update(self, dets, frame):
        self.counts["updates"] += 1
        return np.array([[d[0], d[1], d[2], d[3], 5, d[4], d[5], j] for j, d in enumerate(dets)], np.float32).reshape(-1, 8)


@pytest.fixture
def tracked(world, monkeypatch):
    counts = {"created": 0, "updates": 0, "devices": []}

    def factory_of(device, reid_path):
        counts["devices"].append(device)
        assert reid_path and reid_path.endswith("osnet_x0_25_msmt17.pt")

        def make(config, fps):
            counts["created"] += 1
            return _FakeTracker(counts)
        return make
    monkeypatch.setattr(BT, "factory", factory_of)
    assert cli.main(["detect", "--run-id", "t", "--experiment", str(world["exp"]), "--device", "cpu"]) == 0
    return {**world, "counts": counts}


def _track(m, *extra) -> int:
    return cli.main(["track", "--run-id", "t", "--experiment", str(m["exp"]), *extra])


def _report(m) -> dict:
    return json.loads((m["runs"] / "t" / "reports" / "tracking_coverage.json").read_text(encoding="utf-8"))


def test_track_requires_device(tracked):
    with pytest.raises(SystemExit):
        cli.main(["track", "--run-id", "t", "--experiment", str(tracked["exp"])])


def test_track_completes_and_then_does_nothing(tracked):
    assert _track(tracked, "--device", "cpu") == 0
    rep = _report(tracked)
    assert rep["expected"] == 27 * 2 * 2 and len(rep["complete"]) == 108 and rep["failed"] == [] and rep["missing_detection"] == []
    assert tracked["counts"]["created"] == 108 and tracked["counts"]["updates"] == 108 * 4
    tracked["counts"]["created"] = 0
    assert _track(tracked, "--device", "cpu") == 0
    assert tracked["counts"]["created"] == 0 and len(_report(tracked)["complete"]) == 108


def test_track_filters_configurations(tracked):
    assert _track(tracked, "--device", "cpu", "--configs", "yolo26m_r640__bytetrack,dfine_l_r1280__botsort", "--seeds", "42") == 0
    rep = _report(tracked)
    assert rep["expected"] == 2 * 2 * 1 and len(rep["complete"]) == 4


def test_missing_detection_cache_is_left_waiting_and_does_not_fail_the_rest(tracked):
    det = tracked["runs"] / "t" / "detections"
    for cell in det.glob("*/SOMPT22__SOMPT22-B/seed43"):
        shutil.rmtree(cell)
    assert _track(tracked, "--device", "cpu") == 1
    rep = _report(tracked)
    assert len(rep["missing_detection"]) == 27 and len(rep["complete"]) == 108 - 27
    assert all(x["sequence"] == "SOMPT22/SOMPT22-B" and x["seed"] == 43 for x in rep["missing_detection"])


def test_track_with_cuda_uses_the_lock_and_passes_the_device(tracked):
    from pedestrian_catalog import runtime as R
    assert _track(tracked, "--device", "cuda:0", "--configs", "yolo26m_r640__botsort", "--seeds", "42") == 0
    assert tracked["counts"]["devices"] == ["cuda:0"]
    assert not (tracked["runs"] / R.GPU_LOCK_NAME).exists()
    rep = _report(tracked)
    assert rep["device"] == "cuda:0" and len(rep["complete"]) == 2


@pytest.mark.parametrize("command", ["track"])
def test_commands_that_load_models_refuse_without_the_seal(tracked, capsys, command):
    (tracked["cfg"].parent / "manifests" / "option_check.json").unlink()
    base = ["--run-id", "t", "--experiment", str(tracked["exp"]), "--device", "cpu"]
    before = (tracked["counts"]["created"], len(tracked["loaded"]))
    assert cli.main([command, *base]) == 2
    assert "check-options" in capsys.readouterr().err
    assert (tracked["counts"]["created"], len(tracked["loaded"])) == before


def test_track_with_cuda_refuses_a_busy_gpu(tracked, capsys):
    import os
    from pedestrian_catalog import runtime as R
    lock = tracked["runs"] / R.GPU_LOCK_NAME
    lock.write_text(json.dumps({"pid": os.getpid(), "command": "other"}), encoding="utf-8")
    assert _track(tracked, "--device", "cuda:0") == 2
    assert "GPU busy" in capsys.readouterr().err
    lock.unlink()


def test_track_refuses_to_resume_the_run_in_another_environment(tracked, monkeypatch, capsys):
    from pedestrian_catalog import runtime as R
    assert _track(tracked, "--device", "cpu", "--seeds", "42") == 0
    real = R.execution_environment
    monkeypatch.setattr(R, "execution_environment", lambda device: {**real(device), "name": "other machine"})
    assert _track(tracked, "--device", "cpu", "--seeds", "42") == 2
    assert "mixing environments" in capsys.readouterr().err


def test_identity_environment_follows_where_each_family_runs(tracked):
    assert _track(tracked, "--device", "cuda:0", "--configs", "yolo26m_r640__botsort,yolo26m_r640__bytetrack",
                  "--seeds", "42") == 0
    ms = [json.loads(p.read_text(encoding="utf-8")) for p in (tracked["runs"] / "t" / "tracks").glob("*/*/seed42/manifest.json")]
    kinds = {(m["tracker_family"], m["identity"]["environment"]["kind"]) for m in ms}
    assert kinds == {("botsort", "cuda"), ("bytetrack", "cpu")}


def test_track_shows_progress_per_cell_and_only_of_what_is_left(tracked, capsys):
    assert _track(tracked, "--device", "cpu") == 0
    out = capsys.readouterr().out
    assert "[track] 108 cells to do (0 already complete), 432 frames" in out
    assert "[track] 108/108 " in out and "remaining ~" in out and "update " in out
    assert _track(tracked, "--device", "cpu") == 0
    out = capsys.readouterr().out
    assert "[track] 0 cells to do (108 already complete)" in out and "[track] 1/" not in out

# -*- coding: utf-8 -*-
"""The detect and run commands, with `load_model` and `bind` replaced by fakes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from pedestrian_catalog import cli
from pedestrian_catalog import corpus as K
from pedestrian_catalog import runtime as R
from pedestrian_catalog.backends import detectors as DB

import _test_configs

ROOT = Path(__file__).resolve().parents[1]


def _png_seq(base: Path, rel: str, offset: int, n: int = 4) -> Path:
    import cv2
    d = base / rel
    (d / "img1").mkdir(parents=True)
    (d / "gt").mkdir()
    (d / "seqinfo.ini").write_text(f"[Sequence]\nname = {d.name}\nimDir = img1\nframeRate = 25\nseqLength = {n}\n"
                                   "imWidth = 64\nimHeight = 48\nimExt = .png\n", encoding="utf-8")
    for i in range(1, n + 1):
        cv2.imwrite(str(d / "img1" / f"{i:06d}.png"), np.full((48, 64, 3), i + offset, np.uint8))
    (d / "gt" / "gt.txt").write_text("\n".join(f"{i},1,1,1,8,8,1,1,1" for i in range(1, n + 1)) + "\n", encoding="utf-8")
    return d


class _Model:
    def __init__(self, name):
        self.name = name


class _View:
    """The box follows the pixel value; the input size follows the resolution unless `collision`."""

    def __init__(self, model, variant, collision=False):
        self.model, self.v, self.collision = model, variant, collision

    def describe(self):
        return {"backend": "fake", "device": "cpu", "precision": "fp32", "versions": {"fake": "1"}, "label_space": {"name": "fake", "person_label": 0, "labels": {"0": "person"}}}

    def detect(self, frame):
        val, r = int(frame[0, 0, 0]), int(self.v["resolution"]["requested"])
        hw = (320, 320) if self.collision else (r // 2, r)
        d = np.array([[val, 2, val + 10, 30, 0.9, 0]], np.float32)
        return d, {"input_hw": hw, "n_before_nms": 1, "n_after_nms": 1, "raw": d.copy()}


def _seal(tmp_path: Path, problems=(), **overrides) -> Path:
    from pedestrian_catalog import options_check as OC
    from pedestrian_catalog.backends import trackers as BT
    res = json.loads((tmp_path / "manifests" / "catalog.resolved.json").read_text(encoding="utf-8"))
    reg = OC.seal_record(res, OC.current_versions(res, DB.backend_versions, BT.tracker_versions), list(problems),
                         {"note": "test seal"}, "cpu")
    return OC.write_seal(tmp_path / "manifests" / "option_check.json", {**reg, **overrides})


@pytest.fixture
def world(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _png_seq(data, "MOT20/train/MOT20-A", 0)
    _png_seq(data, "SOMPT22/train/SOMPT22-B", 10)
    (data / "MOT17" / "train").mkdir(parents=True)
    (data / "CrowdTrack-person").mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "yolo26m.pt").write_bytes(b"yolo-weights")
    (weights / "osnet_x0_25_msmt17.pt").write_bytes(b"reid-weights")
    base_hf = tmp_path / "hf" / "models--ustc-community--dfine-large-coco"
    (base_hf / "snapshots" / "abc123").mkdir(parents=True)
    (base_hf / "snapshots" / "abc123" / "config.json").write_bytes(b'{"num_queries": 300}')
    hub = tmp_path / "torchhub" / "checkpoints"
    hub.mkdir(parents=True)
    (hub / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth").write_bytes(b"frcnn-weights")
    paths = {"data_root": str(data),
             "datasets": {"MOT17": {"dir": "MOT17/train", "split": "train"}, "MOT20": {"dir": "MOT20/train", "split": "train"},
                          "SOMPT22": {"dir": "SOMPT22/train", "split": "train"},
                          "CrowdTrack-person": {"dir": "CrowdTrack-person", "split": "all"}},
             "weights_dir": str(weights), "hf_cache_dir": str(tmp_path / "hf"), "torch_hub_dir": str(tmp_path / "torchhub"),
             "trackeval": {"python": sys.executable, "repo": str(tmp_path)},
             "runs_root": str(tmp_path / "runs"), "physical_groups": None}
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "paths.local.yaml").write_text(yaml.safe_dump(paths, allow_unicode=True), encoding="utf-8")
    _test_configs.copy(cfg)
    # turn on the determinism sample to exercise the check seed
    e = yaml.safe_load((cfg / "experiment.yaml").read_text(encoding="utf-8"))
    e["quality"]["determinism_check"] = {"seed": 43, "sequences_per_dataset": 1}
    (cfg / "experiment.yaml").write_text(yaml.safe_dump(e, allow_unicode=True), encoding="utf-8")
    (tmp_path / "manifests").mkdir()
    exp = cfg / "experiment.yaml"
    assert cli.main(["inventory", "--paths", str(cfg / "paths.local.yaml"), "--out", str(tmp_path / "manifests" / "corpus.csv")]) == 0
    assert cli.main(["plan", "--experiment", str(exp)]) == 0
    _seal(tmp_path)  # the gate itself is tested in test_options_check
    loaded: list[str] = []
    checkpoints: list = []

    def load(det, _paths, _device, precision="fp32", checkpoint=None):
        loaded.append(det["name"])
        checkpoints.append((det["name"], checkpoint))
        return _Model(det["name"])
    monkeypatch.setattr(DB, "load_model", load)
    monkeypatch.setattr(DB, "bind", lambda m, v: _View(m, v))
    return {"exp": exp, "paths": paths, "cfg": cfg, "runs": tmp_path / "runs", "data": data, "loaded": loaded,
            "checkpoints": checkpoints, "corpus": tmp_path / "manifests" / "corpus.csv"}


def _detect(m, *extra) -> int:
    return cli.main(["detect", "--run-id", "t", "--experiment", str(m["exp"]), *extra])


def _report(m) -> dict:
    return json.loads((m["runs"] / "t" / "reports" / "detection_coverage.json").read_text(encoding="utf-8"))


def test_resolve_sequence_from_the_manifest(world):
    lines = K.read_corpus_csv(world["corpus"])
    s = K.resolve_sequence(next(l for l in lines if l["sequence_id"] == "MOT20-A"), world["paths"])
    assert Path(s["dir"]).is_dir() and (s["im_dir"], s["im_ext"], s["index_width"], s["n_frames"]) == ("img1", ".png", 6, 4)
    assert s["frames_manifest_sha256"] and s["dataset"] == "MOT20"
    with pytest.raises(ValueError, match="is not eligible"):
        K.resolve_sequence(dict(lines[0], eligible=False), world["paths"])


def test_detect_requires_an_explicit_device(world):
    with pytest.raises(SystemExit):
        cli.main(["detect", "--run-id", "t", "--experiment", str(world["exp"])])


def test_full_detect_loads_each_detector_once_and_then_nothing(world):
    assert _detect(world, "--device", "cpu") == 0
    rep = _report(world)
    assert rep["expected"] == 9 * 2 * 2 and len(rep["complete"]) == 36 and rep["failed"] == [] and rep["missing"] == []
    assert rep["effective_input_error"] is None
    assert sorted(world["loaded"]) == ["dfine_l", "frcnn_r50_fpn_v2", "yolo26m"]
    world["loaded"].clear()
    assert _detect(world, "--device", "cpu") == 0
    assert world["loaded"] == [] and len(_report(world)["complete"]) == 36


def test_detect_filters_detectors_sequences_and_seeds(world):
    assert _detect(world, "--device", "cpu", "--detectors", "yolo26m", "--sequences", "MOT20/MOT20-A", "--seeds", "42") == 0
    rep = _report(world)
    assert rep["expected"] == 3 and len(rep["complete"]) == 3 and world["loaded"] == ["yolo26m"]


def test_unreadable_frame_fails_only_the_cells_of_its_sequence(world):
    (world["data"] / "SOMPT22" / "train" / "SOMPT22-B" / "img1" / "000002.png").write_bytes(b"corrupted")
    assert _detect(world, "--device", "cpu") == 1
    rep = _report(world)
    assert len(rep["complete"]) == 18 and len(rep["failed"]) == 18
    assert all("SOMPT22-B" in f["sequence"] and "UnreadableFrame" in f["error"] for f in rep["failed"])


def test_same_effective_input_at_two_resolutions_fails_the_run(world, monkeypatch):
    monkeypatch.setattr(DB, "bind", lambda m, v: _View(m, v, collision=True))
    assert _detect(world, "--device", "cpu") == 1
    assert "produce the same effective input" in _report(world)["effective_input_error"]


@pytest.mark.parametrize("state", ["missing", "refused", "other_catalog", "other_versions"])
def test_detect_refuses_without_the_approved_check_options_seal(world, capsys, state):
    seal = world["cfg"].parent / "manifests" / "option_check.json"
    if state == "missing":
        seal.unlink()
    elif state == "refused":
        _seal(world["cfg"].parent, ["yolo26m_r640 and yolo26m_r800 give the same output"])
    elif state == "other_catalog":
        _seal(world["cfg"].parent, catalog_identity="0" * 16)
    else:
        _seal(world["cfg"].parent, versions={"detectors": {}, "trackers": {}})
    assert _detect(world, "--device", "cpu") == 2
    assert "check-options" in capsys.readouterr().err and world["loaded"] == []


def test_catalog_changed_after_plan_is_refused(world, capsys):
    cat = world["cfg"] / "catalog.yaml"
    cat.write_text(cat.read_text(encoding="utf-8") + "\n# changed after plan\n", encoding="utf-8")
    assert _detect(world, "--device", "cpu") == 2
    assert "plan" in capsys.readouterr().err


def test_gpu_lock_refuses_concurrent_work_and_takes_over_an_orphaned_lock(world, capsys):
    lock = world["runs"] / R.GPU_LOCK_NAME
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "command": "other"}), encoding="utf-8")
    assert _detect(world, "--device", "cuda:0") == 2
    assert "GPU busy" in capsys.readouterr().err
    lock.write_text(json.dumps({"pid": 2 ** 22 + 12345, "command": "dead"}), encoding="utf-8")  # no such process
    assert _detect(world, "--device", "cuda:0") == 0
    assert not lock.exists()


def test_gpu_lock_is_released_even_on_failure(tmp_path):
    lock = tmp_path / R.GPU_LOCK_NAME
    with pytest.raises(RuntimeError):
        with R.gpu_lock(lock, "test"):
            assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
            raise RuntimeError("crash")
    assert not lock.exists()


def test_loader_receives_the_checkpoint_resolved_by_plan(world):
    assert _detect(world, "--device", "cpu") == 0
    res = json.loads((world["cfg"].parent / "manifests" / "catalog.resolved.json").read_text(encoding="utf-8"))
    by_name = {v["detector"]["name"]: v["checkpoint"] for v in res["detector_variants"]}
    assert world["checkpoints"] and all(ck == by_name[n] for n, ck in world["checkpoints"])


def test_detect_refuses_weights_replaced_after_plan_before_processing(world, capsys):
    (Path(world["paths"]["weights_dir"]) / "yolo26m.pt").write_bytes(b"new weights with the same name")
    assert _detect(world, "--device", "cpu") == 2
    assert "changed after plan" in capsys.readouterr().err
    assert world["loaded"] == [] and not (world["runs"] / "t" / "detections").exists()


def test_detect_refuses_to_resume_the_run_in_another_environment(world, monkeypatch, capsys):
    assert _detect(world, "--device", "cpu") == 0
    before = len(world["loaded"])
    monkeypatch.setattr(R, "execution_environment", lambda device: {"kind": "cuda", "name": "another board"})
    assert _detect(world, "--device", "cpu") == 2
    assert "mixing environments" in capsys.readouterr().err and len(world["loaded"]) == before
    manifest = json.loads(next((world["runs"] / "t" / "detections").glob("*/*/seed42/manifest.json")).read_text(encoding="utf-8"))
    assert manifest["identity"]["environment"]["kind"] == "cpu"


def _log_records(runs: Path, run_id: str) -> list[dict]:
    return [json.loads(l) for l in (runs / run_id / "logs" / "run.jsonl").read_text(encoding="utf-8").splitlines()]


def test_run_chains_the_stages_one_process_per_detector_and_stops_at_the_first_failure(world, monkeypatch):
    calls, logs = [], []
    codes = {}

    def step(argv, log):
        calls.append(list(argv))
        logs.append(log)
        return codes.get(argv[0], 0)
    monkeypatch.setattr(cli, "_run_step", step)
    base = ["--run-id", "r", "--experiment", str(world["exp"])]
    assert cli.main(["run", *base, "--device", "cuda:0", "--workers", "3"]) == 0
    assert [c[0] for c in calls] == ["detect", "detect", "detect", "track", "evaluate"]
    detectors = [c[c.index("--detectors") + 1] for c in calls if c[0] == "detect"]
    assert sorted(detectors) == ["dfine_l", "frcnn_r50_fpn_v2", "yolo26m"]
    assert all("--device" in c and c[c.index("--device") + 1] == "cuda:0" for c in calls if c[0] in ("detect", "track"))
    ev = next(c for c in calls if c[0] == "evaluate")
    assert ev[ev.index("--workers") + 1] == "3" and "--device" not in ev
    assert all(c[c.index("--run-id") + 1] == "r" for c in calls)
    assert all(l.parent == world["runs"] / "r" / "logs" for l in logs) and len({l.name for l in logs}) == 5
    assert logs[0].name.endswith("_01_detect_" + detectors[0] + ".log") and logs[-1].name.endswith("_05_evaluate.log")
    reg = _log_records(world["runs"], "r")
    assert [x["state"] for x in reg] == ["running", "complete"] * 5 and all(x["returncode"] == 0 for x in reg[1::2])
    calls.clear()
    codes["track"] = 1
    assert cli.main(["run", *base, "--device", "cuda:0"]) == 1
    assert [c[0] for c in calls] == ["detect", "detect", "detect", "track"]
    last = _log_records(world["runs"], "r")[-1]
    assert (last["name"], last["state"], last["returncode"]) == ("track", "failed", 1)


def test_run_refuses_without_the_seal_before_calling_any_stage(world, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "_run_step", lambda argv, log: calls.append(argv) or 0)
    (world["cfg"].parent / "manifests" / "option_check.json").unlink()
    assert cli.main(["run", "--run-id", "r", "--experiment", str(world["exp"]), "--device", "cuda:0"]) == 2
    assert calls == [] and "check-options" in capsys.readouterr().err


def test_stage_shows_its_output_in_the_terminal_and_writes_the_log(tmp_path, capfd):
    log = tmp_path / "logs" / "x.log"
    assert cli._run_step(["--help"], log) == 0
    assert "usage" in log.read_text(encoding="utf-8") and "usage" in capfd.readouterr().out


def test_detect_shows_progress_per_cell_and_on_resumption_says_it_is_done(world, capsys):
    assert _detect(world, "--device", "cpu", "--part-size", "2") == 0
    out = capsys.readouterr().out
    for d in ("yolo26m", "dfine_l", "frcnn_r50_fpn_v2"):
        assert f"[detect {d}] 12 cells to do (0 already complete), 48 frames" in out
        assert f"[detect {d}] 12/12 " in out
    assert "remaining ~" in out and "detector " in out and "raw boxes" in out
    assert "part 2 of 2 (frames 3 to 4 of 4)" in out
    assert _detect(world, "--device", "cpu") == 0
    assert "[detect yolo26m] 12 cells already complete; nothing to do" in capsys.readouterr().out


def test_detect_shows_the_failure_right_away(world, capsys):
    (world["data"] / "SOMPT22" / "train" / "SOMPT22-B" / "img1" / "000002.png").write_bytes(b"corrupted")
    assert _detect(world, "--device", "cpu") == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "SOMPT22/SOMPT22-B: UnreadableFrame" in out


def test_the_pynvml_warning_does_not_show():
    import subprocess
    code = "from pedestrian_catalog import cli\ncli._quiet_warnings()\nimport torch\nprint('ok')\n"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0 and "pynvml" not in r.stderr

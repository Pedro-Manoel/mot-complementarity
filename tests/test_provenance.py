# -*- coding: utf-8 -*-
"""The loaded weights must be the checkpoints resolved by plan."""
from __future__ import annotations

import pytest

from pedestrian_catalog import provenance as PV
from pedestrian_catalog.backends import detectors as DB

YOLO = {"name": "yolo26m", "backend": "ultralytics", "checkpoint": {"file": "yolo26m.pt"}}
FRCNN = {"name": "frcnn_r50_fpn_v2", "backend": "torchvision",
         "checkpoint": {"torchvision_weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1", "file": "frcnn.pth"}}
DFINE = {"name": "dfine_l", "backend": "dfine", "checkpoint": {"hf_repo": "ustc-community/dfine-large-coco"}}
BASE_HF = "hf/models--ustc-community--dfine-large-coco"


def _paths(tmp):
    return {"weights_dir": str(tmp / "weights"), "hf_cache_dir": str(tmp / "hf"), "torch_hub_dir": str(tmp / "hub")}


def _snapshot(tmp, rev, content, main):
    base = tmp / BASE_HF
    (base / "snapshots" / rev).mkdir(parents=True, exist_ok=True)
    (base / "snapshots" / rev / "model.safetensors").write_bytes(content)
    (base / "refs").mkdir(exist_ok=True)
    (base / "refs" / "main").write_text(main, encoding="utf-8")


@pytest.mark.parametrize("det,folder", [(YOLO, "weights"), (FRCNN, "hub/checkpoints")])
def test_file_replaced_after_plan_is_refused(tmp_path, det, folder):
    p = tmp_path / folder / det["checkpoint"]["file"]
    p.parent.mkdir(parents=True)
    p.write_bytes(b"weights of the plan")
    paths = _paths(tmp_path)
    resolved = PV.resolve_checkpoint(det, paths)
    assert PV.verify_checkpoint(det, resolved, paths)["path"] == p
    p.write_bytes(b"new weights with the same name")
    with pytest.raises(PV.CheckpointMismatch, match="changed after plan"):
        PV.verify_checkpoint(det, resolved, paths)


def test_dfine_stays_at_the_recorded_revision_even_with_main_updated(tmp_path):
    _snapshot(tmp_path, "aaa111", b"old weights", main="aaa111")
    paths = _paths(tmp_path)
    resolved = PV.resolve_checkpoint(DFINE, paths)
    assert resolved["revision"] == "aaa111"
    _snapshot(tmp_path, "bbb222", b"new weights", main="bbb222")
    target = PV.verify_checkpoint(DFINE, resolved, paths)
    assert target["revision"] == "aaa111" and target["snapshot"].name == "aaa111"
    (tmp_path / BASE_HF / "snapshots" / "aaa111" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(PV.CheckpointMismatch, match="aaa111"):
        PV.verify_checkpoint(DFINE, resolved, paths)


def test_recorded_revision_missing_from_the_cache_is_refused(tmp_path):
    _snapshot(tmp_path, "aaa111", b"weights", main="aaa111")
    paths = _paths(tmp_path)
    resolved = {**PV.resolve_checkpoint(DFINE, paths), "revision": "ccc333"}
    with pytest.raises(FileNotFoundError, match="ccc333"):
        PV.verify_checkpoint(DFINE, resolved, paths)


def test_without_a_resolved_checkpoint_nothing_is_loaded(tmp_path):
    with pytest.raises(PV.CheckpointMismatch, match="plan"):
        PV.verify_checkpoint(YOLO, None, _paths(tmp_path))
    with pytest.raises(ValueError, match="checkpoint resolved"):
        DB.load_model(YOLO, _paths(tmp_path), "cpu")


def test_reid_weights_are_checked(tmp_path):
    p = tmp_path / "weights" / "osnet.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"reid")
    paths = _paths(tmp_path)
    asset = PV.resolve_weights_file("osnet", paths)
    assert PV.verify_weights_file(asset, paths) == p
    p.write_bytes(b"other reid")
    with pytest.raises(PV.CheckpointMismatch, match="osnet.pt"):
        PV.verify_weights_file(asset, paths)

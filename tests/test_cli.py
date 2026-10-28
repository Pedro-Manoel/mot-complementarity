"""Preparation commands (doctor, inventory, plan), the adopted experiment and export."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from pedestrian_catalog import cli

import _test_configs

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "torchvision", "ultralytics", "transformers", "boxmot", "cv2")

# a Windows drive path or a Unix home path
_ABSOLUTE = re.compile(r"(^|[^A-Za-z])[A-Za-z]:[\\/]|^/(Users|home|tmp)/")


def _no_private_paths(obj, root: Path) -> None:
    targets = {str(root), str(root).replace("\\", "/")}
    if isinstance(obj, dict):
        for k, v in obj.items():
            _no_private_paths(k, root)
            _no_private_paths(v, root)
    elif isinstance(obj, list):
        for v in obj:
            _no_private_paths(v, root)
    elif isinstance(obj, str):
        assert not any(a in obj for a in targets) and not _ABSOLUTE.search(obj), f"private path: {obj!r}"


def test_path_detector_recognizes_both_slashes():
    for bad in ("C:\\data\\x", "C:/data/x", "d:\\data", "see C:/tmp/a", "/home/someone"):
        try:
            _no_private_paths({"k": bad}, Path("Z:/not/used"))
        except AssertionError:
            continue
        raise AssertionError(f"did not recognize {bad!r}")
    _no_private_paths({"file": "yolo26m.pt", "sha256": "ab" * 32, "rel": "MOT20/train/MOT20-01"}, Path("Z:/x"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _seq(base: Path, rel: str, n: int = 4, content: bytes = b"x") -> None:
    d = base / rel
    (d / "img1").mkdir(parents=True)
    (d / "gt").mkdir()
    (d / "seqinfo.ini").write_text(f"[Sequence]\nname = {d.name}\nimDir = img1\nframeRate = 25\nseqLength = {n}\n"
                                   "imWidth = 640\nimHeight = 480\nimExt = .jpg\n", encoding="utf-8")
    for i in range(1, n + 1):
        (d / "img1" / f"{i:06d}.jpg").write_bytes(content)
    (d / "gt" / "gt.txt").write_text("\n".join(f"{i},1,1,1,2,4,1,1,1" for i in range(1, n + 1)) + "\n", encoding="utf-8")


def _setup(tmp: Path) -> dict:
    """Toy installation: data, fake weights, Hugging Face and torch caches and a fake TrackEval."""
    data = tmp / "data"
    _seq(data, "MOT20/train/MOT20-01", content=b"a")
    _seq(data, "SOMPT22/train/SOMPT22-02", content=b"b")
    (data / "MOT17" / "train").mkdir(parents=True)
    (data / "CrowdTrack-person").mkdir()
    weights = tmp / "weights"
    weights.mkdir()
    (weights / "yolo26m.pt").write_bytes(b"yolo-weights")
    (weights / "osnet_x0_25_msmt17.pt").write_bytes(b"reid-weights")
    snap = tmp / "hf" / "models--ustc-community--dfine-large-coco" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for name in ("model.safetensors", "preprocessor_config.json"):
        (snap / name).write_bytes(name.encode())
    (snap / "config.json").write_text(json.dumps({"num_queries": 300}), encoding="utf-8")
    (tmp / "hf" / "models--ustc-community--dfine-large-coco" / "refs").mkdir()
    (tmp / "hf" / "models--ustc-community--dfine-large-coco" / "refs" / "main").write_text("abc123", encoding="utf-8")
    hub = tmp / "torchhub" / "checkpoints"
    hub.mkdir(parents=True)
    (hub / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth").write_bytes(b"frcnn-weights")
    te = tmp / "te_repo"
    (te / "trackeval").mkdir(parents=True)
    (te / "trackeval" / "__init__.py").write_text("__version__ = '1.0.dev1'\nclass Evaluator:\n    pass\n", encoding="utf-8")
    paths = {
        "data_root": str(data),
        "datasets": {"MOT17": {"dir": "MOT17/train", "split": "train"}, "MOT20": {"dir": "MOT20/train", "split": "train"},
                     "SOMPT22": {"dir": "SOMPT22/train", "split": "train"},
                     "CrowdTrack-person": {"dir": "CrowdTrack-person", "split": "all"}},
        "weights_dir": str(weights), "hf_cache_dir": str(tmp / "hf"), "torch_hub_dir": str(tmp / "torchhub"),
        "trackeval": {"python": sys.executable, "repo": str(te)},
        "runs_root": str(tmp / "runs"), "physical_groups": None,
    }
    cfg = tmp / "configs"
    cfg.mkdir()
    (cfg / "paths.local.yaml").write_text(yaml.safe_dump(paths, allow_unicode=True), encoding="utf-8")
    _test_configs.copy(cfg)
    exp = yaml.safe_load((cfg / "experiment.yaml").read_text(encoding="utf-8"))
    exp["paths"] = "paths.local.yaml"
    (cfg / "experiment.yaml").write_text(yaml.safe_dump(exp, allow_unicode=True), encoding="utf-8")
    (tmp / "manifests").mkdir()
    return {"cfg": cfg, "paths": cfg / "paths.local.yaml", "exp": cfg / "experiment.yaml", "weights": weights,
            "snap": snap, "hub": hub}


def test_inventory_writes_the_manifest_and_summarizes(tmp_path, capsys):
    m = _setup(tmp_path)
    out = tmp_path / "manifests" / "corpus.csv"
    assert cli.main(["inventory", "--paths", str(m["paths"]), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "MOT20-01" in text and "SOMPT22-02" in text and str(tmp_path) not in text
    assert "sequences 2, eligible 2" in capsys.readouterr().out


def test_plan_resolves_checkpoints_and_counts_cells_without_importing_backends(tmp_path):
    m = _setup(tmp_path)
    assert cli.main(["inventory", "--paths", str(m["paths"]), "--out", str(tmp_path / "manifests" / "corpus.csv")]) == 0
    code = ("import sys\nfrom pedestrian_catalog.cli import main\n"
            f"rc = main(['plan', '--experiment', r'{m['exp']}'])\n"
            f"print('HEAVY', sorted(x for x in {HEAVY!r} if x in sys.modules))\nsys.exit(rc)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HEAVY []" in r.stdout
    assert "cells: detection 18, tracking 54, evaluation 54" in r.stdout
    res = json.loads((tmp_path / "manifests" / "catalog.resolved.json").read_text(encoding="utf-8"))
    _no_private_paths(res, tmp_path)
    assert len(res["configs"]) == 27 and len(res["detector_variants"]) == 9
    assert {c["resolution"]["requested"] for c in res["configs"]} == {640, 800, 1280}
    ck = {v["detector"]["name"]: v["checkpoint"] for v in res["detector_variants"]}
    assert ck["yolo26m"]["sha256"] == _sha(m["weights"] / "yolo26m.pt")
    assert ck["frcnn_r50_fpn_v2"]["sha256"] == _sha(m["hub"] / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth")
    assert ck["dfine_l"]["revision"] == "abc123" and len(ck["dfine_l"]["sha256"]) == 64
    reid = res["tracker_assets"]["reid_weights"]
    assert reid["sha256"] == _sha(m["weights"] / "osnet_x0_25_msmt17.pt")
    n = res["corpus"]["n_eligible"]
    assert n == 2 and res["cells"] == {"detection": 9 * n, "tracking": 27 * n, "evaluation": 27 * n}
    assert res["quality"]["replicate_seeds"] == [42] and res["quality"]["sequence_seed_pairs"] == 2
    assert res["quality"]["determinism_sequences"] == []
    assert not {"checkpoints_confirmed_by_author", "premises", "profiling"} & set(res)
    assert res["boxmot_distribution"] == "17.0.0" and res["boxmot_module_version"] == "16.0.11"
    assert res["detector_limits"] == {"dfine_l": {"num_queries": 300}}


def test_plan_refuses_a_box_limit_other_than_the_fixed_dfine_queries(tmp_path, capsys):
    m = _setup(tmp_path)
    assert cli.main(["inventory", "--paths", str(m["paths"]), "--out", str(tmp_path / "manifests" / "corpus.csv")]) == 0
    cat = m["cfg"] / "catalog.yaml"
    cat.write_text(cat.read_text(encoding="utf-8").replace("max_det: 300", "max_det: 1000"), encoding="utf-8")
    assert cli.main(["plan", "--experiment", str(m["exp"])]) == 2
    assert "300 fixed queries" in capsys.readouterr().err


def test_plan_fails_with_a_missing_checkpoint(tmp_path, capsys):
    m = _setup(tmp_path)
    assert cli.main(["inventory", "--paths", str(m["paths"]), "--out", str(tmp_path / "manifests" / "corpus.csv")]) == 0
    (m["weights"] / "yolo26m.pt").unlink()
    assert cli.main(["plan", "--experiment", str(m["exp"])]) != 0
    assert "yolo26m.pt" in capsys.readouterr().err


def test_doctor_records_the_environment_without_private_paths(tmp_path):
    m = _setup(tmp_path)
    out = tmp_path / "manifests" / "environment.json"
    code = ("import sys\nfrom pedestrian_catalog.cli import main\n"
            f"rc = main(['doctor', '--paths', r'{m['paths']}', '--out', r'{out}'])\n"
            f"print('HEAVY', sorted(x for x in {HEAVY!r} if x in sys.modules))\nsys.exit(rc)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HEAVY []" in r.stdout
    env = json.loads(out.read_text(encoding="utf-8"))
    _no_private_paths(env, tmp_path)
    assert env["python"]["version"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert "numpy" in env["packages"] and "boxmot" in env["packages"]
    # the BoxMOT module __version__ (16.0.11) differs from the distribution version (17.0.0)
    assert "boxmot" in env["module_versions"]
    assert env["module_versions"]["boxmot"] is None or env["module_versions"]["boxmot"][0].isdigit()
    assert env["trackeval"]["status"] == "ok" and env["trackeval"]["has_evaluator"] is True


def test_doctor_fails_when_the_imported_trackeval_has_no_evaluator(tmp_path):
    m = _setup(tmp_path)
    (tmp_path / "te_repo" / "trackeval" / "__init__.py").write_text("__version__ = 'x'\n", encoding="utf-8")
    out = tmp_path / "manifests" / "environment.json"
    assert cli.main(["doctor", "--paths", str(m["paths"]), "--out", str(out)]) == 1
    te = json.loads(out.read_text(encoding="utf-8"))["trackeval"]
    assert te["status"] == "failed" and te["has_evaluator"] is False


def test_doctor_fails_when_trackeval_does_not_import(tmp_path):
    m = _setup(tmp_path)
    paths = yaml.safe_load(m["paths"].read_text(encoding="utf-8"))
    paths["trackeval"]["repo"] = str(tmp_path / "does_not_exist")
    m["paths"].write_text(yaml.safe_dump(paths), encoding="utf-8")
    out = tmp_path / "manifests" / "environment.json"
    assert cli.main(["doctor", "--paths", str(m["paths"]), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["trackeval"]["status"] == "failed"


def test_adopted_experiment_has_one_seed_resource_sampling_and_a_clip_variant_of_the_catalog():
    from pedestrian_catalog import catalog as C
    cfg = ROOT / "configs"
    exp = yaml.safe_load((cfg / "experiment.yaml").read_text(encoding="utf-8"))
    spec = yaml.safe_load((cfg / exp["catalog"]).read_text(encoding="utf-8"))
    spec["trackers"] = yaml.safe_load((cfg / exp["trackers"]).read_text(encoding="utf-8"))["trackers"]
    configs = C.build_catalog(spec)
    assert len(configs) == 108
    assert exp["quality"] == {"replicate_seeds": [42]}
    assert exp["resources"] == {"sample_hz": 4}
    assert "profiling" not in exp and "analysis" not in exp
    assert exp["option_check"]["tracker_clip"]["detector_variant"] in {c["detector_variant_id"] for c in configs}


def test_inventory_reads_the_camera_partition_relative_to_the_paths_file(tmp_path, capsys):
    m = _setup(tmp_path)
    (m["cfg"] / "camera.csv").write_text("dataset,sequence_id,median_frac,camera_motion,source\n"
                                         "MOT20,MOT20-01,0.0003,fixed,measured\nSOMPT22,SOMPT22-02,0.002,moving,measured\n",
                                         encoding="utf-8")
    p = yaml.safe_load(m["paths"].read_text(encoding="utf-8"))
    m["paths"].write_text(yaml.safe_dump({**p, "camera_motion": "camera.csv"}, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "manifests" / "corpus.csv"
    assert cli.main(["inventory", "--paths", str(m["paths"]), "--out", str(out)]) == 0
    from pedestrian_catalog import corpus as K
    lines = {l["sequence_id"]: l for l in K.read_corpus_csv(out)}
    assert (lines["MOT20-01"]["camera_motion"], lines["SOMPT22-02"]["camera_motion"]) == ("fixed", "moving")
    assert "static camera 1, moving 1" in capsys.readouterr().out


def test_inventory_reads_selection_and_groups_relative_to_the_paths_file(tmp_path, capsys):
    m = _setup(tmp_path)
    _seq(tmp_path / "data", "SOMPT22/train/SOMPT22-04", content=b"c")
    (m["cfg"] / "sel.csv").write_text("sequence_id\nSOMPT22-02\n", encoding="utf-8")
    (m["cfg"] / "groups.csv").write_text("dataset,sequence_id,physical_group_id,source\nSOMPT22,SOMPT22-02,SOMPT22:pole,draw\n",
                                         encoding="utf-8")
    p = yaml.safe_load(m["paths"].read_text(encoding="utf-8"))
    p.update(selection={"SOMPT22": "sel.csv"}, physical_groups="groups.csv")
    m["paths"].write_text(yaml.safe_dump(p, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "manifests" / "corpus.csv"
    assert cli.main(["inventory", "--paths", str(m["paths"]), "--out", str(out)]) == 0
    from pedestrian_catalog import corpus as K
    lines = {l["sequence_id"]: l for l in K.read_corpus_csv(out)}
    assert lines["SOMPT22-02"]["eligible"] and lines["SOMPT22-02"]["physical_group_id"] == "SOMPT22:pole"
    assert lines["SOMPT22-04"]["exclusion_reason"] == "outside_selection" and lines["MOT20-01"]["eligible"]


def test_export_refuses_a_run_without_evaluated_cells(tmp_path, capsys):
    m = _setup(tmp_path)
    assert cli.main(["inventory", "--paths", str(m["paths"]), "--out", str(tmp_path / "manifests" / "corpus.csv")]) == 0
    assert cli.main(["plan", "--experiment", str(m["exp"])]) == 0
    out = tmp_path / "tables"
    assert cli.main(["export", "--run-id", "r1", "--experiment", str(m["exp"]), "--out", str(out)]) == 2
    assert "54 cells without a valid evaluation" in capsys.readouterr().err
    assert not list(out.glob("*.csv"))


def test_shipped_resolved_catalog_matches_the_configs():
    def sha(p):
        return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    r = json.loads((ROOT / "manifests" / "catalog.resolved.json").read_text(encoding="utf-8"))
    assert r["catalog_spec_sha256"] == sha("configs/catalog.yaml")
    assert r["trackers_spec_sha256"] == sha("configs/trackers.yaml")
    assert r["corpus"]["manifest_sha256"] == sha("manifests/corpus.csv")

# -*- coding: utf-8 -*-
"""Sequence evaluation with the real TrackEval (skipped without configs/paths.local.yaml or its environment)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from pedestrian_catalog import evaluation as EV
from pedestrian_catalog.detection import InvalidCache

ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "configs" / "paths.local.yaml"
N = 10


@pytest.fixture(scope="module")
def evaluator():
    if not PATHS.is_file():
        pytest.skip("configs/paths.local.yaml missing")
    te = yaml.safe_load(PATHS.read_text(encoding="utf-8"))["trackeval"]
    if not Path(te["python"]).is_file():
        pytest.skip("separate TrackEval environment missing")
    return {"python": te["python"], "repo": te["repo"], "do_preproc": True}


def _gt_lines(extra=()) -> list[str]:
    lines = []
    for i in range(1, N + 1):
        lines.append(f"{i},1,{100 + 3 * i},100,40,100,1,1,1.0")
        lines.append(f"{i},2,{400 - 3 * i},120,40,100,1,1,1.0")
    return lines + list(extra)


def _sequence(tmp: Path, dataset: str, extra=(), name="SEQ-01", after=0) -> dict:
    d = tmp / "data" / dataset / name
    (d / "gt").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(f"[Sequence]\nname = {name}\nimDir = img1\nframeRate = 30\nseqLength = {N}\n"
                                   "imWidth = 640\nimHeight = 480\nimExt = .jpg\n", encoding="utf-8")
    beyond = [f"{N + 1 + j},1,{100 + 3 * (N + 1 + j)},100,40,100,1,1,1.0" for j in range(after)]
    (d / "gt" / "gt.txt").write_text("\n".join(_gt_lines(extra) + beyond) + "\n", encoding="utf-8")
    return {"dataset": dataset, "sequence_id": name, "dir": str(d), "n_frames": N, "width": 640, "height": 480, "fps": 30.0,
            "gt_sha256": hashlib.sha256((d / "gt" / "gt.txt").read_bytes()).hexdigest(), "gt_rows_after_last_frame": after,
            "annotation_protocol": {"MOT17": "motchallenge_mot17", "MOT20": "motchallenge_mot20"}.get(dataset, "sompt22_mot")}


def _tracks(tmp: Path, name: str, lines: list[str], seq="SEQ-01") -> dict:
    d = tmp / "tracks" / name
    d.mkdir(parents=True)
    (d / "tracks_mot.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    sha = hashlib.sha256((d / "tracks_mot.txt").read_bytes()).hexdigest()
    return {"cell_id": sha[:16], "cell_dir": str(d), "config_id": name, "sequence_key": f"X/{seq}", "replicate_seed": 42,
            "files": {"tracks_mot.txt": {"sha256": sha}}}


def _perfect() -> list[str]:
    return [f"{l.split(',')[0]},{l.split(',')[1]},{','.join(l.split(',')[2:6])},1,-1,-1,-1" for l in _gt_lines()]


def _run(tmp: Path, evaluator: dict) -> dict:
    return {**evaluator, "run_dir": tmp / "run"}


def test_perfect_tracks_give_hota_100_on_the_x100_scale(tmp_path, evaluator):
    m = EV.evaluate_sequence(_sequence(tmp_path, "MOT17"), _tracks(tmp_path, "perfect", _perfect()), _run(tmp_path, evaluator))
    met = EV.load_metrics(m)
    assert met["normalized"]["HOTA"] == pytest.approx(100.0) and met["normalized"]["DetA"] == pytest.approx(100.0)
    assert met["normalized"]["AssA"] == pytest.approx(100.0) and met["normalized"]["IDSW"] == 0
    assert met["units"]["HOTA"] == "x100" and met["units"]["IDSW"] == "count"
    assert "HOTA___AUC" in met["native"] and "HOTA___50" in met["native"] and "Frag" in met["native"]
    assert met["benchmark"] == "MOT17" and met["do_preproc"] is True and met["evaluated_frames"] == N
    assert met["research_cost"]["batch_cells"] == 1 and met["research_cost"]["seconds_per_cell_frame"] > 0
    for name in ("pedestrian_detailed.csv", "pedestrian_summary.txt", "metrics.json"):
        assert (Path(m["cell_dir"]) / name).is_file() and name in m["files"]


def test_no_predictions_give_zero_and_all_the_metrics(tmp_path, evaluator):
    m = EV.evaluate_sequence(_sequence(tmp_path, "MOT17"), _tracks(tmp_path, "empty", []), _run(tmp_path, evaluator))
    met = EV.load_metrics(m)
    assert met["normalized"]["HOTA"] == pytest.approx(0.0) and met["normalized"]["DetA"] == pytest.approx(0.0)
    assert set(EV.NORMALIZED) <= set(met["normalized"])


def test_identity_swap_lowers_the_association_and_not_the_detection(tmp_path, evaluator):
    swapped = []
    for l in _perfect():
        c = l.split(",")
        if int(c[0]) > N // 2:
            c[1] = {"1": "2", "2": "1"}[c[1]]
        swapped.append(",".join(c))
    m = EV.evaluate_sequence(_sequence(tmp_path, "MOT17"), _tracks(tmp_path, "swapped", swapped), _run(tmp_path, evaluator))
    everything = EV.load_metrics(m)
    met = everything["normalized"]
    assert met["DetA"] == pytest.approx(100.0) and met["AssA"] < 90.0 and met["IDSW"] >= 2 and met["HOTA"] < 95.0
    assert isinstance(met["IDSW"], int) and met["IDSW"] == everything["native"]["IDSW"]


def test_prediction_on_a_distractor_is_forgiven_only_with_preprocessing(tmp_path, evaluator):
    distractor = [f"{i},3,500,300,40,100,0,7,1.0" for i in range(1, N + 1)]  # static person: class 7, conf 0
    seq = _sequence(tmp_path, "MOT17", extra=distractor)
    on_it = _tracks(tmp_path, "on_distractor", _perfect() + [f"{i},9,500,300,40,100,1,-1,-1,-1" for i in range(1, N + 1)])
    with_pre = EV.load_metrics(EV.evaluate_sequence(seq, on_it, _run(tmp_path, evaluator)))
    without_pre = EV.load_metrics(EV.evaluate_sequence(seq, on_it, {**_run(tmp_path, evaluator), "do_preproc": False,
                                                                    "run_dir": tmp_path / "no_preproc"}))
    assert with_pre["normalized"]["HOTA"] == pytest.approx(100.0)
    assert without_pre["normalized"]["HOTA"] < 100.0


def test_non_mot_vehicle_is_a_distractor_only_on_the_mot20_benchmark(tmp_path, evaluator):
    vehicle = [f"{i},4,500,300,60,60,0,6,1.0" for i in range(1, N + 1)]  # non_mot_vehicle: class 6
    on_it = _perfect() + [f"{i},9,500,300,60,60,1,-1,-1,-1" for i in range(1, N + 1)]
    mot20 = EV.load_metrics(EV.evaluate_sequence(_sequence(tmp_path, "MOT20", extra=vehicle), _tracks(tmp_path, "a", on_it),
                                                 _run(tmp_path, evaluator)))
    mot17 = EV.load_metrics(EV.evaluate_sequence(_sequence(tmp_path, "MOT17", extra=vehicle), _tracks(tmp_path, "b", on_it),
                                                 _run(tmp_path, evaluator)))
    assert mot20["benchmark"] == "MOT20" and mot20["normalized"]["HOTA"] == pytest.approx(100.0)
    assert mot17["benchmark"] == "MOT17" and mot17["normalized"]["HOTA"] < 100.0


def test_batch_equals_the_isolated_evaluation_and_resumes_without_redoing(tmp_path, evaluator):
    seq = _sequence(tmp_path, "SOMPT22")
    tr = [_tracks(tmp_path, "perfect", _perfect()), _tracks(tmp_path, "empty", []),
          _tracks(tmp_path, "half", _perfect()[: N])]
    batch = EV.evaluate_batch(seq, tr, _run(tmp_path, evaluator))
    assert all(EV.load_metrics(m)["research_cost"]["batch_cells"] == 3 for m in batch)
    for t, m in zip(tr, batch):
        alone = EV.evaluate_sequence(seq, t, {**_run(tmp_path, evaluator), "run_dir": tmp_path / "isolated"})
        assert EV.load_metrics(m)["native"] == EV.load_metrics(alone)["native"]
    again = EV.evaluate_batch(seq, tr, _run(tmp_path, evaluator))
    assert [m["cell_id"] for m in again] == [m["cell_id"] for m in batch]
    states = [json.loads(l)["state"] for l in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert states.count("skipped") == 3


def test_gt_changed_after_the_inventory_is_refused(tmp_path, evaluator):
    seq = _sequence(tmp_path, "MOT17")
    Path(seq["dir"], "gt", "gt.txt").write_text("\n".join(_gt_lines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(EV.GTMismatch):
        EV.evaluate_sequence(seq, _tracks(tmp_path, "p", _perfect()), _run(tmp_path, evaluator))


def test_tampered_tracks_are_refused(tmp_path, evaluator):
    t = _tracks(tmp_path, "p", _perfect())
    (Path(t["cell_dir"]) / "tracks_mot.txt").write_text("1,1,0,0,1,1,1,-1,-1,-1\n", encoding="utf-8")
    with pytest.raises(InvalidCache):
        EV.evaluate_sequence(_sequence(tmp_path, "MOT17"), t, _run(tmp_path, evaluator))


def test_with_and_without_preprocessing_in_the_same_run_are_distinct_cells(tmp_path, evaluator):
    distractor = [f"{i},3,500,300,40,100,0,7,1.0" for i in range(1, N + 1)]
    seq = _sequence(tmp_path, "MOT17", extra=distractor)
    on_it = _tracks(tmp_path, "on_distractor", _perfect() + [f"{i},9,500,300,40,100,1,-1,-1,-1" for i in range(1, N + 1)])
    with_pre = EV.evaluate_sequence(seq, on_it, _run(tmp_path, evaluator))
    without_pre = EV.evaluate_sequence(seq, on_it, {**_run(tmp_path, evaluator), "do_preproc": False})
    assert with_pre["cell_id"] != without_pre["cell_id"] and EV.load_metrics(without_pre)["normalized"]["HOTA"] < 100.0


def test_gt_after_the_last_frame_is_left_out_of_the_evaluation(tmp_path, evaluator):
    seq = _sequence(tmp_path, "CROWD", after=3)
    m = EV.evaluate_sequence(seq, _tracks(tmp_path, "perfect", _perfect()), _run(tmp_path, evaluator))
    met = EV.load_metrics(m)
    assert met["normalized"]["HOTA"] == pytest.approx(100.0) and met["evaluated_frames"] == N
    assert m["identity"]["gt_rows_after_last_frame"] == 3


@pytest.mark.parametrize("declared", [0, 2])
def test_rows_after_the_last_frame_different_from_the_inventory_are_refused(tmp_path, evaluator, declared):
    seq = dict(_sequence(tmp_path, "CROWD", after=3), gt_rows_after_last_frame=declared)
    with pytest.raises(EV.GTMismatch, match="after the last frame"):
        EV.evaluate_sequence(seq, _tracks(tmp_path, "p", _perfect()), _run(tmp_path, evaluator))


def test_evaluation_identity():
    seq = {"dataset": "MOT17", "sequence_id": "S", "gt_sha256": "g" * 64, "gt_rows_after_last_frame": 0}
    tr = {"cell_id": "t" * 16, "files": {"tracks_mot.txt": {"sha256": "s" * 64}}}
    ev = {"trackeval_commit": "a"}
    base = EV.evaluation_cell_identity(seq, tr, "MOT17", True, ev)["cell_id"]
    others = [EV.evaluation_cell_identity(dict(seq, gt_sha256="h" * 64), tr, "MOT17", True, ev),
              EV.evaluation_cell_identity(seq, dict(tr, files={"tracks_mot.txt": {"sha256": "x" * 64}}), "MOT17", True, ev),
              EV.evaluation_cell_identity(seq, tr, "MOT20", True, ev),
              EV.evaluation_cell_identity(seq, tr, "MOT17", False, ev),
              EV.evaluation_cell_identity(seq, tr, "MOT17", True, {"trackeval_commit": "b"}),
              EV.evaluation_cell_identity(dict(seq, gt_rows_after_last_frame=141), tr, "MOT17", True, ev)]
    assert len({o["cell_id"] for o in others} | {base}) == 7


def test_normalization_refuses_a_missing_column():
    native = {c: 1.0 for c in EV._COLUMNS.values()}
    assert EV._normalize(native)["HOTA"] == pytest.approx(100.0)
    del native["AssA___AUC"]
    with pytest.raises(EV.EvaluationError, match="AssA___AUC"):
        EV._normalize(native)


def test_benchmark_by_annotation_protocol():
    assert EV.benchmark_of({"annotation_protocol": "motchallenge_mot20"}) == "MOT20"
    assert EV.benchmark_of({"annotation_protocol": "motchallenge_mot17"}) == "MOT17"
    assert EV.benchmark_of({"annotation_protocol": "crowdtrack_person_mot"}) == "MOT17"
    with pytest.raises(ValueError):
        EV.benchmark_of({"annotation_protocol": "unknown"})


def test_evaluation_keeps_the_resources_of_the_trackeval_subprocess(tmp_path, evaluator):
    m = EV.evaluate_sequence(_sequence(tmp_path, "MOT17"), _tracks(tmp_path, "perfect", _perfect()), _run(tmp_path, evaluator))
    r = EV.load_metrics(m)["resources"]
    assert r["batch_cells"] == 1 and r["duration_s"] > 0 and r["provenance"]["from_process_start"] is True
    assert r["cpu_tree_time_s"] is not None and r["cpu_tree_time_s"] > 0 and r["ram_tree_rss_peak_mb"] > 0
    assert r["gpu_energy_j"] is None
    import os
    assert r["provenance"]["measured_pid"] not in (None, os.getpid())


IGNORE_ID = 1_000_000_000


def _sequence_pp22(tmp: Path, name="PersonPath22-00018") -> dict:
    """Keyframes on the odd frames; the even ones carry an ignore region (class 13), as in the conversion."""
    lines = []
    for l in _gt_lines():
        if int(l.split(",")[0]) % 2:
            lines.append(l.replace(",1,1,1.0", ",1,1,-1,-1"))
    for i in range(2, N + 1, 2):
        lines.append(f"{i},{IGNORE_ID},-640,-480,1920,1440,1,13,-1,-1")
    d = tmp / "data" / "PersonPath22" / name
    (d / "gt").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(f"[Sequence]\nname = {name}\nimDir = img1\nframeRate = 30\nseqLength = {N}\n"
                                   "imWidth = 640\nimHeight = 480\nimExt = .jpg\nevalDataset = PersonPath22\n", encoding="utf-8")
    (d / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"dataset": "PersonPath22", "sequence_id": name, "dir": str(d), "n_frames": N, "width": 640, "height": 480, "fps": 30.0,
            "gt_sha256": hashlib.sha256((d / "gt" / "gt.txt").read_bytes()).hexdigest(), "gt_rows_after_last_frame": 0,
            "annotation_protocol": "personpath22_keyframes"}


def test_personpath22_class_and_benchmark():
    pp = {"annotation_protocol": "personpath22_keyframes"}
    assert EV.benchmark_of(pp) == "person_path_22" and EV.dataset_class_of(pp) == "PersonPath22"
    for protocol in ("motchallenge_mot17", "motchallenge_mot20", "sompt22_mot", "crowdtrack_person_mot"):
        assert EV.dataset_class_of({"annotation_protocol": protocol}) == "MotChallenge2DBox"
    with pytest.raises(ValueError):
        EV.dataset_class_of({"annotation_protocol": "unknown"})


def test_evaluation_identity_keys_do_not_include_the_dataset_class():
    seq = {"dataset": "MOT17", "sequence_id": "S", "gt_sha256": "g" * 64, "gt_rows_after_last_frame": 0}
    tr = {"cell_id": "t" * 16, "files": {"tracks_mot.txt": {"sha256": "s" * 64}}}
    ident = EV.evaluation_cell_identity(seq, tr, "MOT17", True, {"trackeval_commit": "a"})
    assert set(ident) == {"cell_id", "stage", "tracks_cell_id", "tracks_sha256", "gt_sha256", "gt_rows_after_last_frame",
                          "sequence_key", "benchmark", "do_preproc", "metrics", "evaluator"}


def test_personpath22_ignores_predictions_outside_the_keyframes(tmp_path, evaluator):
    seq = _sequence_pp22(tmp_path)
    tr = _tracks(tmp_path, "perfect_pp22", _perfect(), seq=seq["sequence_id"])
    pp = EV.load_metrics(EV.evaluate_sequence(seq, tr, _run(tmp_path, evaluator)))
    assert pp["benchmark"] == "person_path_22"
    assert pp["normalized"]["HOTA"] == pytest.approx(100.0) and pp["normalized"]["CLR_FP"] == 0
    mot = EV.load_metrics(EV.evaluate_sequence(dict(seq, annotation_protocol="motchallenge_mot17"), tr,
                                               {**_run(tmp_path, evaluator), "run_dir": tmp_path / "mot"}))
    assert mot["normalized"]["HOTA"] < 90.0 and mot["normalized"]["CLR_FP"] > 0

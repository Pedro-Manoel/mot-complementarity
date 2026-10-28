# -*- coding: utf-8 -*-
"""The evaluate command and the quality matrix (uses the separate TrackEval environment; skipped without it)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml

from pedestrian_catalog import cli
from test_detect_run import _Model, _View, world  # noqa: F401
from test_track_run import tracked  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "configs" / "paths.local.yaml"


@pytest.fixture
def evaluated(tracked):
    if not PATHS.is_file():
        pytest.skip("configs/paths.local.yaml missing")
    te = yaml.safe_load(PATHS.read_text(encoding="utf-8"))["trackeval"]
    if not Path(te["python"]).is_file():
        pytest.skip("separate TrackEval environment missing")
    p = tracked["cfg"] / "paths.local.yaml"
    paths = yaml.safe_load(p.read_text(encoding="utf-8"))
    paths["trackeval"] = te
    p.write_text(yaml.safe_dump(paths, allow_unicode=True), encoding="utf-8")
    assert cli.main(["track", "--run-id", "t", "--experiment", str(tracked["exp"]), "--device", "cpu"]) == 0
    return tracked


def _evaluate(m, *extra) -> int:
    return cli.main(["evaluate", "--run-id", "t", "--experiment", str(m["exp"]), *extra])


def _coverage(m) -> dict:
    return json.loads((m["runs"] / "t" / "reports" / "evaluation_coverage.json").read_text(encoding="utf-8"))


def _matrix(m) -> list[dict]:
    with (m["runs"] / "t" / "reports" / "quality_long.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _events(m, state) -> int:
    return sum(1 for l in (m["runs"] / "t" / "events.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(l)["stage"] == "evaluate" and json.loads(l)["state"] == state)


def test_evaluate_complete_matrix_and_resume(evaluated):
    assert _evaluate(evaluated) == 0
    cov = _coverage(evaluated)
    assert cov["expected"] == 108 and len(cov["complete"]) == 108 and cov["failed"] == [] and cov["missing_tracks"] == []
    rows = _matrix(evaluated)
    assert len(rows) == 108 and {"config_id", "dataset", "sequence_id", "seed", "HOTA", "DetA", "AssA", "IDF1", "IDSW"} <= set(rows[0])
    assert all(0.0 <= float(l["HOTA"]) <= 100.0 for l in rows)
    assert _events(evaluated, "complete") == 108
    assert _evaluate(evaluated) == 0 and _events(evaluated, "skipped") == 108 and _events(evaluated, "complete") == 108


def test_missing_tracks_show_up_in_the_coverage(evaluated):
    for cell in (evaluated["runs"] / "t" / "tracks").glob("*/MOT20__MOT20-A/seed43"):
        shutil.rmtree(cell)
    assert _evaluate(evaluated) == 1
    cov = _coverage(evaluated)
    assert len(cov["missing_tracks"]) == 27 and len(cov["complete"]) == 108 - 27
    assert all(x["sequence"] == "MOT20/MOT20-A" and x["seed"] == 43 for x in cov["missing_tracks"])


def test_changed_gt_fails_only_its_own_sequence(evaluated):
    gt = evaluated["cfg"].parent / "data" / "SOMPT22" / "train" / "SOMPT22-B" / "gt" / "gt.txt"
    gt.write_text(gt.read_text(encoding="utf-8") + "4,2,20,20,8,8,1,1,1\n", encoding="utf-8")
    assert _evaluate(evaluated) == 1
    cov = _coverage(evaluated)
    assert len(cov["failed"]) == 54 and len(cov["complete"]) == 54 and cov["missing_tracks"] == []
    assert all(x["sequence"] == "SOMPT22/SOMPT22-B" and "GTMismatch" in x["error"] for x in cov["failed"])
    assert len(_matrix(evaluated)) == 54 and _events(evaluated, "complete") == 54


def test_parallel_equals_serial(evaluated):
    assert _evaluate(evaluated, "--configs", "yolo26m_r640__bytetrack,dfine_l_r800__ocsort") == 0
    serial = {(l["config_id"], l["sequence_id"], l["seed"]): l["HOTA"] for l in _matrix(evaluated)}
    shutil.rmtree(evaluated["runs"] / "t" / "metrics")
    assert _evaluate(evaluated, "--configs", "yolo26m_r640__bytetrack,dfine_l_r800__ocsort", "--workers", "2") == 0
    parallel = {(l["config_id"], l["sequence_id"], l["seed"]): l["HOTA"] for l in _matrix(evaluated)}
    assert parallel == serial and len(serial) == 2 * 2 * 2


def test_collection_ignores_an_evaluation_made_with_another_count_of_gt_rows_after_the_last_frame(evaluated):
    from pedestrian_catalog import evaluation as EV
    from pedestrian_catalog.corpus import read_corpus_csv
    assert _evaluate(evaluated) == 0
    root = evaluated["cfg"].parent
    configs = json.loads((root / "manifests" / "catalog.resolved.json").read_text(encoding="utf-8"))["configs"]
    lines = [l for l in read_corpus_csv(evaluated["corpus"]) if l["eligible"]]
    ok, missing = EV.collect_metrics(evaluated["runs"] / "t", configs, lines, [42], True)
    assert missing == [] and len(ok) == len(configs) * len(lines)
    other = [dict(l, gt_rows_after_last_frame=l["gt_rows_after_last_frame"] + 1) if i == 0 else l for i, l in enumerate(lines)]
    ok, missing = EV.collect_metrics(evaluated["runs"] / "t", configs, other, [42], True)
    assert len(missing) == len(configs) and all(f.endswith("no evaluation") for f in missing)
    assert len(ok) == len(configs) * (len(lines) - 1)


def test_evaluate_shows_progress_per_sequence(evaluated, capsys):
    assert _evaluate(evaluated, "--workers", "2") == 0
    out = capsys.readouterr().out
    assert "[evaluate] 2 sequences with tracks, 108 cells" in out
    assert "[evaluate] 2/2 " in out and "54 cells in" in out

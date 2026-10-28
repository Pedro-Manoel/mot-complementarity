"""The analysis rebuilds the shipped result tables from the exported tables."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from pedestrian_catalog import analysis, cli, results, stats

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "paper_results"
TABLES = sorted(p.stem for p in (SHIPPED / "analysis").glob("*.csv"))


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory) -> Path:
    src = tmp_path_factory.mktemp("results")
    for f in results.FILES:
        shutil.copyfile(SHIPPED / f, src / f)
    assert cli.main(["analyze", "--results", str(src)]) == 0
    return src / "analysis"


def test_every_table_is_written(rebuilt):
    assert sorted(p.stem for p in rebuilt.glob("*.csv")) == TABLES and len(TABLES) == 17


@pytest.mark.parametrize("name", TABLES)
def test_tables_are_the_shipped_ones(rebuilt, name):
    assert (rebuilt / f"{name}.csv").read_text(encoding="utf-8") == \
        (SHIPPED / "analysis" / f"{name}.csv").read_text(encoding="utf-8")


def test_headline_numbers(rebuilt):
    g = results.read_csv(rebuilt / "gap_summary.csv").set_index("scope")
    assert g.loc["all", "fixed_config"] == "yolo26m_r1280__botsort_h0.45"
    assert g.loc["all", "videos"] == 100
    assert round(g.loc["all", "fixed_mean"], 2) == 45.63
    assert round(g.loc["all", "oracle_mean"], 2) == 47.35
    assert round(g.loc["all", "gap_mean"], 2) == 1.72
    assert len(results.read_csv(rebuilt / "runs.csv")) == 3


def test_gap_by_video_matches_the_summary(rebuilt):
    v = results.read_csv(rebuilt / "gap_by_video.csv")
    g = results.read_csv(rebuilt / "gap_summary.csv").set_index("scope")
    assert len(v) == 100 and (v["gap"] >= 0).all()
    assert v["gap"].mean() == pytest.approx(g.loc["all", "gap_mean"], abs=1e-12)
    assert (v["oracle_hota"] - v["fixed_hota"]).to_numpy() == pytest.approx(v["gap"].to_numpy(), abs=1e-12)


def test_cost_is_the_median_of_the_runs():
    D = results.load(SHIPPED)
    S = np.stack([X.to_numpy() for X in D["L_runs"].values()])
    assert np.array_equal(D["L"].to_numpy(), np.median(S, axis=0))


def test_twenty_five_scene_relations(rebuilt):
    rel = results.read_csv(rebuilt / "scene_relations.csv")
    assert len(rel) == 25 and (rel["p_holm"] >= rel["p_value"]).all()


def test_fixed_is_the_best_mean():
    D = analysis.load(SHIPPED)
    assert analysis.fixed_of(D["H"]) == D["H"].mean().idxmax()


def test_holm_is_monotone_and_bounded():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}
    h = stats.holm(p)
    assert h == pytest.approx({"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5}, abs=1e-15)


def test_bca_of_a_constant_has_no_width():
    r = stats.bca_macro({str(i): 1.0 for i in range(10)}, {str(i): "all" for i in range(10)},
                        {str(i): str(i) for i in range(10)})
    assert r["status"] == "zero_variance" and r["ci"] == [1.0, 1.0]


def test_missing_table_stops(tmp_path):
    with pytest.raises(SystemExit):
        results.load(tmp_path)

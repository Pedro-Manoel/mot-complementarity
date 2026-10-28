"""Replicate seeds and the determinism sample."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pedestrian_catalog import replicates as RP
from pedestrian_catalog.schema import SpecError

ROOT = Path(__file__).resolve().parents[1]

LINES = [{"dataset": "MOT20", "sequence_id": "MOT20-05", "eligible": True},
         {"dataset": "MOT20", "sequence_id": "MOT20-01", "eligible": True},
         {"dataset": "MOT17", "sequence_id": "MOT17-09", "eligible": "true"},
         {"dataset": "MOT17", "sequence_id": "MOT17-02", "eligible": False},
         {"dataset": "SOMPT22", "sequence_id": "SOMPT22-02", "eligible": True}]
Q = {"replicate_seeds": [42], "determinism_check": {"seed": 43, "sequences_per_dataset": 1}}


def test_sample_is_the_first_eligible_sequence_of_each_dataset():
    assert RP.determinism_sample(LINES, 1) == ["MOT17/MOT17-09", "MOT20/MOT20-01", "SOMPT22/SOMPT22-02"]
    assert RP.determinism_sample(LINES, 2) == ["MOT17/MOT17-09", "MOT20/MOT20-01", "MOT20/MOT20-05", "SOMPT22/SOMPT22-02"]


def test_seeds_by_sequence_of_the_design():
    s = RP.seeds_by_sequence(LINES, Q)
    assert s == {"MOT20/MOT20-05": [42], "MOT20/MOT20-01": [42, 43], "MOT17/MOT17-09": [42, 43],
                 "SOMPT22/SOMPT22-02": [42, 43]}
    assert RP.seeds_by_sequence(LINES, {"replicate_seeds": [42, 43]})["MOT20/MOT20-05"] == [42, 43]
    with pytest.raises(SpecError, match="is already a primary seed"):
        RP.seeds_by_sequence(LINES, {"replicate_seeds": [42, 43], "determinism_check": {"seed": 43}})
    seq = {"dataset": "MOT20", "sequence_id": "MOT20-05"}
    assert RP.seeds_for(seq, [7], s) == [42] and RP.seeds_for(seq, [7], None) == [7]


@pytest.mark.parametrize("experiment", [ROOT / "configs" / "experiment.yaml",
                                        ROOT / "tests" / "test_configs" / "experiment.yaml"],
                         ids=["paper", "frozen_test"])
def test_the_experiments_use_only_seed_42(experiment):
    q = yaml.safe_load(experiment.read_text(encoding="utf-8"))["quality"]
    assert q == {"replicate_seeds": [42]}
    assert {tuple(s) for s in RP.seeds_by_sequence(LINES, q).values()} == {(42,)}

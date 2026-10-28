"""Execution environment in the cell identity and in the run records."""
from __future__ import annotations

import pytest

from pedestrian_catalog import runtime as R


def test_cpu_and_gpu_environment():
    cpu = R.execution_environment("cpu")
    assert cpu["kind"] == "cpu" and cpu["name"]
    gpu = R.execution_environment("cuda:1")  # faked in conftest
    assert gpu == {"kind": "cuda", "name": "test gpu", "cuda": "0.0", "cudnn": 0}


def test_run_records_the_environment_of_each_stage_and_refuses_another(tmp_path):
    a, b = {"kind": "cpu", "name": "A"}, {"kind": "cuda", "name": "B"}
    assert R.run_environment(tmp_path, "detection") is None
    R.require_run_environment(tmp_path, "detection", a)
    R.require_run_environment(tmp_path, "detection", a)
    assert R.run_environment(tmp_path, "detection") == a
    with pytest.raises(R.EnvironmentMismatch, match="another --run-id"):
        R.require_run_environment(tmp_path, "detection", b)
    R.require_run_environment(tmp_path, "tracking", b)
    assert R.run_environment(tmp_path, "detection") == a and R.run_environment(tmp_path, "tracking") == b

"""Common test configuration."""
from __future__ import annotations

import pytest

from pedestrian_catalog import runtime as R


@pytest.fixture(autouse=True)
def _fake_gpu_description(request, monkeypatch):
    """Fakes the GPU description (part of the cell identity) so cuda devices need no GPU, except in `gpu` tests."""
    if request.node.get_closest_marker("gpu"):
        return
    monkeypatch.setattr(R, "_describe_gpu", lambda device: {"name": "test gpu", "cuda": "0.0", "cudnn": 0})

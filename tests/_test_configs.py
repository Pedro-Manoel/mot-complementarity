"""Frozen 27-config catalog and experiment for the integration tests, so their counts do not follow configs/."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIGS = Path(__file__).resolve().parent / "test_configs"


def copy(dest: Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("catalog.yaml", "experiment.yaml"):
        shutil.copyfile(TEST_CONFIGS / name, dest / name)
    shutil.copyfile(ROOT / "configs" / "trackers.yaml", dest / "trackers.yaml")
    return dest / "experiment.yaml"

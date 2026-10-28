# -*- coding: utf-8 -*-
"""Degenerate boxes are dropped from the person cache and counted, and kept in the raw output."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pedestrian_catalog import detection as D
from test_cache import ENV_CPU, Fake, _run, _sequence, _variant


class WithDegenerate(Fake):
    def detect(self, frame: np.ndarray):
        dets, info = super().detect(frame)
        if int(frame[0, 0, 0]) != 2:
            return dets, info
        deg = np.array([[20.0, 5.0, 20.0, 25.0, 0.9, 0.0]], np.float32)  # x2 == x1
        return np.vstack([dets, deg]), dict(info, raw=np.vstack([info["raw"], deg]))


def _read(cell, name):
    return pd.read_parquet(cell / name)


def test_degenerate_box_is_dropped_from_the_cache_counted_and_kept_in_the_raw_output(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    m = D.detect_sequence(seq, v, _run(tmp_path, WithDegenerate()))
    cell = D.cell_dir(tmp_path / "run", v, seq, 42)
    dets, raw, idx = _read(cell, "detections.parquet"), _read(cell, "raw_detections.parquet"), _read(cell, "frames_index.parquet")
    assert not ((dets.x2 <= dets.x1) | (dets.y2 <= dets.y1)).any()
    assert int(idx.loc[idx.frame_id == 2, "n_detections"].iloc[0]) == 1
    assert int(((raw.frame_id == 2) & (raw.x2 <= raw.x1)).sum()) == 1
    assert m["counts"]["n_degenerate_dropped"] == 1
    assert json.loads((cell / "manifest.json").read_text(encoding="utf-8"))["counts"]["n_degenerate_dropped"] == 1


def test_without_a_degenerate_box_the_cell_is_unchanged_and_counts_zero(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    m = D.detect_sequence(seq, v, _run(tmp_path, Fake()))
    cell = D.cell_dir(tmp_path / "run", v, seq, 42)
    dets = _read(cell, "detections.parquet")
    assert m["counts"]["n_degenerate_dropped"] == 0
    assert len(dets) == 4 and sorted(dets.frame_id) == [1, 2, 4, 5]
    np.testing.assert_allclose(dets.sort_values("frame_id").x1.to_numpy(), [2.0, 4.0, 8.0, 10.0])


def test_cell_identity_does_not_depend_on_the_rule(tmp_path):
    seq, v = _sequence(tmp_path), _variant()
    assert D.cell_dir(tmp_path / "run", v, seq, 42) == D.cell_dir(tmp_path / "run", v, seq, 42)
    m1 = D.detect_sequence(seq, v, _run(tmp_path, Fake()))
    assert m1["identity"]["replicate_seed"] == 42 and m1["identity"]["environment"] == ENV_CPU

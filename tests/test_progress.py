"""Progress lines in the terminal."""
from __future__ import annotations

import io

from pedestrian_catalog import progress as PG


def test_readable_durations():
    assert [PG.fmt_duration(x) for x in (None, 0, 42.4, 61, 3599, 3600, 9000)] == \
        ["?", "0 s", "42 s", "1 min 01 s", "59 min 59 s", "1 h 00 min", "2 h 30 min"]


def test_estimate_from_the_observed_rate_and_failure_marked():
    t = [100.0]
    out = io.StringIO()
    p = PG.Progress("detect yolo26n", 3, 3000, clock=lambda: t[0], out=out)
    assert p.eta() is None
    t[0] = 110.0
    line = p.advance("MOT20-01: 1000 frames", 1000)
    assert line.startswith("[detect yolo26n] 1/3 MOT20-01") and "elapsed 10 s, remaining ~20 s" in line
    t[0] = 130.0
    line = p.advance("MOT20-02: error", 1000, failed=True)
    assert "2/3 FAILED MOT20-02" in line and "remaining ~15 s" in line and p.failed == 1
    assert out.getvalue().count("\n") == 2


def test_resources_on_the_line_only_what_was_measured():
    assert PG.resources_text({"gpu_power_mean_w": 142.4, "gpu_util_mean_pct": 97.2, "torch_vram_peak_allocated_mb": 1126.4,
                              "cpu_process_occupancy_pct": 101.0}) == "GPU 142 W, 97 %, VRAM 1.1 GB, CPU 101 %"
    assert PG.resources_text({"gpu_power_mean_w": None, "cpu_process_occupancy_pct": 99.6}) == "CPU 100 %"
    assert PG.resources_text(None) == ""

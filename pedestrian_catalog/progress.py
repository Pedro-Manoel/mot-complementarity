"""Terminal progress: one line per finished cell, with the elapsed time and an estimate of what remains.

The estimate divides the remaining work (frames, or cells) by the rate observed in this call.
"""
from __future__ import annotations

import sys
import threading
import time

_LOCK = threading.Lock()


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "?"
    s = max(0, int(round(float(seconds))))
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def say(text: str, out=None) -> None:
    with _LOCK:
        print(text, file=out or sys.stdout, flush=True)


class Progress:
    def __init__(self, stage: str, total_cells: int, total_work: float, *, clock=time.perf_counter, out=None):
        self.stage, self.total, self.work_total = stage, int(total_cells), float(total_work)
        self.done, self.work_done, self.failed = 0, 0.0, 0
        self._clock, self._out = clock, out
        self._t0 = clock()

    def info(self, text: str) -> None:
        say(f"[{self.stage}] {text}", self._out)

    def eta(self):
        elapsed = self._clock() - self._t0
        if self.work_done <= 0:
            return None
        return elapsed / self.work_done * max(0.0, self.work_total - self.work_done)

    def advance(self, text: str, work: float = 0.0, failed: bool = False) -> str:
        self.done += 1
        self.work_done += float(work)
        self.failed += int(bool(failed))
        elapsed = self._clock() - self._t0
        line = (f"[{self.stage}] {self.done}/{self.total} {'FAILED ' if failed else ''}{text} | elapsed "
                f"{fmt_duration(elapsed)}, remaining ~{fmt_duration(self.eta())}")
        say(line, self._out)
        return line


def resources_text(resources: dict | None) -> str:
    r = resources or {}
    parts = []
    if r.get("gpu_power_mean_w") is not None:
        parts.append(f"GPU {r['gpu_power_mean_w']:.0f} W")
    if r.get("gpu_util_mean_pct") is not None:
        parts.append(f"{r['gpu_util_mean_pct']:.0f} %")
    if r.get("torch_vram_peak_allocated_mb") is not None:
        parts.append(f"VRAM {r['torch_vram_peak_allocated_mb'] / 1024:.1f} GB")
    if r.get("cpu_process_occupancy_pct") is not None:
        parts.append(f"CPU {r['cpu_process_occupancy_pct']:.0f} %")
    return ", ".join(parts)

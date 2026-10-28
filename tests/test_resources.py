"""Hardware resource monitor, with fake NVML, psutil and torch."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from pedestrian_catalog import resources as RS


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class _NVML:
    NVML_FI_DEV_POWER_INSTANT, NVML_TEMPERATURE_GPU, NVML_CLOCK_SM, NVML_CLOCK_MEM = 1, 0, 0, 1

    def __init__(self, clock, with_counter=True, fail=False):
        self.clock, self.with_counter, self.fail, self.open = clock, with_counter, fail, False

    def nvmlInit(self):
        self.open = True

    def nvmlShutdown(self):
        self.open = False

    def nvmlDeviceGetHandleByIndex(self, i):
        return f"gpu{i}"

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        if not self.with_counter:
            raise RuntimeError("no counter")
        return 5_000_000.0 + 150_000.0 * (self.clock.t - 100.0)  # 150 W in mJ/s

    def nvmlDeviceGetFieldValues(self, h, fields):
        return [NS(nvmlReturn=0, value=NS(uiVal=int(150_000)))]

    def nvmlDeviceGetPowerUsage(self, h):
        return 150_000

    def nvmlDeviceGetUtilizationRates(self, h):
        if self.fail:
            raise RuntimeError("reading failed")
        return NS(gpu=80, memory=40)

    def nvmlDeviceGetMemoryInfo(self, h):
        return NS(used=2 * 2 ** 30, total=16 * 2 ** 30)

    def nvmlDeviceGetTemperature(self, h, s):
        return 60

    def nvmlDeviceGetClockInfo(self, h, c):
        return 2000 if c == 0 else 800

    def nvmlDeviceGetPerformanceState(self, h):
        return 0

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, h):
        return 4

    def nvmlDeviceGetFanSpeed(self, h):
        return 35

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [1, 2]

    def nvmlDeviceGetName(self, h):
        return b"FAKE GPU"

    def nvmlSystemGetDriverVersion(self):
        return "1.0"

    def nvmlSystemGetNVMLVersion(self):
        return "12"

    def nvmlDeviceGetVbiosVersion(self, h):
        return "v"

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        return 285_000

    def nvmlDeviceGetCurrPcieLinkGeneration(self, h):
        return 4

    def nvmlDeviceGetCurrPcieLinkWidth(self, h):
        return 16


class _Proc:
    def __init__(self, clock, children=()):
        self.clock, self.kids, self.pid = clock, list(children), 1

    def cpu_times(self):
        return NS(user=2.0 * (self.clock.t - 100.0), system=0.5 * (self.clock.t - 100.0))

    def memory_info(self):
        return NS(rss=500 * 2 ** 20)

    def io_counters(self):
        return NS(read_bytes=1000.0 * self.clock.t, write_bytes=10.0 * self.clock.t, read_count=self.clock.t,
                  write_count=1.0)

    def num_ctx_switches(self):
        return NS(voluntary=3 * self.clock.t, involuntary=0)

    def children(self, recursive=True):
        return self.kids


class _Child:
    def __init__(self, clock, pid):
        self.clock, self.pid = clock, pid

    def cpu_times(self):
        return NS(user=1.0 * (self.clock.t - 100.0), system=0.0)

    def memory_info(self):
        return NS(rss=200 * 2 ** 20)


class _Psutil:
    def cpu_percent(self, _):
        return 25.0

    def virtual_memory(self):
        return NS(percent=50.0, total=64 * 2 ** 30)

    def disk_io_counters(self):
        return NS(read_bytes=7.0, write_bytes=9.0)

    def cpu_count(self, logical=True):
        return 24 if logical else 12

    def cpu_freq(self):
        return NS(max=4400.0)


class _Torch:
    class cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats(d):
            pass

        @staticmethod
        def memory_allocated(d):
            return 100 * 2 ** 20

        @staticmethod
        def max_memory_allocated(d):
            return 900 * 2 ** 20

        @staticmethod
        def max_memory_reserved(d):
            return 1200 * 2 ** 20


def _measure(clock, dur=10.0, samples=3, **kw):
    m = RS.ResourceMonitor(kw.pop("device", "cuda:0"), sample_hz=0, clock=clock, psutil_mod=_Psutil(),
                           torch_mod=_Torch, **kw)
    with m:
        for _ in range(samples):
            clock.t += dur / samples
            m.take_sample()
    return m


def test_gpu_energy_from_the_counter_and_cpu_from_the_difference_of_cpu_times():
    clk = _Clock()
    m = _measure(clk, nvml=_NVML(clk), proc=_Proc(clk))
    r = m.summary()
    assert r["gpu_energy_via"] == "nvmlDeviceGetTotalEnergyConsumption"
    assert r["gpu_energy_j"] == pytest.approx(1500.0) and r["gpu_power_mean_w"] == pytest.approx(150.0)
    assert r["duration_s"] == pytest.approx(10.0)
    assert r["cpu_process_time_s"] == pytest.approx(25.0) and r["cpu_process_occupancy_pct"] == pytest.approx(250.0)
    assert r["cpu_process_mean_pct"] == pytest.approx(250.0)
    assert (r["gpu_util_mean_pct"], r["gpu_util_max_pct"], r["gpu_mem_used_max_mb"]) == (80.0, 80.0, 2048.0)
    assert (r["gpu_pstate_min"], r["gpu_throttle_reasons_seen"], r["gpu_fan_max_pct"]) == (0.0, 4, 35.0)
    assert (r["gpu_compute_processes_start"], r["gpu_compute_processes_end"]) == (2, 2)
    assert (r["torch_vram_peak_allocated_mb"], r["torch_vram_peak_reserved_mb"]) == (900.0, 1200.0)
    assert r["io_read_bytes"] == pytest.approx(10000.0) and r["ctx_switches_voluntary"] == pytest.approx(30.0)
    p = r["provenance"]
    assert (p["gpu_name"], p["gpu_power_limit_w"], p["gpu_pcie_generation"], p["cpu_logical"]) == ("FAKE GPU", 285.0, 4, 24)
    assert len(m.samples()) == 5 and set(m.samples()[0]) == set(RS.SAMPLE_FIELDS)


def test_without_a_counter_the_energy_is_the_declared_integral_of_the_samples():
    clk = _Clock()
    r = _measure(clk, nvml=_NVML(clk, with_counter=False), proc=_Proc(clk)).summary()
    assert r["gpu_energy_via"] == "integral of the power samples" and r["gpu_energy_j"] == pytest.approx(1500.0)


def test_cpu_energy_is_never_estimated_and_the_summary_has_no_nan():
    clk = _Clock()
    r = _measure(clk, nvml=_NVML(clk), proc=_Proc(clk)).summary()
    assert r["cpu_energy_j"] is None and "never estimated from TDP" in r["cpu_energy_via"]
    json.dumps(r, allow_nan=False)


def test_on_cpu_nvml_is_not_opened_and_the_gpu_fields_stay_missing():
    clk = _Clock()
    nvml = _NVML(clk)
    r = _measure(clk, device="cpu", nvml=nvml, proc=_Proc(clk)).summary()
    assert nvml.open is False and r["gpu_energy_j"] is None and r["gpu_util_mean_pct"] is None
    assert r["torch_vram_peak_allocated_mb"] is None and r["cpu_process_time_s"] == pytest.approx(25.0)


def test_failing_reading_brings_nothing_down_and_is_counted():
    clk = _Clock()
    r = _measure(clk, nvml=_NVML(clk, fail=True), proc=_Proc(clk)).summary()
    assert r["read_errors"] > 0 and r["gpu_util_mean_pct"] is None and r["gpu_energy_j"] == pytest.approx(1500.0)


def test_children_add_cpu_and_ram():
    clk = _Clock()
    proc = _Proc(clk, children=[_Child(clk, 7), _Child(clk, 8)])
    r = _measure(clk, device="cpu", nvml=None, proc=proc, include_children=True).summary()
    assert r["cpu_children_time_s"] == pytest.approx(20.0) and r["ram_children_rss_peak_mb"] == pytest.approx(400.0)
    assert r["cpu_tree_time_s"] == pytest.approx(45.0) and r["ram_tree_rss_peak_mb"] == pytest.approx(900.0)
    assert r["cpu_children_mean_pct"] == pytest.approx(200.0)


def test_sampler_thread_samples_by_itself_and_stops():
    import time
    m = RS.ResourceMonitor("cpu", sample_hz=50, nvml=None, torch_mod=None)
    with m:
        time.sleep(0.2)
    assert m.summary()["n_samples"] >= 4 and m._thread is not None and not m._thread.is_alive()


def test_parts_combine_by_sum_maximum_and_weighted_mean():
    a = {"duration_s": 10.0, "n_samples": 2, "read_errors": 0, "gpu_energy_j": 100.0, "gpu_energy_via": "c",
         "gpu_util_mean_pct": 80.0, "gpu_util_max_pct": 90.0, "torch_vram_peak_allocated_mb": 500.0,
         "cpu_process_time_s": 5.0, "gpu_throttle_reasons_seen": 1, "ram_rss_start_mb": 10.0}
    b = {"duration_s": 30.0, "n_samples": 6, "read_errors": 1, "gpu_energy_j": 300.0, "gpu_energy_via": "c",
         "gpu_util_mean_pct": 40.0, "gpu_util_max_pct": 70.0, "torch_vram_peak_allocated_mb": 900.0,
         "cpu_process_time_s": 15.0, "gpu_throttle_reasons_seen": 4, "ram_rss_start_mb": 99.0}
    r = RS.combine([a, b])
    assert (r["duration_s"], r["gpu_energy_j"], r["cpu_process_time_s"], r["n_parts"]) == (40.0, 400.0, 20.0, 2)
    assert r["gpu_util_mean_pct"] == pytest.approx(50.0) and r["gpu_util_max_pct"] == 90.0
    assert r["torch_vram_peak_allocated_mb"] == 900.0 and r["gpu_throttle_reasons_seen"] == 5
    assert r["gpu_power_mean_w"] == pytest.approx(10.0) and r["ram_rss_start_mb"] == 10.0
    assert RS.combine([a, {**b, "gpu_energy_j": None}])["gpu_energy_j"] is None
    assert RS.per_frame(r, 400) == {"gpu_energy_j_per_frame": 1.0, "gpu_energy_mj_per_frame": 1000.0,
                                    "cpu_process_ms_per_frame": 50.0}


class _Ended(_Proc):
    def __init__(self, clock, end):
        super().__init__(clock)
        self.end = end

    def cpu_times(self):
        if self.clock.t > self.end:
            raise RuntimeError("NoSuchProcess")
        return super().cpu_times()

    def io_counters(self):
        if self.clock.t > self.end:
            raise RuntimeError("NoSuchProcess")
        return super().io_counters()


def test_child_process_counts_from_its_start_and_keeps_the_last_reading_after_it_ends():
    clk = _Clock()
    clk.t = 104.0  # the child started 4 s before the monitor
    m = RS.ResourceMonitor("cpu", sample_hz=0, clock=clk, psutil_mod=_Psutil(), torch_mod=None, nvml=None,
                           proc=_Ended(clk, end=108.0), from_process_start=True)
    with m:
        for _ in range(2):
            clk.t += 2.0
            m.take_sample()
        clk.t += 5.0
    r = m.summary()
    assert r["cpu_process_time_s"] == pytest.approx(20.0)  # 2.5 s of CPU per s for the 8 s the child ran
    assert r["io_read_bytes"] == pytest.approx(108000.0)

"""Hardware resources of a stage (GPU energy, power and VRAM; CPU, RAM and I/O), sampled in the same execution.

GPU energy comes from the NVML counter (power samples are integrated only without it); VRAM comes from the torch
allocator, since NVML has no per-process memory under WDDM. Nothing here raises: unreadable values are `None`.
"""
from __future__ import annotations

import math
import platform
import threading
import warnings
from pathlib import Path
from time import perf_counter

AUTO = object()          # discover the module; an explicit None means there is none
NAN = float("nan")
RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")     # Linux; absent on Windows
SAMPLE_FIELDS = ("t_s", "gpu_power_w", "gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mb", "gpu_temperature_c",
                 "gpu_clock_sm_mhz", "gpu_clock_mem_mhz", "gpu_pstate", "gpu_throttle_reasons", "gpu_fan_pct",
                 "cpu_process_pct", "cpu_children_pct", "cpu_system_pct", "ram_rss_mb", "ram_children_rss_mb",
                 "ram_system_used_pct")
CPU_ENERGY_NOT_MEASURED = "not measured: RAPL is unavailable on this platform; never estimated from TDP"


def _default_nvml():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # the old pynvml warns that it became nvidia-ml-py
            import pynvml
        return pynvml
    except Exception:
        return None


def _default_torch():
    import sys
    return sys.modules.get("torch")                  # never imports torch just to measure


def _default_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def device_index(device) -> int | None:
    d = str(device)
    if not d.startswith("cuda"):
        return None
    return int(d.split(":")[1]) if ":" in d else 0


def _finite(xs) -> list[float]:
    out = []
    for x in xs:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _mean(xs):
    v = _finite(xs)
    return sum(v) / len(v) if v else None


def _max(xs):
    v = _finite(xs)
    return max(v) if v else None


def _min(xs):
    v = _finite(xs)
    return min(v) if v else None


def _p95(xs):
    v = sorted(_finite(xs))
    if not v:
        return None
    k = (len(v) - 1) * 0.95
    a, b = int(math.floor(k)), int(math.ceil(k))
    return v[a] + (v[b] - v[a]) * (k - a)


def trapezoid_integral(ts, ys):
    pairs = [(float(t), float(y)) for t, y in zip(ts, ys)
             if t is not None and y is not None and math.isfinite(float(t)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    return sum(0.5 * (y0 + y1) * (t1 - t0) for (t0, y0), (t1, y1) in zip(pairs, pairs[1:]))


def json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


class _Sampler(threading.Thread):
    def __init__(self, read_fn, period_s: float):
        super().__init__(daemon=True, name="pcat-resources")
        self._read_fn, self._period, self._stop_event = read_fn, float(period_s), threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._read_fn()
            except Exception:
                pass
            self._stop_event.wait(self._period)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5.0)


class ResourceMonitor:
    """`with ResourceMonitor("cuda:0") as m: ...`, then `m.summary()` and `m.samples()`."""

    def __init__(self, device="cpu", *, sample_hz: float = 4.0, include_children: bool = False, from_process_start: bool = False,
                 nvml=AUTO, psutil_mod=AUTO, torch_mod=AUTO, proc=None, clock=perf_counter):
        self.device, self.sample_hz, self.include_children = str(device), float(sample_hz), bool(include_children)
        # a freshly created child counts from its start; its last reading holds after it ends (up to one interval short)
        self.from_process_start = bool(from_process_start)
        self._cpu_seen, self._io_seen = (NAN, NAN), {}
        self._nvml = _default_nvml() if nvml is AUTO else nvml
        self._psutil = _default_psutil() if psutil_mod is AUTO else psutil_mod
        self._torch = _default_torch() if torch_mod is AUTO else torch_mod
        self._proc, self._clock = proc, clock
        self._h, self._nvml_on = None, False
        self._samples: list[dict] = []
        self._errors = 0
        self._children: dict[int, float] = {}          # pid: last user + system CPU time seen
        self._last_cpu = None                          # (t, process CPU, children CPU)
        self._thread = None
        self._summary = None
        self.provenance: dict = {}

    def _call(self, f, default=NAN):
        try:
            return f()
        except Exception:
            self._errors += 1
            return default

    def _open_nvml(self) -> None:
        idx = device_index(self.device)
        if self._nvml is None or idx is None:
            return
        try:
            self._nvml.nvmlInit()
            self._nvml_on = True
            self._h = self._nvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            self._h = None

    def _close_nvml(self) -> None:
        if self._nvml_on:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_on = False

    def _energy_mj(self) -> float:
        if self._h is None:
            return NAN
        try:
            return float(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h))
        except Exception:
            return NAN

    def _power_w(self) -> tuple[float, str | None]:
        if self._h is None:
            return NAN, None
        n = self._nvml
        try:
            v = n.nvmlDeviceGetFieldValues(self._h, [n.NVML_FI_DEV_POWER_INSTANT])[0]
            if int(getattr(v, "nvmlReturn", 0)) == 0:
                return float(v.value.uiVal) / 1000.0, "NVML_FI_DEV_POWER_INSTANT"
        except Exception:
            pass
        try:
            return float(n.nvmlDeviceGetPowerUsage(self._h)) / 1000.0, "nvmlDeviceGetPowerUsage"
        except Exception:
            return NAN, None

    def _gpu_processes(self):
        if self._h is None:
            return None
        try:
            return int(len(self._nvml.nvmlDeviceGetComputeRunningProcesses(self._h)))
        except Exception:
            return None

    def _process_cpu(self) -> tuple[float, float]:
        if self._proc is None:
            return NAN, NAN
        try:
            t = self._proc.cpu_times()
            self._cpu_seen = (float(t.user), float(t.system))
        except Exception:
            pass
        return self._cpu_seen

    def _children_cpu(self) -> tuple[float, float]:
        """(CPU time of all children seen so far, summed RSS of the live ones)."""
        if not self.include_children or self._proc is None:
            return NAN, NAN
        rss = 0.0
        try:
            kids = self._proc.children(recursive=True)
        except Exception:
            return sum(self._children.values()), NAN
        for f in kids:
            try:
                t = f.cpu_times()
                self._children[int(f.pid)] = float(t.user) + float(t.system)
                rss += float(f.memory_info().rss) / 2 ** 20
            except Exception:
                continue
        return sum(self._children.values()), rss

    def _io_counters(self) -> dict:
        d = {"read_bytes": NAN, "write_bytes": NAN, "read_count": NAN, "write_count": NAN,
             "ctx_voluntary": NAN, "ctx_involuntary": NAN, "disk_read_bytes": NAN, "disk_write_bytes": NAN}
        if self._proc is not None:
            try:
                io = self._proc.io_counters()
                self._io_seen.update(read_bytes=float(io.read_bytes), write_bytes=float(io.write_bytes),
                                     read_count=float(io.read_count), write_count=float(io.write_count))
            except Exception:
                pass
            try:
                c = self._proc.num_ctx_switches()
                self._io_seen.update(ctx_voluntary=float(c.voluntary), ctx_involuntary=float(c.involuntary))
            except Exception:
                pass
            d.update(self._io_seen)
        if self._psutil is not None:
            try:
                disk = self._psutil.disk_io_counters()
                d.update(disk_read_bytes=float(disk.read_bytes), disk_write_bytes=float(disk.write_bytes))
            except Exception:
                pass
        return d

    def _torch_mb(self, fn_name: str):
        t = self._torch
        if t is None or device_index(self.device) is None:
            return None
        try:
            if not t.cuda.is_available():
                return None
            return float(getattr(t.cuda, fn_name)(self.device)) / 2 ** 20
        except Exception:
            return None

    def take_sample(self) -> dict:
        now = float(self._clock())
        a = {c: NAN for c in SAMPLE_FIELDS}
        a["t_s"] = now
        if self._h is not None:
            n, h = self._nvml, self._h
            a["gpu_power_w"] = self._power_w()[0]
            u = self._call(lambda: n.nvmlDeviceGetUtilizationRates(h), None)
            if u is not None:
                a["gpu_util_pct"], a["gpu_mem_util_pct"] = float(u.gpu), float(u.memory)
            m = self._call(lambda: n.nvmlDeviceGetMemoryInfo(h), None)
            if m is not None:
                a["gpu_mem_used_mb"] = float(m.used) / 2 ** 20
            a["gpu_temperature_c"] = float(self._call(lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)))
            a["gpu_clock_sm_mhz"] = float(self._call(lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)))
            a["gpu_clock_mem_mhz"] = float(self._call(lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)))
            a["gpu_pstate"] = float(self._call(lambda: n.nvmlDeviceGetPerformanceState(h)))
            a["gpu_throttle_reasons"] = float(self._call(lambda: n.nvmlDeviceGetCurrentClocksThrottleReasons(h)))
            a["gpu_fan_pct"] = float(self._call(lambda: n.nvmlDeviceGetFanSpeed(h)))
        cpu_u, cpu_s = self._process_cpu()
        self._io_counters()                     # keeps the last valid I/O: a child may end before the stage
        child_cpu, child_rss = self._children_cpu()
        if self._last_cpu is not None:
            t0, c0, f0 = self._last_cpu
            dt = now - t0
            if dt > 0:
                a["cpu_process_pct"] = 100.0 * ((cpu_u + cpu_s) - c0) / dt
                a["cpu_children_pct"] = 100.0 * (child_cpu - f0) / dt
        self._last_cpu = (now, cpu_u + cpu_s, child_cpu)
        a["ram_children_rss_mb"] = child_rss
        if self._proc is not None:
            a["ram_rss_mb"] = float(self._call(lambda: self._proc.memory_info().rss)) / 2 ** 20
        if self._psutil is not None:
            a["cpu_system_pct"] = float(self._call(lambda: self._psutil.cpu_percent(None)))
            a["ram_system_used_pct"] = float(self._call(lambda: self._psutil.virtual_memory().percent))
        self._samples.append(a)
        return a

    def _build_provenance(self, cost_ms) -> dict:
        p = {"device": self.device, "platform": platform.platform(), "python": platform.python_version(),
             "sample_hz": self.sample_hz, "sample_cost_ms": cost_ms, "include_children": self.include_children,
             "from_process_start": self.from_process_start,
             "measured_pid": getattr(self._proc, "pid", None),
             "gpu_energy_counter": "nvmlDeviceGetTotalEnergyConsumption", "gpu_power_via": None,
             "vram_via": ("torch.cuda.max_memory_allocated/max_memory_reserved"
                          if self._torch is not None and device_index(self.device) is not None else None),
             "cpu_ram_io_via": "psutil" if self._psutil is not None else None,
             "cpu_energy_via": "RAPL (/sys/class/powercap)" if RAPL.exists() else CPU_ENERGY_NOT_MEASURED,
             "gpu_name": None, "gpu_driver": None, "nvml_version": None, "gpu_vbios": None, "gpu_power_limit_w": None,
             "gpu_mem_total_mb": None, "gpu_pcie_generation": None, "gpu_pcie_width": None,
             "cpu_logical": None, "cpu_physical": None, "cpu_freq_max_mhz": None, "ram_total_mb": None}
        if self._psutil is not None:
            for k, f in (("cpu_logical", lambda: int(self._psutil.cpu_count(logical=True))),
                         ("cpu_physical", lambda: int(self._psutil.cpu_count(logical=False))),
                         ("cpu_freq_max_mhz", lambda: float(self._psutil.cpu_freq().max)),
                         ("ram_total_mb", lambda: float(self._psutil.virtual_memory().total) / 2 ** 20)):
                try:
                    p[k] = f()
                except Exception:
                    pass
        if self._h is not None:
            n, h = self._nvml, self._h
            for k, f in (("gpu_name", lambda: n.nvmlDeviceGetName(h)), ("gpu_driver", lambda: n.nvmlSystemGetDriverVersion()),
                         ("nvml_version", lambda: n.nvmlSystemGetNVMLVersion()), ("gpu_vbios", lambda: n.nvmlDeviceGetVbiosVersion(h)),
                         ("gpu_power_limit_w", lambda: float(n.nvmlDeviceGetEnforcedPowerLimit(h)) / 1000.0),
                         ("gpu_mem_total_mb", lambda: float(n.nvmlDeviceGetMemoryInfo(h).total) / 2 ** 20),
                         ("gpu_pcie_generation", lambda: int(n.nvmlDeviceGetCurrPcieLinkGeneration(h))),
                         ("gpu_pcie_width", lambda: int(n.nvmlDeviceGetCurrPcieLinkWidth(h)))):
                try:
                    x = f()
                    p[k] = x.decode() if isinstance(x, bytes) else x
                except Exception:
                    pass
            p["gpu_power_via"] = self._power_w()[1]
        return p

    def __enter__(self) -> "ResourceMonitor":
        try:
            self._open_nvml()
            if self._proc is None and self._psutil is not None:
                try:
                    self._proc = self._psutil.Process()
                except Exception:
                    self._proc = None
            if self._psutil is not None:
                self._call(lambda: self._psutil.cpu_percent(None))      # primes psutil: the first call returns 0.0
            if self._torch is not None and device_index(self.device) is not None:
                try:
                    if self._torch.cuda.is_available():
                        self._torch.cuda.reset_peak_memory_stats(self.device)
                except Exception:
                    pass
            t0 = self._clock()
            first = self.take_sample()
            self.provenance = self._build_provenance(1000.0 * (float(self._clock()) - float(t0)))
            io0 = self._io_counters()
            if self.from_process_start:
                io0.update({k: 0.0 for k in ("read_bytes", "write_bytes", "read_count", "write_count",
                                             "ctx_voluntary", "ctx_involuntary")})
            self._start = {"gpu_mem_used_mb": first["gpu_mem_used_mb"], "gpu_temperature_c": first["gpu_temperature_c"],
                           "ram_rss_mb": first["ram_rss_mb"], "torch_allocated_mb": self._torch_mb("memory_allocated"),
                           "gpu_processes": self._gpu_processes(), "io": io0}
            self._cpu0 = (0.0, 0.0) if self.from_process_start else self._process_cpu()
            self._t0, self._e0 = float(self._clock()), self._energy_mj()
            if self.sample_hz > 0:
                self._thread = _Sampler(self.take_sample, 1.0 / self.sample_hz)
                self._thread.start()
        except Exception:
            self._errors += 1
        return self

    def __exit__(self, *exc) -> bool:
        try:
            if self._thread is not None:
                self._thread.stop()
            t1, e1 = float(self._clock()), self._energy_mj()
            self.take_sample()
            self._summary = self._assemble(t1, e1, self._process_cpu(), self._io_counters())
        except Exception:
            self._errors += 1
            self._summary = {"error": "the resource summary could not be assembled", "read_errors": self._errors}
        finally:
            self._close_nvml()
        return False

    def _assemble(self, t1, e1, cpu1, io1) -> dict:
        am = list(self._samples)
        dur = t1 - self._t0
        col = lambda k: [a[k] for a in am]                                  # noqa: E731
        if math.isfinite(e1) and math.isfinite(self._e0):
            energy, via = (e1 - self._e0) / 1000.0, "nvmlDeviceGetTotalEnergyConsumption"
        else:
            energy = trapezoid_integral(col("t_s"), col("gpu_power_w"))
            via = "integral of the power samples" if energy is not None else None
        u0, s0 = self._cpu0
        u1, s1 = cpu1
        cpu_u = u1 - u0 if math.isfinite(u1) and math.isfinite(u0) else None
        cpu_s = s1 - s0 if math.isfinite(s1) and math.isfinite(s0) else None
        cpu_t = cpu_u + cpu_s if cpu_u is not None and cpu_s is not None else None
        io0 = self._start["io"]
        delta = {k: (io1[k] - io0[k] if math.isfinite(io1[k]) and math.isfinite(io0[k]) else None) for k in io0}
        throttle = 0
        seen = False
        for x in _finite(col("gpu_throttle_reasons")):
            throttle |= int(x)
            seen = True
        kids = self._children
        r = {
            "duration_s": dur, "n_samples": len(am), "read_errors": int(self._errors),
            "gpu_energy_j": energy, "gpu_energy_via": via,
            "gpu_power_mean_w": (energy / dur if energy is not None and dur > 0 else None),
            "gpu_power_samples_mean_w": _mean(col("gpu_power_w")), "gpu_power_samples_p95_w": _p95(col("gpu_power_w")),
            "gpu_power_samples_max_w": _max(col("gpu_power_w")),
            "gpu_util_mean_pct": _mean(col("gpu_util_pct")), "gpu_util_p95_pct": _p95(col("gpu_util_pct")),
            "gpu_util_max_pct": _max(col("gpu_util_pct")), "gpu_mem_util_mean_pct": _mean(col("gpu_mem_util_pct")),
            "gpu_mem_used_start_mb": self._start["gpu_mem_used_mb"], "gpu_mem_used_max_mb": _max(col("gpu_mem_used_mb")),
            "gpu_temperature_start_c": self._start["gpu_temperature_c"],
            "gpu_temperature_max_c": _max(col("gpu_temperature_c")),
            "gpu_clock_sm_mean_mhz": _mean(col("gpu_clock_sm_mhz")), "gpu_clock_sm_max_mhz": _max(col("gpu_clock_sm_mhz")),
            "gpu_clock_mem_mean_mhz": _mean(col("gpu_clock_mem_mhz")),
            "gpu_pstate_min": _min(col("gpu_pstate")), "gpu_pstate_max": _max(col("gpu_pstate")),
            "gpu_throttle_reasons_seen": throttle if seen else None, "gpu_fan_max_pct": _max(col("gpu_fan_pct")),
            "gpu_compute_processes_start": self._start["gpu_processes"], "gpu_compute_processes_end": self._gpu_processes(),
            "torch_vram_start_allocated_mb": self._start["torch_allocated_mb"],
            "torch_vram_peak_allocated_mb": self._torch_mb("max_memory_allocated"),
            "torch_vram_peak_reserved_mb": self._torch_mb("max_memory_reserved"),
            "cpu_process_time_s": cpu_t, "cpu_process_user_s": cpu_u, "cpu_process_system_s": cpu_s,
            "cpu_process_occupancy_pct": (100.0 * cpu_t / dur if cpu_t is not None and dur > 0 else None),
            "cpu_process_mean_pct": _mean(col("cpu_process_pct")), "cpu_process_max_pct": _max(col("cpu_process_pct")),
            "cpu_children_time_s": (sum(kids.values()) if self.include_children else None),
            "cpu_children_mean_pct": _mean(col("cpu_children_pct")), "cpu_children_max_pct": _max(col("cpu_children_pct")),
            "cpu_tree_time_s": ((cpu_t or 0.0) + sum(kids.values()) if self.include_children and cpu_t is not None else None),
            "cpu_system_mean_pct": _mean(col("cpu_system_pct")), "cpu_system_max_pct": _max(col("cpu_system_pct")),
            "cpu_energy_j": None, "cpu_energy_via": self.provenance.get("cpu_energy_via", CPU_ENERGY_NOT_MEASURED),
            "ram_rss_start_mb": self._start["ram_rss_mb"], "ram_rss_peak_mb": _max(col("ram_rss_mb")),
            "ram_children_rss_peak_mb": _max(col("ram_children_rss_mb")),
            "ram_tree_rss_peak_mb": (_max((a["ram_rss_mb"] if math.isfinite(a["ram_rss_mb"]) else 0.0)
                                          + (a["ram_children_rss_mb"] if math.isfinite(a["ram_children_rss_mb"]) else 0.0)
                                          for a in am) if self.include_children else None),
            "ram_system_used_max_pct": _max(col("ram_system_used_pct")),
            "io_read_bytes": delta["read_bytes"], "io_write_bytes": delta["write_bytes"],
            "io_read_count": delta["read_count"], "io_write_count": delta["write_count"],
            "ctx_switches_voluntary": delta["ctx_voluntary"], "ctx_switches_involuntary": delta["ctx_involuntary"],
            "disk_system_read_bytes": delta["disk_read_bytes"], "disk_system_write_bytes": delta["disk_write_bytes"],
            "provenance": dict(self.provenance),
        }
        return json_safe(r)

    def summary(self) -> dict:
        if self._summary is None:
            raise RuntimeError("summary() is only available after leaving the `with` block")
        return dict(self._summary)

    def samples(self) -> list[dict]:
        return [json_safe(dict(a)) for a in self._samples]


_SUMS = ("duration_s", "n_samples", "read_errors", "gpu_energy_j", "cpu_process_time_s", "cpu_process_user_s",
         "cpu_process_system_s", "cpu_children_time_s", "cpu_tree_time_s", "io_read_bytes", "io_write_bytes", "io_read_count",
         "io_write_count", "ctx_switches_voluntary", "ctx_switches_involuntary", "disk_system_read_bytes",
         "disk_system_write_bytes")
_MAXIMA = ("gpu_power_samples_max_w", "gpu_power_samples_p95_w", "gpu_util_max_pct", "gpu_util_p95_pct",
           "gpu_mem_used_max_mb", "gpu_temperature_max_c", "gpu_clock_sm_max_mhz", "gpu_pstate_max", "gpu_fan_max_pct",
           "gpu_compute_processes_start", "gpu_compute_processes_end", "torch_vram_peak_allocated_mb",
           "torch_vram_peak_reserved_mb", "cpu_process_max_pct", "cpu_children_max_pct", "cpu_system_max_pct",
           "ram_rss_peak_mb", "ram_children_rss_peak_mb", "ram_tree_rss_peak_mb", "ram_system_used_max_pct")
_WEIGHTED_MEANS = ("gpu_power_samples_mean_w", "gpu_util_mean_pct", "gpu_mem_util_mean_pct", "gpu_clock_sm_mean_mhz",
                   "gpu_clock_mem_mean_mhz", "cpu_process_mean_pct", "cpu_children_mean_pct", "cpu_system_mean_pct")


def combine(summaries: list[dict]) -> dict:
    """One summary from several parts: sums, maxima and duration-weighted means; a sum with a missing part is None."""
    rs = [r for r in summaries if r and "error" not in r]
    if not rs:
        return {"error": "no part with measured resources", "n_parts": len(summaries)}
    out: dict = {"n_parts": len(summaries), "n_parts_measured": len(rs)}
    for k in _SUMS:
        v = [r.get(k) for r in rs]
        out[k] = None if any(x is None for x in v) else sum(v)
    for k in _MAXIMA:
        out[k] = _max(r.get(k) for r in rs)
    weights = [float(r.get("duration_s") or 0.0) for r in rs]
    for k in _WEIGHTED_MEANS:
        pairs = [(w, float(r[k])) for w, r in zip(weights, rs) if r.get(k) is not None and w > 0]
        total = sum(w for w, _ in pairs)
        out[k] = sum(w * x for w, x in pairs) / total if total > 0 else None
    out["gpu_pstate_min"] = _min(r.get("gpu_pstate_min") for r in rs)
    seen = [r.get("gpu_throttle_reasons_seen") for r in rs if r.get("gpu_throttle_reasons_seen") is not None]
    out["gpu_throttle_reasons_seen"] = None
    for x in seen:
        out["gpu_throttle_reasons_seen"] = (out["gpu_throttle_reasons_seen"] or 0) | int(x)
    vias = {r.get("gpu_energy_via") for r in rs}
    out["gpu_energy_via"] = vias.pop() if len(vias) == 1 else "mixed: " + ", ".join(sorted(str(v) for v in vias))
    dur = out["duration_s"]
    out["gpu_power_mean_w"] = out["gpu_energy_j"] / dur if out["gpu_energy_j"] is not None and dur else None
    out["cpu_process_occupancy_pct"] = (100.0 * out["cpu_process_time_s"] / dur
                                        if out["cpu_process_time_s"] is not None and dur else None)
    for k in ("gpu_mem_used_start_mb", "gpu_temperature_start_c", "torch_vram_start_allocated_mb", "ram_rss_start_mb"):
        out[k] = rs[0].get(k)
    out["cpu_energy_j"] = None
    out["cpu_energy_via"] = rs[0].get("cpu_energy_via")
    out["provenance"] = rs[0].get("provenance")
    return json_safe(out)


def per_frame(resource_summary: dict, n_frames: int) -> dict:
    n = max(int(n_frames), 1)
    e, c = resource_summary.get("gpu_energy_j"), resource_summary.get("cpu_process_time_s")
    return {"gpu_energy_j_per_frame": None if e is None else e / n,
            "gpu_energy_mj_per_frame": None if e is None else 1000.0 * e / n,
            "cpu_process_ms_per_frame": None if c is None else 1000.0 * c / n}


__all__ = ["ResourceMonitor", "combine", "per_frame", "json_safe", "trapezoid_integral", "device_index",
           "SAMPLE_FIELDS", "CPU_ENERGY_NOT_MEASURED", "AUTO"]

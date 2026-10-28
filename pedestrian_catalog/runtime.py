"""The GPU lock that keeps two jobs of this package from running at once, and the execution environment of a run.

The lock is a file holding the owner's pid; a lock whose pid no longer exists is taken over.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .provenance import write_json_atomic
from .schema import SpecError

GPU_LOCK_NAME = ".gpu.lock"
RUN_ENVIRONMENT_FILE = "environment.json"


class GpuBusy(RuntimeError):
    """Another command of this package is already using the GPU."""


def _alive(pid) -> bool:
    import psutil
    return isinstance(pid, int) and pid > 0 and psutil.pid_exists(pid)


@contextmanager
def gpu_lock(path, command: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                owner = {}
            if _alive(owner.get("pid")):
                raise GpuBusy(f"GPU busy with another command of this package (pid {owner.get('pid')}: "
                              f"{owner.get('command')}); wait for it to finish") from None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "command": command,
                       "since_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, fh)
        break
    else:
        raise GpuBusy("could not acquire the GPU lock")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class EnvironmentMismatch(SpecError):
    """A stage was requested in an environment different from the one already recorded for it."""


def _describe_gpu(device: str) -> dict:
    import torch
    idx = torch.device(device).index or 0
    return {"name": torch.cuda.get_device_name(idx), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def execution_environment(device: str) -> dict:
    """Device kind and model (plus CUDA and cuDNN on the GPU); the board index is left out on purpose."""
    if str(device).startswith("cuda"):
        return {"kind": "cuda", **_describe_gpu(device)}
    import platform
    return {"kind": "cpu", "name": platform.processor() or platform.machine()}


def run_environment(run_dir, stage: str) -> dict | None:
    p = Path(run_dir) / RUN_ENVIRONMENT_FILE
    return json.loads(p.read_text(encoding="utf-8")).get(stage) if p.is_file() else None


def require_run_environment(run_dir, stage: str, environment: dict) -> dict:
    """The first call records the environment of the stage; resumptions must request the same one."""
    p = Path(run_dir) / RUN_ENVIRONMENT_FILE
    reg = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    if stage in reg and reg[stage] != environment:
        raise EnvironmentMismatch(f"run {Path(run_dir).name} did {stage} on {reg[stage]} and this call requests "
                                  f"{environment}: mixing environments in one run would change numbers and timings; "
                                  "use another --run-id")
    if stage not in reg:
        reg[stage] = environment
        write_json_atomic(p, reg)
    return reg

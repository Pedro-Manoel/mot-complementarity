"""Hashes of files and checkpoints, and atomic writes.

Exportable manifests carry file names and hashes, never absolute paths of the machine.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .schema import SpecError

_CHUNK = 1 << 20


class CheckpointMismatch(SpecError):
    """The weights on disk are not the ones `plan` recorded."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bytes_atomic(dest: str | Path, data: bytes) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)
    return dest


def write_text_atomic(dest: str | Path, text: str) -> Path:
    return write_bytes_atomic(dest, text.encode("utf-8"))


def write_json_atomic(dest: str | Path, obj: Any) -> Path:
    return write_text_atomic(dest, json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n")


def _hf_snapshot(hf_cache_dir: Path, repo: str) -> tuple[str, Path]:
    base = hf_cache_dir / ("models--" + repo.replace("/", "--"))
    ref = base / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
    else:
        snaps = sorted(p.name for p in (base / "snapshots").glob("*") if p.is_dir()) if (base / "snapshots").is_dir() else []
        if len(snaps) != 1:
            raise FileNotFoundError(f"checkpoint {repo}: local snapshot not determined in the Hugging Face cache "
                                    f"({len(snaps)} found); nothing is downloaded")
        revision = snaps[0]
    snap = base / "snapshots" / revision
    if not snap.is_dir():
        raise FileNotFoundError(f"checkpoint {repo}: snapshot {revision} missing from the Hugging Face cache; "
                                "nothing is downloaded")
    return revision, snap


def resolve_checkpoint(detector: dict, paths: dict) -> dict:
    """Locates and hashes a detector's checkpoint without loading it; nothing is downloaded."""
    backend, ck, model_name = detector["backend"], detector["checkpoint"], detector["name"]
    if backend == "ultralytics":
        p = Path(paths["weights_dir"]) / ck["file"]
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint {model_name}: {ck['file']} not found in weights_dir")
        return {"kind": "file", "file": ck["file"], "sha256": sha256_file(p), "size": p.stat().st_size}
    if backend == "dfine":
        revision, snap = _hf_snapshot(Path(paths["hf_cache_dir"]), ck["hf_repo"])
        files, combined = _snapshot_hash(snap)
        if not files:
            raise FileNotFoundError(f"checkpoint {model_name}: snapshot {revision} is empty")
        return {"kind": "hf_snapshot", "hf_repo": ck["hf_repo"], "revision": revision, "files": files,
                "sha256": combined}
    if backend == "torchvision":
        p = Path(paths["torch_hub_dir"]) / "checkpoints" / ck["file"]
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint {model_name}: {ck['file']} not found in the torch cache; "
                                    "nothing is downloaded")
        return {"kind": "torch_hub", "torchvision_weights": ck["torchvision_weights"], "file": ck["file"],
                "sha256": sha256_file(p), "size": p.stat().st_size}
    raise ValueError(f"unknown backend: {backend!r}")


def _snapshot_hash(snap: Path) -> tuple[dict, str]:
    files = {f.relative_to(snap).as_posix(): sha256_file(f) for f in sorted(snap.rglob("*")) if f.is_file()}
    return files, sha256_bytes("\n".join(f"{k},{v}" for k, v in files.items()).encode("utf-8"))


def _checkpoint_file(detector: dict, paths: dict) -> Path:
    if detector["backend"] == "ultralytics":
        return Path(paths["weights_dir"]) / detector["checkpoint"]["file"]
    return Path(paths["torch_hub_dir"]) / "checkpoints" / detector["checkpoint"]["file"]


def verify_checkpoint(detector: dict, resolved: dict | None, paths: dict) -> dict:
    """Checks without loading that the weights on disk are those of the resolved catalog.

    For a Hugging Face snapshot the recorded revision is used, not the one `refs/main` points to today.
    """
    model_name, backend = detector["name"], detector["backend"]
    if not resolved or not resolved.get("sha256"):
        raise CheckpointMismatch(f"checkpoint {model_name}: no checkpoint resolved by plan; run plan before executing")
    if backend in ("ultralytics", "torchvision"):
        p = _checkpoint_file(detector, paths)
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint {model_name}: {p.name} not found; nothing is downloaded")
        current = sha256_file(p)
        if current != resolved["sha256"]:
            raise CheckpointMismatch(f"checkpoint {model_name}: {p.name} changed after plan (sha256 {current[:12]} on "
                                     f"disk, {resolved['sha256'][:12]} recorded); run plan again or restore the "
                                     "weights")
        return {"path": p, "sha256": current}
    if backend == "dfine":
        rev = resolved.get("revision")
        base = Path(paths["hf_cache_dir"]) / ("models--" + detector["checkpoint"]["hf_repo"].replace("/", "--"))
        snap = base / "snapshots" / str(rev)
        if not rev or not snap.is_dir():
            raise FileNotFoundError(f"checkpoint {model_name}: revision {rev} recorded by plan is not in the local "
                                    "Hugging Face cache; nothing is downloaded")
        _files, combined = _snapshot_hash(snap)
        if combined != resolved["sha256"]:
            raise CheckpointMismatch(f"checkpoint {model_name}: the files of revision {rev} changed after plan "
                                     f"(sha256 {combined[:12]} on disk, {resolved['sha256'][:12]} recorded)")
        return {"revision": rev, "snapshot": snap, "sha256": combined}
    raise ValueError(f"unknown backend: {backend!r}")


def verify_weights_file(asset: dict, paths: dict) -> Path:
    p = Path(paths["weights_dir"]) / asset["file"]
    if not p.is_file():
        raise FileNotFoundError(f"auxiliary weights {asset['file']} not found in weights_dir")
    current = sha256_file(p)
    if current != asset["sha256"]:
        raise CheckpointMismatch(f"auxiliary weights {asset['file']} changed after plan (sha256 {current[:12]} on "
                                 f"disk, {asset['sha256'][:12]} recorded); run plan again or restore the weights")
    return p


def hf_snapshot_config(detector: dict, paths: dict) -> dict:
    _revision, snap = _hf_snapshot(Path(paths["hf_cache_dir"]), detector["checkpoint"]["hf_repo"])
    p = snap / "config.json"
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint {detector['name']}: config.json missing from the local snapshot")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"checkpoint {detector['name']}: config.json unreadable ({e})") from None


def resolve_weights_file(model_name: str, paths: dict) -> dict:
    """An auxiliary weights file in `weights_dir` (for example the BoT-SORT ReID), with its hash."""
    filename = model_name if model_name.endswith((".pt", ".pth")) else f"{model_name}.pt"
    p = Path(paths["weights_dir"]) / filename
    if not p.is_file():
        raise FileNotFoundError(f"auxiliary weights {filename} not found in weights_dir")
    return {"file": filename, "sha256": sha256_file(p), "size": p.stat().st_size}


def to_native(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x

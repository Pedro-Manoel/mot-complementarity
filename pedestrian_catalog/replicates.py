"""Seeds of the quality replicates.

A single seed is used because detections and HOTA did not vary across seeds; `quality.determinism_check` can add a
second seed on the first sequences of each dataset.
"""
from __future__ import annotations

from .schema import SpecError


def _k(l: dict) -> str:
    return f"{l['dataset']}/{l['sequence_id']}"


def _is_true(x) -> bool:
    return x is True or str(x).strip().lower() in ("true", "1")


def primary_seeds(quality: dict) -> list[int]:
    return [int(s) for s in quality["replicate_seeds"]]


def determinism_sample(lines: list[dict], per_dataset: int) -> list[str]:
    by_dataset: dict[str, list[str]] = {}
    for l in sorted((l for l in lines if _is_true(l["eligible"])), key=lambda l: (l["dataset"], l["sequence_id"])):
        bucket = by_dataset.setdefault(l["dataset"], [])
        if len(bucket) < int(per_dataset):
            bucket.append(_k(l))
    return [k for ds in sorted(by_dataset) for k in by_dataset[ds]]


def seeds_by_sequence(lines: list[dict], quality: dict) -> dict[str, list[int]]:
    """The primary seeds for every eligible sequence, plus the check seed on the determinism sample."""
    primary = primary_seeds(quality)
    out = {_k(l): list(primary) for l in lines if _is_true(l["eligible"])}
    chk = quality.get("determinism_check")
    if chk:
        seed = int(chk["seed"])
        if seed in primary:
            raise SpecError(f"quality.determinism_check.seed {seed} is already a primary seed")
        for k in determinism_sample(lines, int(chk.get("sequences_per_dataset", 1))):
            out[k] = out[k] + [seed]
    return out


def seeds_for(sequence: dict, seeds: list[int], by_seq: dict | None) -> list[int]:
    if by_seq is None:
        return [int(s) for s in seeds]
    return [int(s) for s in by_seq.get(f"{sequence['dataset']}/{sequence['sequence_id']}", seeds)]

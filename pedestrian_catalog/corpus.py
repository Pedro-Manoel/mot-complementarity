"""Corpus inventory and the `corpus.csv` manifest.

Every sequence of the declared datasets becomes one line, eligible or excluded with a reason. GT rows after the last
frame (e.g. CrowdTrack track0004) do not exclude a sequence: they are counted and left out of the evaluation.
"""
from __future__ import annotations

import configparser
import csv
import io
import math
import os
import re
from pathlib import Path

from .provenance import sha256_bytes, sha256_file, write_text_atomic

CORPUS_COLUMNS = [
    "dataset", "sequence_id", "source_path", "split", "n_frames", "fps", "width", "height",
    "physical_group_id", "physical_group_source", "annotation_protocol", "eligible",
    "exclusion_reason", "exclusion_detail", "gt_sha256", "frames_manifest_sha256",
    "gt_rows_after_last_frame", "camera_motion", "camera_motion_median_frac", "camera_motion_source",
]
CAMERA_MOTION_LABELS = ("fixed", "moving")

# PersonPath22 annotates only keyframes; the other frames are ignore regions that only TrackEval's PersonPath22 honours.
DATASET_PROTOCOLS = {
    "MOT17": {"annotation_protocol": "motchallenge_mot17", "evaluator_benchmark": "MOT17",
              "eval_dataset": "MotChallenge2DBox"},
    "MOT20": {"annotation_protocol": "motchallenge_mot20", "evaluator_benchmark": "MOT20",
              "eval_dataset": "MotChallenge2DBox"},
    "SOMPT22": {"annotation_protocol": "sompt22_mot", "evaluator_benchmark": "MOT17", "eval_dataset": "MotChallenge2DBox"},
    "CrowdTrack-person": {"annotation_protocol": "crowdtrack_person_mot", "evaluator_benchmark": "MOT17",
                          "eval_dataset": "MotChallenge2DBox"},
    "PersonPath22": {"annotation_protocol": "personpath22_keyframes", "evaluator_benchmark": "person_path_22",
                     "eval_dataset": "PersonPath22"},
}
DEFAULT_EVAL_DATASET = "MotChallenge2DBox"

# The DPM, FRCNN and SDP versions of a MOT17 sequence share frames and GT.
_MOT17_PHYSICAL = re.compile(r"^(MOT17-\d\d)-(FRCNN|SDP|DPM)$")
_MOT17_PREFERENCE = ("FRCNN", "SDP", "DPM")

DEFAULT_GROUP_SOURCE = "own_sequence_unverified"
_INT_COLUMNS = ("n_frames", "width", "height", "gt_rows_after_last_frame")


def _empty_line(dataset: str, seq_dir: Path, root: Path, split: str) -> dict:
    line = {c: "" for c in CORPUS_COLUMNS}
    line.update(dataset=dataset, sequence_id=seq_dir.name, source_path=seq_dir.relative_to(root).as_posix(),
                split=split, n_frames=0, fps=0.0, width=0, height=0, eligible=False, gt_rows_after_last_frame=0,
                physical_group_id=f"{dataset}:{seq_dir.name}", physical_group_source=DEFAULT_GROUP_SOURCE)
    return line


def _exclude(line: dict, reason: str, detail: str) -> dict:
    line["eligible"] = False
    line["exclusion_reason"] = reason
    line["exclusion_detail"] = detail
    return line


def _frames_manifest(files: list[os.DirEntry]) -> str:
    """Name and size of every frame plus the hash of the first, middle and last one; hashing all frames is too slow."""
    ordered = sorted(files, key=lambda e: e.name)
    lines = [f"{e.name},{e.stat().st_size}" for e in ordered]
    if ordered:
        sample = {e.name: e for e in (ordered[0], ordered[len(ordered) // 2], ordered[-1])}
        for e in sample.values():
            lines.append(f"amostra:{e.name}:{sha256_file(e.path)}")  # literal kept: part of the published hashes
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def _read_gt(file_path: Path) -> list[list[str]]:
    with file_path.open(encoding="utf-8", newline="") as fh:
        return [line for line in csv.reader(fh) if line and any(c.strip() for c in line)]


def _inspect(dataset: str, split: str, root: Path, seq_dir: Path) -> dict:
    line = _empty_line(dataset, seq_dir, root, split)
    protocol = DATASET_PROTOCOLS.get(dataset)
    if protocol is None:
        return _exclude(line, "unknown_dataset", f"no evaluation protocol declared for {dataset}")
    line["annotation_protocol"] = protocol["annotation_protocol"]

    ini = seq_dir / "seqinfo.ini"
    if not ini.is_file():
        return _exclude(line, "missing_seqinfo", "seqinfo.ini missing")
    cp = configparser.ConfigParser()
    cp.read(ini, encoding="utf-8")
    if "Sequence" not in cp:
        return _exclude(line, "invalid_seqinfo", "[Sequence] section missing")
    s = cp["Sequence"]
    declared_class = s.get("evalDataset", DEFAULT_EVAL_DATASET).strip()
    if declared_class != protocol["eval_dataset"]:
        return _exclude(line, "evaluation_class_mismatch",
                        f"seqinfo declares evalDataset={declared_class!r}; protocol {protocol['annotation_protocol']} "
                        f"requires {protocol['eval_dataset']}")
    try:
        n, im_width, im_height = int(s.get("seqLength", "0")), int(s.get("imWidth", "0")), int(s.get("imHeight", "0"))
    except ValueError:
        return _exclude(line, "invalid_seqinfo", "seqLength, imWidth or imHeight not numeric")
    if n <= 0 or im_width <= 0 or im_height <= 0:
        return _exclude(line, "invalid_seqinfo", f"seqLength={n}, imWidth={im_width}, imHeight={im_height}")
    line.update(n_frames=n, width=im_width, height=im_height)
    raw = s.get("frameRate", "")
    try:
        fps = float(raw)
    except ValueError:
        return _exclude(line, "invalid_fps", f"frameRate={raw!r}")
    if not math.isfinite(fps) or fps <= 0:
        return _exclude(line, "invalid_fps", f"frameRate={raw!r}")
    line["fps"] = fps

    im_dir, ext = s.get("imDir", "img1"), s.get("imExt", ".jpg").lower()
    img_dir = seq_dir / im_dir
    if not img_dir.is_dir():
        return _exclude(line, "missing_frames", f"directory {im_dir} missing")
    files = [e for e in os.scandir(img_dir) if e.is_file() and e.name.lower().endswith(ext)]
    line["frames_manifest_sha256"] = _frames_manifest(files)
    stems = [Path(e.name).stem for e in files]
    widths = {len(h) for h in stems if h.isdigit()}
    indices = sorted(int(h) for h in stems if h.isdigit())
    if len(widths) != 1 or len(indices) != len(stems) or indices != list(range(1, n + 1)):
        bounds = f", first {indices[0]}, last {indices[-1]}" if indices else ""
        return _exclude(line, "misaligned_frames", f"{len(stems)} frames, expected 1..{n}{bounds}")

    gt = seq_dir / "gt" / "gt.txt"
    if not gt.is_file():
        return _exclude(line, "missing_gt", "gt/gt.txt missing")
    line["gt_sha256"] = sha256_file(gt)
    try:
        lines = _read_gt(gt)
        if any(len(r) < 6 for r in lines):
            raise ValueError
        frames = [int(float(r[0])) for r in lines]
    except (ValueError, IndexError):
        return _exclude(line, "invalid_gt_format", "row with fewer than 6 columns or non-numeric frame")
    before = [q for q in frames if q < 1]
    if before:
        return _exclude(line, "gt_out_of_range",
                        f"{len(before)} GT rows before frame 1, frames {min(before)} to {max(before)}")
    line["gt_rows_after_last_frame"] = sum(1 for q in frames if q > n)
    inside = [r for r, q in zip(lines, frames) if q <= n]
    pedestrians = [r for r in inside if len(r) < 8 or (int(float(r[7])) == 1 and float(r[6]) != 0)]
    if not pedestrians:
        return _exclude(line, "gt_without_pedestrians", "no row with class 1 and conf other than 0")
    line["eligible"] = True
    return line


def _apply_mot17(lines: list[dict]) -> None:
    """Keeps one version per physical MOT17 sequence, in the order of _MOT17_PREFERENCE."""
    groups: dict[str, list[dict]] = {}
    for line in lines:
        m = _MOT17_PHYSICAL.match(line["sequence_id"]) if line["dataset"] == "MOT17" else None
        if m:
            line["physical_group_id"] = f"MOT17:{m.group(1)}"
            if line["eligible"]:
                groups.setdefault(line["physical_group_id"], []).append(line)
    for members in groups.values():
        members.sort(key=lambda r: _MOT17_PREFERENCE.index(_MOT17_PHYSICAL.match(r["sequence_id"]).group(2)))
        for other in members[1:]:
            _exclude(other, "copy_of_same_physical_sequence", f"same physical sequence as {members[0]['sequence_id']}")


def _apply_identical_copies(lines: list[dict]) -> None:
    seen: dict[tuple[str, str], dict] = {}
    for line in lines:
        if not line["eligible"]:
            continue
        fingerprint = (line["gt_sha256"], line["frames_manifest_sha256"])
        if fingerprint in seen:
            orig = seen[fingerprint]
            _exclude(line, "identical_copy", f"identical to {orig['dataset']}/{orig['sequence_id']}")
        else:
            seen[fingerprint] = line


def _apply_selection(lines: list[dict], selection, declared) -> None:
    """Only listed sequences stay eligible (used for PersonPath22, whose converted folder holds extra sequences)."""
    for dataset, file_path in (selection or {}).items():
        if dataset not in declared:
            raise ValueError(f"selection declared for dataset {dataset}, which is not in paths['datasets']")
        with Path(file_path).open(encoding="utf-8", newline="") as fh:
            chosen = {r["sequence_id"] for r in csv.DictReader(fh)}
        present = {l["sequence_id"] for l in lines if l["dataset"] == dataset}
        missing = sorted(chosen - present)
        if missing:
            raise ValueError(f"selection {Path(file_path).name} of {dataset} lists sequences that are not on disk: "
                             f"{missing}")
        for line in lines:
            if line["dataset"] == dataset and line["eligible"] and line["sequence_id"] not in chosen:
                _exclude(line, "outside_selection", f"not in the declared selection {Path(file_path).name}")


def _read_groups(file_path) -> dict[tuple[str, str], tuple[str, str]]:
    if not file_path:
        return {}
    with Path(file_path).open(encoding="utf-8", newline="") as fh:
        return {(r["dataset"], r["sequence_id"]): (r["physical_group_id"], r.get("source", "")) for r in csv.DictReader(fh)}


def _read_camera(file_path) -> dict[tuple[str, str], dict]:
    if not file_path:
        return {}
    with Path(file_path).open(encoding="utf-8", newline="") as fh:
        lines = list(csv.DictReader(fh))
    out = {}
    for r in lines:
        if r["camera_motion"] not in CAMERA_MOTION_LABELS:
            raise ValueError(f"camera partition: camera_motion {r['camera_motion']!r} in {r['dataset']}/"
                             f"{r['sequence_id']}; accepted {list(CAMERA_MOTION_LABELS)}")
        out[(r["dataset"], r["sequence_id"])] = r
    return out


def inspect_corpus(paths: dict) -> list[dict]:
    root = Path(paths["data_root"])
    lines: list[dict] = []
    for dataset, spec in paths["datasets"].items():
        d = root / spec["dir"]
        if not d.is_dir():
            raise FileNotFoundError(f"directory of dataset {dataset} not found: {spec['dir']} (under data_root)")
        for seq_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            lines.append(_inspect(dataset, str(spec.get("split", "")), root, seq_dir))
    lines.sort(key=lambda r: (r["dataset"], r["sequence_id"]))
    _apply_mot17(lines)
    _apply_identical_copies(lines)
    _apply_selection(lines, paths.get("selection"), paths["datasets"])
    for (dataset, seq), (group_id, source) in _read_groups(paths.get("physical_groups")).items():
        for line in lines:
            if line["dataset"] == dataset and line["sequence_id"] == seq:
                line["physical_group_id"] = group_id
                line["physical_group_source"] = f"declared: {source}" if source else "declared"
    camera = _read_camera(paths.get("camera_motion"))
    if camera:
        unclassified = [f"{l['dataset']}/{l['sequence_id']}" for l in lines
                        if l["eligible"] and (l["dataset"], l["sequence_id"]) not in camera]
        if unclassified:
            raise ValueError(f"camera partition lacks the eligible sequences {unclassified}")
        for line in lines:
            r = camera.get((line["dataset"], line["sequence_id"]))
            if r is not None:
                line.update(camera_motion=r["camera_motion"], camera_motion_median_frac=r["median_frac"],
                            camera_motion_source=r["source"])
    return lines


def _format(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_corpus_csv(lines: list[dict], dest) -> Path:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CORPUS_COLUMNS, lineterminator="\n")
    w.writeheader()
    for line in lines:
        w.writerow({c: _format(line[c]) for c in CORPUS_COLUMNS})
    return write_text_atomic(dest, buf.getvalue())


def read_corpus_csv(src) -> list[dict]:
    with Path(src).open(encoding="utf-8", newline="") as fh:
        lines = list(csv.DictReader(fh))
    for line in lines:
        if list(line) != CORPUS_COLUMNS:
            raise ValueError(f"corpus.csv with unexpected columns: {list(line)}")
        for c in _INT_COLUMNS:
            line[c] = int(line[c])
        line["fps"] = float(line["fps"])
        line["eligible"] = line["eligible"] == "true"
    return lines


def resolve_sequence(line: dict, paths: dict) -> dict:
    """The frame-index width is read from disk (6 digits in MOT, 8 in other datasets)."""
    if not line["eligible"]:
        raise ValueError(f"{line['dataset']}/{line['sequence_id']} is not eligible: {line['exclusion_reason']}")
    d = Path(paths["data_root"]) / line["source_path"]
    cp = configparser.ConfigParser()
    cp.read(d / "seqinfo.ini", encoding="utf-8")
    s = cp["Sequence"]
    im_dir, ext = s.get("imDir", "img1"), s.get("imExt", ".jpg")
    im_width = next((len(Path(e.name).stem) for e in os.scandir(d / im_dir)
                     if e.is_file() and e.name.lower().endswith(ext.lower()) and Path(e.name).stem.isdigit()), None)
    if im_width is None:
        raise FileNotFoundError(f"{line['dataset']}/{line['sequence_id']}: no frames in {im_dir}")
    return {"dataset": line["dataset"], "sequence_id": line["sequence_id"], "dir": str(d), "im_dir": im_dir,
            "im_ext": ext, "index_width": im_width, "n_frames": line["n_frames"], "width": line["width"],
            "height": line["height"], "fps": line["fps"], "frames_manifest_sha256": line["frames_manifest_sha256"],
            "gt_sha256": line["gt_sha256"], "gt_rows_after_last_frame": line["gt_rows_after_last_frame"],
            "physical_group_id": line["physical_group_id"], "annotation_protocol": line["annotation_protocol"],
            "camera_motion": line["camera_motion"]}

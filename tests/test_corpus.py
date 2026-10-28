"""Corpus inventory: manifest metadata, exclusion reasons and the camera partition."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from pedestrian_catalog import corpus as K


def _seq(base: Path, rel: str, n: int = 5, fps: str = "30", ext: str = ".jpg", gt: bool = True,
         frames=None, gt_lines=None, width: int = 6, content: bytes = b"x") -> Path:
    d = base / rel
    (d / "img1").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname = {d.name}\nimDir = img1\nframeRate = {fps}\nseqLength = {n}\n"
        f"imWidth = 1920\nimHeight = 1080\nimExt = {ext}\n", encoding="utf-8")
    for i in (frames if frames is not None else range(1, n + 1)):
        (d / "img1" / f"{i:0{width}d}{ext}").write_bytes(content)
    if gt:
        (d / "gt").mkdir()
        lines = gt_lines if gt_lines is not None else [f"{i},1,10,10,5,10,1,1,1.0" for i in range(1, n + 1)]
        (d / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def _paths(base: Path, create: bool = True, **extra) -> dict:
    p = {"data_root": str(base), "datasets": {
        "MOT17": {"dir": "MOT17/train", "split": "train"},
        "MOT20": {"dir": "MOT20/train", "split": "train"},
        "SOMPT22": {"dir": "SOMPT22/train", "split": "train"},
        "CrowdTrack-person": {"dir": "CrowdTrack-person", "split": "all"},
    }}
    if create:
        for spec in p["datasets"].values():
            (base / spec["dir"]).mkdir(parents=True, exist_ok=True)
    p.update(extra)
    return p


def _by_id(rows):
    return {r["sequence_id"]: r for r in rows}


def test_manifest_fields_hashes_and_relative_path(tmp_path):
    d = _seq(tmp_path, "MOT20/train/MOT20-01")
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-01"]
    assert list(r) == K.CORPUS_COLUMNS
    assert r["eligible"] is True and r["exclusion_reason"] == ""
    assert (r["dataset"], r["split"], r["n_frames"], r["fps"], r["width"], r["height"]) == ("MOT20", "train", 5, 30.0, 1920, 1080)
    assert r["source_path"] == "MOT20/train/MOT20-01"
    assert r["gt_sha256"] == hashlib.sha256((d / "gt" / "gt.txt").read_bytes()).hexdigest()
    assert len(r["frames_manifest_sha256"]) == 64
    assert r["annotation_protocol"] == "motchallenge_mot20"
    assert r["physical_group_id"] == "MOT20:MOT20-01" and r["physical_group_source"] == "own_sequence_unverified"


def test_mot17_copies_of_the_same_physical_sequence_keep_one(tmp_path):
    for v in ("FRCNN", "DPM", "SDP"):
        _seq(tmp_path, f"MOT17/train/MOT17-02-{v}")
    rows = _by_id(K.inspect_corpus(_paths(tmp_path)))
    assert rows["MOT17-02-FRCNN"]["eligible"] is True and rows["MOT17-02-FRCNN"]["exclusion_detail"] == ""
    for v in ("DPM", "SDP"):
        assert rows[f"MOT17-02-{v}"]["eligible"] is False
        assert rows[f"MOT17-02-{v}"]["exclusion_reason"] == "copy_of_same_physical_sequence"
        assert rows[f"MOT17-02-{v}"]["exclusion_detail"] == "same physical sequence as MOT17-02-FRCNN"
    assert {r["physical_group_id"] for r in rows.values()} == {"MOT17:MOT17-02"}


def test_identical_copy_in_different_places_is_excluded(tmp_path):
    _seq(tmp_path, "SOMPT22/train/SOMPT22-02", content=b"same")
    _seq(tmp_path, "SOMPT22/train/SOMPT22-99", content=b"same")
    rows = _by_id(K.inspect_corpus(_paths(tmp_path)))
    assert rows["SOMPT22-02"]["eligible"] is True
    assert rows["SOMPT22-99"]["eligible"] is False and rows["SOMPT22-99"]["exclusion_reason"] == "identical_copy"
    assert rows["SOMPT22-99"]["exclusion_detail"] == "identical to SOMPT22/SOMPT22-02"


@pytest.mark.parametrize("fps", ["0", "-25", "abc"])
def test_invalid_fps_excludes(tmp_path, fps):
    _seq(tmp_path, "MOT20/train/MOT20-02", fps=fps)
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-02"]
    assert r["eligible"] is False and r["exclusion_reason"] == "invalid_fps"


def test_missing_gt_excludes(tmp_path):
    _seq(tmp_path, "CrowdTrack-person/track0001", gt=False)
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["track0001"]
    assert r["eligible"] is False and r["exclusion_reason"] == "missing_gt" and r["gt_sha256"] == ""


@pytest.mark.parametrize("frames", [range(1, 5), [1, 2, 4, 5, 6], range(0, 5)])
def test_misaligned_frames_exclude(tmp_path, frames):
    _seq(tmp_path, "CrowdTrack-person/track0002", n=5, frames=frames)
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["track0002"]
    assert r["eligible"] is False and r["exclusion_reason"] == "misaligned_frames"
    assert r["exclusion_detail"].startswith(f"{len(list(frames))} frames, expected 1..5")


def test_index_width_derived_from_disk(tmp_path):
    _seq(tmp_path, "CrowdTrack-person/track0003", width=8, ext=".png")
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["track0003"]
    assert r["eligible"] is True


def test_gt_before_the_first_frame_excludes(tmp_path):
    _seq(tmp_path, "MOT20/train/MOT20-03", n=5, gt_lines=["1,1,10,10,5,10,1,1,1", "0,1,10,10,5,10,1,1,1",
                                                          "-1,1,10,10,5,10,1,1,1", "7,1,10,10,5,10,1,1,1"])
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-03"]
    assert r["eligible"] is False and r["exclusion_reason"] == "gt_out_of_range"
    assert r["exclusion_detail"] == "2 GT rows before frame 1, frames -1 to 0"


def test_gt_after_the_last_frame_stays_eligible_and_counts_the_rows(tmp_path):
    _seq(tmp_path, "CrowdTrack-person/track0004", n=5, gt_lines=["1,1,10,10,5,10,1,1,1", "7,1,10,10,5,10,1,1,1",
                                                                 "8,1,10,10,5,10,1,1,1", "9,2,10,10,5,10,1,1,1"])
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["track0004"]
    assert r["eligible"] is True and r["exclusion_reason"] == "" and r["gt_rows_after_last_frame"] == 3


def test_gt_only_after_the_last_frame_has_no_pedestrians(tmp_path):
    _seq(tmp_path, "MOT20/train/MOT20-05", n=5, gt_lines=["6,1,10,10,5,10,1,1,1", "7,1,10,10,5,10,1,1,1"])
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-05"]
    assert r["eligible"] is False and r["exclusion_reason"] == "gt_without_pedestrians"


@pytest.mark.parametrize("frame,eligible,after", [(0, False, 0), (6, True, 1), (5, True, 0), (1, True, 0)])
def test_gt_at_the_edge_of_the_range(tmp_path, frame, eligible, after):
    _seq(tmp_path, "MOT20/train/MOT20-02", n=5, gt_lines=["1,1,10,10,5,10,1,1,1", f"{frame},2,10,10,5,10,1,1,1"])
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-02"]
    assert r["eligible"] is eligible
    if eligible:
        assert r["gt_rows_after_last_frame"] == after


@pytest.mark.parametrize("line", ["{i},1,10,10,5,10,0,7,1", "{i},1,10,10,5,10,0,1,1"])  # class 7; class 1 with conf 0
def test_gt_without_pedestrians_excludes(tmp_path, line):
    _seq(tmp_path, "MOT20/train/MOT20-05", n=3, gt_lines=[line.format(i=i) for i in (1, 2, 3)])
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-05"]
    assert r["eligible"] is False and r["exclusion_reason"] == "gt_without_pedestrians"


def test_same_size_different_content_is_not_a_copy(tmp_path):
    _seq(tmp_path, "SOMPT22/train/SOMPT22-02", content=b"a")
    _seq(tmp_path, "SOMPT22/train/SOMPT22-04", content=b"b")
    rows = _by_id(K.inspect_corpus(_paths(tmp_path)))
    assert rows["SOMPT22-02"]["eligible"] is True and rows["SOMPT22-04"]["eligible"] is True


def test_unknown_dataset_does_not_become_crowdtrack(tmp_path):
    _seq(tmp_path, "VisDrone-person/uav0000009_03358_v")
    _seq(tmp_path, "CrowdTrack-person/track0004")
    p = _paths(tmp_path)
    p["datasets"]["VisDrone-person"] = {"dir": "VisDrone-person", "split": "all"}
    rows = _by_id(K.inspect_corpus(p))
    r = rows["uav0000009_03358_v"]
    assert r["eligible"] is False and r["exclusion_reason"] == "unknown_dataset" and r["annotation_protocol"] == ""
    assert rows["track0004"]["eligible"] is True


def test_declared_physical_groups_override(tmp_path):
    _seq(tmp_path, "MOT20/train/MOT20-01", content=b"a")
    _seq(tmp_path, "MOT20/train/MOT20-02", content=b"b")
    groups = tmp_path / "groups.csv"
    groups.write_text("dataset,sequence_id,physical_group_id,source\nMOT20,MOT20-01,MOT20:station,MOT20 paper\n"
                      "MOT20,MOT20-02,MOT20:station,MOT20 paper\n", encoding="utf-8")
    rows = _by_id(K.inspect_corpus(_paths(tmp_path, physical_groups=str(groups))))
    assert rows["MOT20-01"]["physical_group_id"] == rows["MOT20-02"]["physical_group_id"] == "MOT20:station"
    assert rows["MOT20-01"]["physical_group_source"] == "declared: MOT20 paper"


def _partition(tmp_path, lines) -> Path:
    p = tmp_path / "camera.csv"
    p.write_text("dataset,sequence_id,median_frac,camera_motion,source\n" + "".join(l + "\n" for l in lines), encoding="utf-8")
    return p


def test_fixed_or_moving_camera_comes_from_the_declared_partition(tmp_path):
    _seq(tmp_path, "MOT20/train/MOT20-01", content=b"a")
    _seq(tmp_path, "MOT17/train/MOT17-05-FRCNN", content=b"b")
    p = _partition(tmp_path, ["MOT20,MOT20-01,0.000377,fixed,measured", "MOT17,MOT17-05-FRCNN,0.007397,moving,measured"])
    rows = _by_id(K.inspect_corpus(_paths(tmp_path, camera_motion=str(p))))
    assert (rows["MOT20-01"]["camera_motion"], rows["MOT20-01"]["camera_motion_median_frac"]) == ("fixed", "0.000377")
    assert rows["MOT17-05-FRCNN"]["camera_motion"] == "moving" and rows["MOT17-05-FRCNN"]["camera_motion_source"] == "measured"


def test_without_a_declared_partition_the_camera_stays_empty(tmp_path):
    _seq(tmp_path, "MOT20/train/MOT20-01")
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-01"]
    assert (r["camera_motion"], r["camera_motion_median_frac"], r["camera_motion_source"]) == ("", "", "")


def test_eligible_sequence_missing_from_the_partition_is_rejected(tmp_path):
    _seq(tmp_path, "MOT20/train/MOT20-01", content=b"a")
    _seq(tmp_path, "MOT20/train/MOT20-02", content=b"b")
    p = _partition(tmp_path, ["MOT20,MOT20-01,0.000377,fixed,measured"])
    with pytest.raises(ValueError, match="MOT20/MOT20-02"):
        K.inspect_corpus(_paths(tmp_path, camera_motion=str(p)))


@pytest.mark.parametrize("label", ["Fixed", "", "static"])
def test_partition_with_an_invalid_label_is_rejected(tmp_path, label):
    _seq(tmp_path, "MOT20/train/MOT20-01")
    p = _partition(tmp_path, [f"MOT20,MOT20-01,0.000377,{label},measured"])
    with pytest.raises(ValueError, match="camera_motion"):
        K.inspect_corpus(_paths(tmp_path, camera_motion=str(p)))


def test_missing_dataset_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="MOT17"):
        K.inspect_corpus(_paths(tmp_path, create=False))


def test_writing_and_reading_the_csv_keep_columns_and_order(tmp_path):
    for rel in ("MOT20/train/MOT20-02", "MOT20/train/MOT20-01", "SOMPT22/train/SOMPT22-04"):
        _seq(tmp_path, rel, content=rel.encode())
    (tmp_path / "MOT17" / "train").mkdir(parents=True)
    (tmp_path / "CrowdTrack-person").mkdir()
    p = _partition(tmp_path, ["MOT20,MOT20-01,0.0003,fixed,m", "MOT20,MOT20-02,0.0002,fixed,m", "SOMPT22,SOMPT22-04,0.00001,fixed,m"])
    rows = K.inspect_corpus(_paths(tmp_path, camera_motion=str(p)))
    assert [(r["dataset"], r["sequence_id"]) for r in rows] == sorted((r["dataset"], r["sequence_id"]) for r in rows)
    out = tmp_path / "corpus.csv"
    K.write_corpus_csv(rows, out)
    with out.open(encoding="utf-8", newline="") as fh:
        read = list(csv.DictReader(fh))
    assert list(read[0]) == K.CORPUS_COLUMNS and len(read) == 3
    assert K.read_corpus_csv(out) == rows


def test_camera_partition_of_the_repository_follows_the_anchored_rule():
    import math
    root = Path(__file__).resolve().parents[1]
    with (root / "manifests" / "camera_motion.csv").open(encoding="utf-8", newline="") as fh:
        lines = list(csv.DictReader(fh))
    v = {r["sequence_id"]: float(r["median_frac"]) for r in lines}
    static = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
    moving = ("MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN")
    threshold = math.sqrt(max(v[s] for s in static) * min(v[s] for s in moving))
    assert list(lines[0]) == ["dataset", "sequence_id", "median_frac", "camera_motion", "source"]
    assert all(r["median_frac"] == f"{float(r['median_frac']):.6f}" for r in lines)
    assert {r["camera_motion"] for r in lines} == set(K.CAMERA_MOTION_LABELS)
    assert all(r["camera_motion"] == ("fixed" if v[r["sequence_id"]] < threshold else "moving") for r in lines)
    assert all(f"moving if >= {threshold:.6f}" in r["source"] for r in lines)
    assert all(r["camera_motion"] == "fixed" for r in lines if r["sequence_id"] in static)
    assert all(r["camera_motion"] == "moving" for r in lines if r["sequence_id"] in moving)
    assert len(lines) == 100 and sum(r["camera_motion"] == "fixed" for r in lines) == 75


IGNORE = "1000000000"


def _seq_pp22(base: Path, name: str, eval_dataset="PersonPath22", n: int = 5, content: bytes = b"p") -> Path:
    """Mimics prepare/personpath22_to_mot.py: people on the keyframes, an ignore region (class 13) on the others."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"{i},1,10,10,5,10,1,1,-1,-1" if i % 2 else f"{i},{IGNORE},-1920,-1080,5760,3240,1,13,-1,-1")
    d = _seq(base, f"PersonPath22/seqs/{name}", n=n, gt_lines=lines, content=content)
    if eval_dataset is not None:
        ini = d / "seqinfo.ini"
        ini.write_text(ini.read_text(encoding="utf-8") + f"evalDataset = {eval_dataset}\n", encoding="utf-8")
    return d


def _paths_pp22(base: Path, **extra) -> dict:
    p = _paths(base, **extra)
    p["datasets"]["PersonPath22"] = {"dir": "PersonPath22/seqs", "split": "all"}
    (base / "PersonPath22/seqs").mkdir(parents=True, exist_ok=True)
    return p


def _selection(base: Path, names) -> Path:
    s = base / "selection.csv"
    s.write_text("sequence_id,uid\n" + "".join(f"{n},uid_{n}\n" for n in names), encoding="utf-8")
    return s


def test_personpath22_has_its_own_protocol(tmp_path):
    _seq_pp22(tmp_path, "PersonPath22-00018")
    r = _by_id(K.inspect_corpus(_paths_pp22(tmp_path)))["PersonPath22-00018"]
    assert r["eligible"] is True and r["annotation_protocol"] == "personpath22_keyframes"
    assert r["source_path"] == "PersonPath22/seqs/PersonPath22-00018"


@pytest.mark.parametrize("declared", [None, "MotChallenge2DBox"])
def test_personpath22_without_the_declared_evaluation_class_is_excluded(tmp_path, declared):
    _seq_pp22(tmp_path, "PersonPath22-00019", eval_dataset=declared)
    r = _by_id(K.inspect_corpus(_paths_pp22(tmp_path)))["PersonPath22-00019"]
    assert r["eligible"] is False and r["exclusion_reason"] == "evaluation_class_mismatch"
    assert "PersonPath22" in r["exclusion_detail"]


def test_mot_dataset_declaring_another_evaluation_class_is_excluded(tmp_path):
    d = _seq(tmp_path, "MOT20/train/MOT20-01")
    (d / "seqinfo.ini").write_text((d / "seqinfo.ini").read_text(encoding="utf-8") + "evalDataset = PersonPath22\n", encoding="utf-8")
    r = _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-01"]
    assert r["eligible"] is False and r["exclusion_reason"] == "evaluation_class_mismatch"


def test_mot_declaring_the_usual_class_stays_eligible(tmp_path):
    d = _seq(tmp_path, "MOT20/train/MOT20-01")
    (d / "seqinfo.ini").write_text((d / "seqinfo.ini").read_text(encoding="utf-8") + "evalDataset = MotChallenge2DBox\n",
                                   encoding="utf-8")
    assert _by_id(K.inspect_corpus(_paths(tmp_path)))["MOT20-01"]["eligible"] is True


def test_declared_selection_keeps_only_the_drawn_sequences_of_its_dataset_eligible(tmp_path):
    for i, n in enumerate(("PersonPath22-00010", "PersonPath22-00011", "PersonPath22-00014")):
        _seq_pp22(tmp_path, n, content=bytes([65 + i]))
    _seq(tmp_path, "MOT20/train/MOT20-01", content=b"m")
    s = _selection(tmp_path, ["PersonPath22-00010", "PersonPath22-00014"])
    rows = _by_id(K.inspect_corpus(_paths_pp22(tmp_path, selection={"PersonPath22": str(s)})))
    assert rows["PersonPath22-00010"]["eligible"] is True and rows["PersonPath22-00014"]["eligible"] is True
    r = rows["PersonPath22-00011"]
    assert r["eligible"] is False and r["exclusion_reason"] == "outside_selection" and "selection.csv" in r["exclusion_detail"]
    assert rows["MOT20-01"]["eligible"] is True


def test_selection_does_not_revive_an_ineligible_sequence(tmp_path):
    _seq_pp22(tmp_path, "PersonPath22-00010", eval_dataset=None)
    s = _selection(tmp_path, ["PersonPath22-00010"])
    r = _by_id(K.inspect_corpus(_paths_pp22(tmp_path, selection={"PersonPath22": str(s)})))["PersonPath22-00010"]
    assert r["eligible"] is False and r["exclusion_reason"] == "evaluation_class_mismatch"


def test_selection_with_a_sequence_that_does_not_exist_raises(tmp_path):
    _seq_pp22(tmp_path, "PersonPath22-00010")
    s = _selection(tmp_path, ["PersonPath22-00010", "PersonPath22-99999"])
    with pytest.raises(ValueError, match="PersonPath22-99999"):
        K.inspect_corpus(_paths_pp22(tmp_path, selection={"PersonPath22": str(s)}))


def test_selection_of_an_undeclared_dataset_raises(tmp_path):
    s = _selection(tmp_path, ["X"])
    with pytest.raises(ValueError, match="VisDrone"):
        K.inspect_corpus(_paths(tmp_path, selection={"VisDrone-person": str(s)}))

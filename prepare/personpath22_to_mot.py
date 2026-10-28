"""Converts the selected PersonPath22 videos to the MOTChallenge format: ground truth, frames and seqinfo.ini, with the
evaluation restricted to the annotated keyframes.

The ground truth follows the official TrackEval one: crowds and frames without annotation become ignore regions (class
13), which only TrackEval's PersonPath22 class honours, hence evalDataset=PersonPath22 in seqinfo.ini.

    python prepare/personpath22_to_mot.py --dataset-root <personpath22> \
        --selection manifests/personpath22_selection.csv --output-root <data_root>/PersonPath22/seqs
"""
from __future__ import annotations

import argparse
import collections
import configparser
import csv
import json
from io import StringIO
from pathlib import Path

PREFIX = "PersonPath22-"
PERSON = frozenset({"person", "sitting_person", "standing_person"})
MIN_HEIGHT = 20
IOA_CROWD = 0.95
IGNORE_ID = 1_000_000_000
JPEG_QUALITY = 95


def ioa(b, c) -> float:
    """Fraction of the area of `b` (x, y, w, h) inside `c`."""
    x1, y1 = max(b[0], c[0]), max(b[1], c[1])
    x2, y2 = min(b[0] + b[2], c[0] + c[2]), min(b[1] + b[3], c[1] + c[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (b[2] * b[3]) if b[2] * b[3] > 0 else 0.0


def gt_class(labels, bb, crowds) -> int:
    lab = set(labels)
    if "crowd" in lab:
        return 13
    if "person" in lab and lab <= PERSON and bb[3] >= MIN_HEIGHT and not any(ioa(bb, c) >= IOA_CROWD for c in crowds):
        return 1
    return 8


def gt_rows(entities: list, n_frames: int, width: float, height: float) -> list[tuple]:
    """The rows `frame, id, x, y, w, h, conf, class, visibility, -1` of gt.txt."""
    by_frame = collections.defaultdict(list)
    for e in entities:
        by_frame[int(e["blob"]["frame_idx"])].append(e)
    out = []
    for f, es in by_frame.items():
        crowds = [e["bb"] for e in es if "crowd" in e["labels"]]
        has_person = any("crowd" not in e["labels"] for e in es)
        for e in es:
            c = gt_class(e["labels"], e["bb"], crowds)
            if c == 13 and not has_person:
                continue
            out.append((f + 1, int(e["id"]), *[float(v) for v in e["bb"]], 1, c, -1, -1))
    for q in range(1, int(n_frames) + 1):
        if q - 1 not in by_frame:
            out.append((q, IGNORE_ID, -float(width), -float(height), 3.0 * width, 3.0 * height, 1, 13, -1, -1))
    return sorted(out, key=lambda r: (r[0], r[1], r[7]))


def sequence_name(uid: str) -> str:
    return PREFIX + uid.replace("uid_vid_", "").replace(".mp4", "")


def seqinfo_ini(name: str, fps: float, n: int, width: int, height: int) -> str:
    """frameRate is rounded to an integer (23.976 becomes 24); the exact value goes to frameRateExact."""
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp["Sequence"] = {"name": name, "imDir": "img1", "frameRate": str(int(round(fps))), "frameRateExact": repr(float(fps)),
                      "seqLength": str(int(n)), "imWidth": str(int(width)), "imHeight": str(int(height)), "imExt": ".jpg",
                      "evalDataset": "PersonPath22"}
    s = StringIO()
    cp.write(s)
    return s.getvalue()


def write_sequence(dataset_root: Path, uid: str, meta: dict, out_root: Path) -> dict:
    md = meta[uid]["metadata"]
    n, W, H, fps = int(md["number_of_frames"]), int(md["resolution"]["width"]), int(md["resolution"]["height"]), float(md["fps"])
    d = out_root / sequence_name(uid)
    (d / "gt").mkdir(parents=True, exist_ok=True)
    entities = json.loads((dataset_root / "annotation" / "anno_amodal_2022" / f"{uid}.json")
                          .read_text(encoding="utf-8"))["entities"]
    rows = gt_rows(entities, n, W, H)
    with (d / "gt" / "gt.txt").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([int(r[0]), int(r[1]), *[f"{x:g}" for x in r[2:6]], int(r[6]), int(r[7]), int(r[8]), int(r[9])])
    (d / "seqinfo.ini").write_text(seqinfo_ini(sequence_name(uid), fps, n, W, H), encoding="utf-8")
    import cv2
    (d / "img1").mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(dataset_root / "raw_data" / uid))
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        k += 1
        if k <= n:
            cv2.imwrite(str(d / "img1" / f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cap.release()
    return {"uid": uid, "name": sequence_name(uid), "n_frames": n, "fps": fps, "gt_rows": len(rows), "decoded": k}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root", type=Path, required=True, help="folder of the PersonPath22 release")
    ap.add_argument("--selection", type=Path, default=Path("manifests/personpath22_selection.csv"),
                    help="CSV with the uid of each selected video")
    ap.add_argument("--output-root", type=Path, required=True, help="output folder, e.g. <data_root>/PersonPath22/seqs")
    a = ap.parse_args(argv)
    meta = json.loads((a.dataset_root / "annotation" / "anno_amodal_2022.json").read_text(encoding="utf-8"))["samples"]
    uids = [r["uid"] for r in csv.DictReader(a.selection.open(encoding="utf-8"))]
    a.output_root.mkdir(parents=True, exist_ok=True)
    for uid in uids:
        r = write_sequence(a.dataset_root, uid, meta, a.output_root)
        status = "ok" if r["decoded"] >= r["n_frames"] else "FEWER FRAMES THAN seqLength"
        print(f"{uid} -> {r['name']}: {r['decoded']} frames decoded, seqLength {r['n_frames']}, "
              f"{r['gt_rows']} ground-truth rows ({status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Converts CrowdTrack to the MOTChallenge format. Handles the two layouts of CrowdTrack.

Layout A (track0001 to track0004) has gt/<seq>.txt with 0-based frames; layout B has one LabelMe json per frame. Every
annotation is a person. seqinfo.ini gets 20 fps, the frame rate reported by the CrowdTrack authors.

    python prepare/crowdtrack_to_mot.py --input-root <CrowdTrack> --output-root <data_root>/CrowdTrack-person
"""
from __future__ import annotations

import argparse
import configparser
import glob
import json
import logging
import os
import shutil

import cv2

log = logging.getLogger("crowdtrack_to_mot")
FRAME_RATE = 20


def _link(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.unlink(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def _write_seqinfo(path: str, name: str, length: int, W: int, H: int, ext: str) -> None:
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp["Sequence"] = {"name": name, "imDir": "img1", "frameRate": str(FRAME_RATE),
                      "seqLength": str(length), "imWidth": str(W),
                      "imHeight": str(H), "imExt": ext}
    with open(path, "w") as f:
        cp.write(f)


def _frame_of(path: str) -> int:
    """1-based MOT frame from a trackNNNN_000000.<ext> file name, which is 0-based."""
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1]) + 1


def _gt_format_a(gt_txt: str) -> list[str]:
    out = []
    for ln in open(gt_txt):
        p = ln.split()
        if len(p) < 6:
            continue
        out.append(f"{int(p[0]) + 1},{int(p[1])},{p[2]},{p[3]},{p[4]},{p[5]},1,1,1")
    return out


def _gt_format_b(seqdir: str) -> list[str]:
    out = []
    for jf in sorted(glob.glob(os.path.join(seqdir, "*.json"))):
        fr = _frame_of(jf)
        try:
            shapes = json.load(open(jf)).get("shapes", [])
        except Exception:
            continue
        for s in shapes:
            try:
                tid = int(s["label"])
                pts = s["points"]
                if len(pts) < 2:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
            except (ValueError, KeyError, TypeError, IndexError):
                continue
            # shapes are rectangles (2 corners) or polygons; take the bounding box of all points
            x, y = max(0.0, min(xs)), max(0.0, min(ys))
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if w <= 0 or h <= 0:
                continue
            out.append(f"{fr},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,1,1")
    return out


def _clean_gt(lines: list[str]) -> tuple[list[str], int, int]:
    """Drops boxes with area <= 0 and duplicated (frame, id) pairs, which TrackEval rejects, keeping the largest box.
    Returns (lines, duplicates removed, zero-area boxes removed)."""
    best: dict[tuple[int, int], tuple[float, str]] = {}
    order: list[tuple[int, int]] = []
    n_area = 0
    for ln in lines:
        p = ln.split(",")
        w, h = float(p[4]), float(p[5])
        if w <= 0 or h <= 0:
            n_area += 1
            continue
        key = (int(p[0]), int(p[1]))
        area = w * h
        if key not in best:
            best[key] = (area, ln)
            order.append(key)
        elif area > best[key][0]:
            best[key] = (area, ln)
    kept = [best[k][1] for k in order]
    return kept, (len(lines) - n_area) - len(kept), n_area


def convert(seqdir: str, out_root: str) -> None:
    name = os.path.basename(seqdir).split(" ")[0]   # drops the " (1)" suffix of duplicated folders
    gtfiles = glob.glob(os.path.join(seqdir, "gt", "*.txt"))
    if gtfiles:
        fmt, ext = "A", ".png"
        gt = _gt_format_a(gtfiles[0])
        imgs = sorted(glob.glob(os.path.join(seqdir, "img", "*.png")))
    elif glob.glob(os.path.join(seqdir, "*.json")):
        fmt, ext = "B", ".jpg"
        gt = _gt_format_b(seqdir)
        imgs = sorted(glob.glob(os.path.join(seqdir, "*.jpg")))
    else:
        log.warning("%s: no gt/ and no *.json, skipped", name)
        return
    if not imgs:
        log.warning("%s: no images, skipped", name)
        return

    gt, ndup, narea = _clean_gt(gt)
    if ndup or narea:
        log.warning("%s: ground truth cleaned, %d duplicated (frame, id) and %d boxes with area <= 0 removed",
                    name, ndup, narea)

    out = os.path.join(out_root, name)
    os.makedirs(os.path.join(out, "gt"), exist_ok=True)
    os.makedirs(os.path.join(out, "img1"), exist_ok=True)
    # CRLF on every platform, so that the hashes match manifests/corpus.csv
    with open(os.path.join(out, "gt", "gt.txt"), "w", newline="\r\n") as f:
        f.write("\n".join(gt) + "\n")

    maxf = 0
    for src in imgs:
        fr = _frame_of(src)
        maxf = max(maxf, fr)
        _link(src, os.path.join(out, "img1", f"{fr:06d}{ext}"))
    H, W = cv2.imread(imgs[0]).shape[:2]
    _write_seqinfo(os.path.join(out, "seqinfo.ini"), name, maxf, W, H, ext)
    log.info("%s: format %s, %d images (last frame %d), %d ground-truth boxes, %dx%d, ext %s",
             name, fmt, len(imgs), maxf, len(gt), W, H, ext)


def main() -> None:
    ap = argparse.ArgumentParser(description="CrowdTrack -> MOTChallenge")
    ap.add_argument("--input-root", required=True, help="folder with the track* sequences of CrowdTrack")
    ap.add_argument("--output-root", required=True, help="output folder, e.g. <data_root>/CrowdTrack-person")
    ap.add_argument("--sequences", nargs="*", help="sequence folder names; default: every track* folder")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.sequences:
        seqdirs = [os.path.join(a.input_root, s) for s in a.sequences]
    else:
        seqdirs = [d for d in sorted(glob.glob(os.path.join(a.input_root, "track*")))
                   if os.path.isdir(d)]
    for d in seqdirs:
        convert(d, a.output_root)


if __name__ == "__main__":
    main()

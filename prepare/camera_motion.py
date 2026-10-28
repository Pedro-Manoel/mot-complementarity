"""Camera motion of each video: how much the image as a whole shifts between consecutive frames.

THRESHOLD is the geometric mean of the highest value among videos published as static and the lowest among videos
published as moving (MOT17, MOT20).

    python prepare/camera_motion.py --paths configs/paths.local.yaml      (writes manifests/camera_motion.csv)
"""
from __future__ import annotations

import argparse
import configparser
import csv
from pathlib import Path

import numpy as np
import yaml

SCALE = 0.5
CORNERS = {"maxCorners": 200, "qualityLevel": 0.01, "minDistance": 10}
LK = {"winSize": (15, 15), "maxLevel": 3}
POINTS = 12
PAIRS_PER_POINT = 4
THRESHOLD = 0.001048
SOURCE = ("prepare/camera_motion.py: median displacement of Shi-Tomasi corners tracked by Lucas-Kanade between "
          "consecutive frames, as a fraction of the image width; moving if >= 0.001048")


def sampled_pairs(n_frames: int, points: int = POINTS, pairs: int = PAIRS_PER_POINT) -> list[tuple[int, int]]:
    step = max(1, (n_frames - pairs - 1) // max(1, points))
    starts = list(range(1, max(2, n_frames - pairs), step))[:points]
    return [(i0 + k - 1, i0 + k) for i0 in starts for k in range(1, pairs + 1)]


def needed_frames(n_frames: int) -> list[int]:
    return sorted({q for pair in sampled_pairs(n_frames) for q in pair})


def correspondences(gray0, gray1):
    """Corners in the first frame and their positions in the second, in pixels of the original image, or None."""
    import cv2
    h, w = gray0.shape
    size = (int(w * SCALE), int(h * SCALE))
    a, b = cv2.resize(gray0, size), cv2.resize(gray1, size)
    corners = cv2.goodFeaturesToTrack(a, **CORNERS)
    if corners is None or len(corners) < 4:
        return None
    try:
        flow, status, _ = cv2.calcOpticalFlowPyrLK(a, b, corners, None, **LK)
    except cv2.error:
        return None
    ok = status.flatten() == 1
    if ok.sum() < 3:
        return None
    return corners[ok].reshape(-1, 2).astype(float) / SCALE, flow[ok].reshape(-1, 2).astype(float) / SCALE


def measure(frames: dict, n_frames: int, width: int) -> float:
    values = []
    for f0, f1 in sampled_pairs(n_frames):
        g0, g1 = frames.get(f0), frames.get(f1)
        if g0 is None or g1 is None:
            continue
        c = correspondences(g0, g1)
        if c is None:
            values.append(0.0)
            continue
        p0, p1 = c
        values.append(float(np.median(np.linalg.norm(p1 - p0, axis=1))))
    return float(np.median(values)) / width if values else float("nan")


def measure_folder(folder: Path) -> float:
    import cv2
    ini = configparser.ConfigParser()
    ini.read(folder / "seqinfo.ini", encoding="utf-8")
    s = ini["Sequence"]
    n, w = int(s["seqLength"]), int(s["imWidth"])
    images, ext = folder / s.get("imDir", "img1"), s.get("imExt", ".jpg")
    digits = len(next(p.stem for p in images.iterdir() if p.stem.isdigit()))
    frames = {i: cv2.imread(str(images / f"{i:0{digits}d}{ext}"), cv2.IMREAD_GRAYSCALE) for i in needed_frames(n)}
    return measure(frames, n, w)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=Path, default=Path("configs/paths.local.yaml"))
    ap.add_argument("--videos", type=Path, default=Path("paper_results/videos.csv"),
                    help="CSV with the dataset and sequence_id of each video")
    ap.add_argument("--out", type=Path, default=Path("manifests/camera_motion.csv"))
    a = ap.parse_args(argv)
    paths = yaml.safe_load(a.paths.read_text(encoding="utf-8"))
    rows = []
    for r in csv.DictReader(a.videos.open(encoding="utf-8")):
        folder = Path(paths["data_root"]) / paths["datasets"][r["dataset"]]["dir"] / r["sequence_id"]
        v = measure_folder(folder)
        rows.append({"dataset": r["dataset"], "sequence_id": r["sequence_id"], "median_frac": f"{v:.6f}",
                     "camera_motion": "fixed" if v < THRESHOLD else "moving", "source": SOURCE})
        print(f"{r['dataset']}/{r['sequence_id']}: {v:.6f} ({rows[-1]['camera_motion']})")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "sequence_id", "median_frac", "camera_motion", "source"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} videos -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

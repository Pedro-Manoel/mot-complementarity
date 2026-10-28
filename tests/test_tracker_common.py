# -*- coding: utf-8 -*-
"""Common tracker parameters act the same in the three families (real BoxMOT on CPU, synthetic scenes)."""
from __future__ import annotations

import numpy as np
import pytest

from pedestrian_catalog.backends import trackers as BT
from test_catalog import _common, _profiles

FAMILIES = ("bytetrack", "ocsort", "botsort")
EMPTY = np.zeros((480, 640, 3), np.uint8)
A = [100, 100, 150, 250, 0.9, 0]
B = [300, 100, 350, 250, 0.9, 0]


def _tracker(family: str, fps: float = 25.0, **common):
    p = {**_profiles()[family], **_common(), **common}
    if family == "botsort":
        p["with_reid"] = False  # no ReID weights in the default suite
    kw = BT.profile_kwargs(family, p, fps)
    if family == "botsort":
        kw["cmc_method"] = None  # empty frames: no camera motion compensation
    return BT.construct(family, kw, "cpu", "no_reid.pt")


def _identities(family: str, frames: list, **kw) -> list[dict]:
    """Per frame, {detection index: identity} of what the tracker returns."""
    t = _tracker(family, **kw)
    return [{int(l[7]): int(l[4]) for l in BT.update(t, np.asarray(d, np.float32).reshape(-1, 6), EMPTY)}
            for d in frames]


def test_new_object_appears_in_the_same_frame_in_the_three_families():
    frames = [[A] if i < 5 else [A, B] for i in range(1, 12)]
    for family in FAMILIES:
        s = _identities(family, frames)
        first_a = next(i for i, x in enumerate(s, start=1) if 0 in x)
        first_b = next(i for i, x in enumerate(s, start=1) if 1 in x)
        assert (first_a, first_b) == (1, 6), family  # tracks of the first frame are confirmed at once


@pytest.mark.parametrize("fps", [14.0, 15.0, 24.0, 25.0, 30.0, 50.0, 59.94, 60.0])
@pytest.mark.parametrize("family", FAMILIES)
def test_lost_track_keeps_the_identity_for_the_common_time(family, fps):
    n = int(round(fps))
    for gap, same in ((n, True), (n + 1, False)):
        s = _identities(family, [[A]] * 10 + [[]] * gap + [[A]] * 3, fps=fps)
        before, back = s[9][0], 10 + gap
        if same:
            assert s[back].get(0) == before, (family, fps, gap)
        else:
            assert s[back].get(0) is None and s[back + 1][0] != before, (family, fps, gap)


def test_below_the_high_threshold_no_track_is_created_in_any_family():
    for score, creates in ((0.44, False), (0.46, True)):
        frames = [[[100, 100, 150, 250, score, 0]]] * 6
        for family in FAMILIES:
            assert any(_identities(family, frames)) == creates, (family, score)


@pytest.mark.parametrize("high", [0.3, 0.6])
def test_high_threshold_other_than_the_default_acts_the_same_in_the_three(high):
    for score, creates in ((high - 0.01, False), (high + 0.01, True)):
        frames = [[[100, 100, 150, 250, score, 0]]] * 6
        for family in FAMILIES:
            assert any(_identities(family, frames, high_score_thresh=high)) == creates, (family, high, score)


@pytest.mark.parametrize("fps", [14.0, 15.0, 24.0, 25.0, 30.0, 50.0, 59.94, 60.0])
def test_retention_translation_compensates_the_count_of_each_family(fps):
    t = {f: _tracker(f, fps=fps) for f in FAMILIES}
    n = int(round(fps))
    # ByteTrack and BoT-SORT keep the identity one frame of absence past max_time_lost
    assert (t["bytetrack"].max_time_lost, t["ocsort"].max_age, t["botsort"].max_time_lost) == (n - 1, n, n - 1)


def test_ocsort_confirmation_follows_the_common_parameter():
    frames = [[A] if i < 5 else [A, B] for i in range(1, 12)]
    s = _identities("ocsort", frames, confirm_frames=3)
    assert next(i for i, x in enumerate(s, start=1) if 1 in x) == 7


@pytest.mark.parametrize("fps, frame_rate, track_buffer", [(25.0, 25, 29), (30.0, 30, 29), (14.0, 14, 28),
                                                            (24.0, 24, 29), (60.0, 30, 59), (59.94, 30, 59),
                                                            (50.0, 30, 49)])
def test_frame_rate_only_changes_when_the_real_fps_does_not_reach_the_target(fps, frame_rate, track_buffer):
    """BoxMOT uses max_time_lost = int(frame_rate / 30 * track_buffer); above 30 fps frame_rate 30 is passed."""
    common = _common()
    for family in ("bytetrack", "botsort"):
        kw = BT.profile_kwargs(family, {**_profiles()[family], **common}, fps)
        assert (kw["frame_rate"], kw["track_buffer"]) == (frame_rate, track_buffer), (family, fps)


def test_ocsort_constructor_gets_no_min_conf():
    kw = BT.profile_kwargs("ocsort", {**_profiles()["ocsort"], **_common()}, 25.0)
    assert "min_conf" not in kw and kw["use_byte"] is False
    with pytest.raises(ValueError, match="fields outside the profile of the family: min_conf"):
        BT.create("ocsort", {**_profiles()["ocsort"], **_common(), "min_conf": 0.1}, 25.0, "cpu", None)

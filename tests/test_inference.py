"""
Tests that don't require trained weights or a GPU — they cover the pure-logic
parts (IoU math, tracker association, config loading) so CI can run on every
push without needing model artifacts checked into the repo.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracker import ByteTracker, iou_batch  # noqa: E402
from config_loader import load_config  # noqa: E402


def test_iou_identical_boxes():
    box = np.array([[0, 0, 10, 10]])
    result = iou_batch(box, box)
    assert result[0, 0] == pytest.approx(1.0)


def test_iou_no_overlap():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[20, 20, 30, 30]])
    result = iou_batch(a, b)
    assert result[0, 0] == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 5, 15, 15]])
    result = iou_batch(a, b)
    # intersection = 5x5=25, union = 100+100-25=175
    assert result[0, 0] == pytest.approx(25 / 175, rel=1e-3)


def test_tracker_assigns_consistent_id_across_frames():
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, track_buffer=5)

    # Frame 1: one object appears
    det_frame1 = np.array([[10, 10, 50, 50, 0.9, 0]])
    tracks1 = tracker.update(det_frame1)
    assert len(tracks1) == 1
    track_id = tracks1[0]["id"]

    # Frame 2: same object moves slightly -> should keep the same ID
    det_frame2 = np.array([[12, 11, 52, 51, 0.88, 0]])
    tracks2 = tracker.update(det_frame2)
    assert len(tracks2) == 1
    assert tracks2[0]["id"] == track_id


def test_tracker_spawns_new_id_for_new_object():
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, track_buffer=5)
    tracker.update(np.array([[10, 10, 50, 50, 0.9, 0]]))

    # A second, spatially distinct object appears
    tracks = tracker.update(np.array([
        [12, 11, 52, 51, 0.88, 0],
        [200, 200, 240, 240, 0.9, 1],
    ]))
    assert len(tracks) == 2
    assert tracks[0]["id"] != tracks[1]["id"]


def test_tracker_drops_stale_tracks_after_buffer():
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, track_buffer=2)
    tracker.update(np.array([[10, 10, 50, 50, 0.9, 0]]))

    # No detections for more frames than track_buffer allows
    for _ in range(5):
        tracks = tracker.update(np.empty((0, 6)))

    assert len(tracks) == 0
    assert len(tracker.tracks) == 0


def test_config_loader_reads_expected_keys():
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = load_config(config_path)
    assert "dataset" in cfg
    assert "training" in cfg
    assert "inference" in cfg
    assert cfg["inference"]["conf_threshold"] > 0

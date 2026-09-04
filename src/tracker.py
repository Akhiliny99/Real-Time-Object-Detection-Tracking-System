"""
A from-scratch ByteTrack-style multi-object tracker.

ByteTrack's key idea over plain SORT/DeepSORT: instead of throwing away
low-confidence detections before matching, it matches high-confidence boxes
first, then does a *second* matching pass against remaining unmatched tracks
using the low-confidence boxes. This recovers true positives during partial
occlusion or motion blur without needing a re-identification embedding
network, so it stays fast enough for real-time video.

This implementation uses a simple constant-velocity Kalman filter per track
and the Hungarian algorithm (via scipy) for IoU-based assignment.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two sets of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    boxes_a = boxes_a.astype(np.float64)
    boxes_b = boxes_b.astype(np.float64)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter tracking a single box as [cx, cy, w, h, vx, vy, vw, vh]."""

    _next_id = 1

    def __init__(self, box_xyxy: np.ndarray, class_id: int, score: float):
        cx, cy, w, h = self._xyxy_to_cxcywh(box_xyxy)
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        # Process noise: how much we trust the constant-velocity assumption
        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.01
        # Measurement noise: how much we trust a single detection's box
        self.R = np.eye(4) * 10.0
        self.P = np.eye(8) * 10.0

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity each step
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.class_id = class_id
        self.score = score
        self.hits = 1
        self.time_since_update = 0
        self.history: list[np.ndarray] = []

    @staticmethod
    def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

    @staticmethod
    def _cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
        cx, cy, w, h = state[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        box = self._cxcywh_to_xyxy(self.state)
        self.history.append(box)
        return box

    def update(self, box_xyxy: np.ndarray, score: float):
        measurement = self._xyxy_to_cxcywh(box_xyxy)
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.score = score

    def get_box(self) -> np.ndarray:
        return self._cxcywh_to_xyxy(self.state)


class ByteTracker:
    """
    Multi-object tracker following the ByteTrack two-stage association strategy.

    Args:
        track_thresh: min confidence for a detection to spawn a *new* track
        match_thresh: IoU threshold below which a match is rejected
        track_buffer: frames a track survives with no matching detection before removal
        min_box_area: discard tiny boxes (noise) below this pixel area
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: float = 10,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.id_switch_count = 0
        self._last_frame_assignments: dict[int, int] = {}  # detection slot -> track id, for informal ID-switch logging

    @staticmethod
    def _associate(tracks_boxes: np.ndarray, det_boxes: np.ndarray, iou_threshold: float):
        if len(tracks_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(tracks_boxes))), list(range(len(det_boxes)))

        iou_matrix = iou_batch(tracks_boxes, det_boxes)
        cost_matrix = 1 - iou_matrix
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matches, unmatched_tracks, unmatched_dets = [], [], []
        matched_track_idx, matched_det_idx = set(), set()

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                matched_track_idx.add(r)
                matched_det_idx.add(c)

        unmatched_tracks = [i for i in range(len(tracks_boxes)) if i not in matched_track_idx]
        unmatched_dets = [i for i in range(len(det_boxes)) if i not in matched_det_idx]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: np.ndarray) -> list[dict]:
        """
        Args:
            detections: array of shape (N, 6): [x1, y1, x2, y2, score, class_id]

        Returns:
            list of dicts: {id, box (xyxy), class_id, score}, one per active track this frame.
        """
        self.frame_count += 1

        if len(detections) > 0:
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            detections = detections[areas >= self.min_box_area]

        high_conf_mask = detections[:, 4] >= self.track_thresh if len(detections) else np.array([], dtype=bool)
        high_dets = detections[high_conf_mask] if len(detections) else np.empty((0, 6))
        low_dets = detections[~high_conf_mask] if len(detections) else np.empty((0, 6))

        # Predict new locations for all existing tracks
        predicted_boxes = np.array([t.predict() for t in self.tracks]) if self.tracks else np.empty((0, 4))

        # --- Stage 1: match high-confidence detections against all tracks ---
        matches, unmatched_track_idx, unmatched_high_idx = self._associate(
            predicted_boxes, high_dets[:, :4] if len(high_dets) else np.empty((0, 4)), self.match_thresh
        )
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high_dets[det_idx, :4], high_dets[det_idx, 4])

        # --- Stage 2: match remaining unmatched tracks against LOW-confidence detections ---
        remaining_track_boxes = predicted_boxes[unmatched_track_idx] if unmatched_track_idx else np.empty((0, 4))
        matches_low, still_unmatched_track_idx, _ = self._associate(
            remaining_track_boxes, low_dets[:, :4] if len(low_dets) else np.empty((0, 4)), iou_threshold=0.5
        )
        for local_track_idx, det_idx in matches_low:
            real_track_idx = unmatched_track_idx[local_track_idx]
            self.tracks[real_track_idx].update(low_dets[det_idx, :4], low_dets[det_idx, 4])

        really_unmatched_track_idx = [unmatched_track_idx[i] for i in still_unmatched_track_idx]

        # --- Spawn new tracks from unmatched high-confidence detections only ---
        for det_idx in unmatched_high_idx:
            box, score, cls_id = high_dets[det_idx, :4], high_dets[det_idx, 4], int(high_dets[det_idx, 5])
            self.tracks.append(KalmanBoxTracker(box, cls_id, score))

        # --- Remove stale tracks ---
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]

        # --- Emit results for confirmed, currently-updated tracks ---
        results = []
        for t in self.tracks:
            if t.time_since_update == 0:
                results.append({
                    "id": t.id,
                    "box": t.get_box().tolist(),
                    "class_id": t.class_id,
                    "score": float(t.score),
                })
        return results

"""
Shared inference engine used by both the FastAPI service and the Streamlit
demo, so detection logic (and its config) lives in exactly one place.

Supports two backends selected via config['inference']['engine']:
  - "pytorch": uses ultralytics YOLO() directly — simplest, good for dev
  - "onnx":    uses onnxruntime directly — used at serving time for lower
               latency and a lighter dependency footprint
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config_loader import load_config


class DetectionEngine:
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        inf_cfg = self.cfg["inference"]
        self.engine_type = inf_cfg["engine"]
        self.conf_threshold = inf_cfg["conf_threshold"]
        self.iou_threshold = inf_cfg["iou_threshold"]
        self.max_detections = inf_cfg["max_detections"]
        self.class_names = self.cfg["dataset"]["class_names"]
        self.imgsz = self.cfg["export"]["imgsz"]

        if self.engine_type == "pytorch":
            self._load_pytorch()
        elif self.engine_type == "onnx":
            self._load_onnx()
        else:
            raise ValueError(f"Unknown inference engine: {self.engine_type}")

    def _load_pytorch(self):
        from ultralytics import YOLO
        weights_path = self.cfg["export"]["weights_path"]
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"{weights_path} not found. Run src/train.py first.")
        self.model = YOLO(weights_path)

    def _load_onnx(self):
        onnx_path = self.cfg["export"]["onnx_path"]
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"{onnx_path} not found. Run src/export_onnx.py first.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess_onnx(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        h0, w0 = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW
        scale_x, scale_y = w0 / self.imgsz, h0 / self.imgsz
        return img, scale_x, scale_y

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        idxs = scores.argsort()[::-1]
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            rest = idxs[1:]
            from tracker import iou_batch  # reuse the vectorized IoU implementation
            ious = iou_batch(boxes[i:i + 1], boxes[rest])[0]
            idxs = rest[ious < iou_threshold]
        return keep

    def _postprocess_onnx(self, raw_output: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
        # YOLOv8 ONNX output shape: (1, 4 + num_classes, num_boxes) -> transpose to (num_boxes, 4+num_classes)
        preds = raw_output[0].transpose(1, 0)
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)

        mask = scores >= self.conf_threshold
        boxes_cxcywh, scores, class_ids = boxes_cxcywh[mask], scores[mask], class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return np.empty((0, 6))

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # scale back to original frame size
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        keep = self._nms(boxes_xyxy, scores, self.iou_threshold)
        keep = keep[: self.max_detections]

        result = np.concatenate(
            [boxes_xyxy[keep], scores[keep, None], class_ids[keep, None].astype(np.float64)], axis=1
        )
        return result

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns detections as array of shape (N, 6): [x1, y1, x2, y2, score, class_id]."""
        if self.engine_type == "pytorch":
            results = self.model.predict(
                frame_bgr, conf=self.conf_threshold, iou=self.iou_threshold,
                max_det=self.max_detections, verbose=False,
            )[0]
            if results.boxes is None or len(results.boxes) == 0:
                return np.empty((0, 6))
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            return np.concatenate([boxes, scores[:, None], classes[:, None]], axis=1)

        # onnx path
        img, scale_x, scale_y = self._preprocess_onnx(frame_bgr)
        raw_output = self.session.run(None, {self.input_name: img.astype(np.float32)})[0]
        return self._postprocess_onnx(raw_output, scale_x, scale_y)

    def class_name(self, class_id: int) -> str:
        try:
            return self.class_names[int(class_id)]
        except (IndexError, TypeError):
            return f"class_{int(class_id)}"

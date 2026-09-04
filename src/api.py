"""
FastAPI service for real-time object detection + tracking.

Endpoints:
  GET  /health              liveness check
  GET  /metrics             basic request/latency counters
  POST /detect               single image -> detections (no tracking, stateless)
  POST /track/video          uploaded video file -> annotated video with track IDs
  WS   /track/stream          websocket: client streams frames, server streams back
                               detections + track IDs per frame (for live webcam use)

Run:
    uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from config_loader import load_config
from inference import DetectionEngine
from tracker import ByteTracker

app = FastAPI(title="Real-Time Detection & Tracking API", version="1.0.0")

CONFIG_PATH = "config.yaml"
_cfg = load_config(CONFIG_PATH)
_engine: DetectionEngine | None = None  # lazy-loaded so /health works even before model files exist

_metrics = {
    "requests_total": 0,
    "detect_requests": 0,
    "track_requests": 0,
    "total_inference_ms": 0.0,
    "errors_total": 0,
}


def get_engine() -> DetectionEngine:
    global _engine
    if _engine is None:
        _engine = DetectionEngine(CONFIG_PATH)
    return _engine


@app.get("/health")
def health():
    model_ready = True
    try:
        get_engine()
    except FileNotFoundError:
        model_ready = False
    return {"status": "ok", "model_ready": model_ready}


@app.get("/metrics")
def metrics():
    avg_latency = (
        _metrics["total_inference_ms"] / _metrics["detect_requests"]
        if _metrics["detect_requests"] else 0.0
    )
    return {**_metrics, "avg_detect_latency_ms": round(avg_latency, 2)}


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """Runs detection (no tracking) on a single uploaded image."""
    _metrics["requests_total"] += 1
    _metrics["detect_requests"] += 1

    max_mb = _cfg["api"]["max_upload_mb"]
    contents = await file.read()
    if len(contents) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {max_mb}MB limit")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=400, detail="Could not decode image file")

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    engine = get_engine()
    start = time.perf_counter()
    detections = engine.detect(frame_bgr)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _metrics["total_inference_ms"] += elapsed_ms

    results = [
        {
            "box": det[:4].tolist(),
            "score": float(det[4]),
            "class_id": int(det[5]),
            "class_name": engine.class_name(det[5]),
        }
        for det in detections
    ]
    return JSONResponse({
        "detections": results,
        "count": len(results),
        "inference_ms": round(elapsed_ms, 2),
    })


@app.post("/track/video")
async def track_video(file: UploadFile = File(...)):
    """
    Runs detection + ByteTrack tracking on every frame of an uploaded video,
    draws boxes + persistent track IDs, and returns the annotated video file.
    """
    _metrics["requests_total"] += 1
    _metrics["track_requests"] += 1

    max_mb = _cfg["api"]["max_upload_mb"]
    contents = await file.read()
    if len(contents) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {max_mb}MB limit")

    engine = get_engine()
    tracking_cfg = _cfg["tracking"]
    tracker = ByteTracker(
        track_thresh=tracking_cfg["track_thresh"],
        match_thresh=tracking_cfg["match_thresh"],
        track_buffer=tracking_cfg["track_buffer"],
        min_box_area=tracking_cfg["min_box_area"],
    )

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp_in:
        tmp_in.write(contents)
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=400, detail="Could not decode video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        detections = engine.detect(frame)
        tracks = tracker.update(detections)

        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            label = f"ID {t['id']} {engine.class_name(t['class_id'])} {t['score']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    Path(input_path).unlink(missing_ok=True)

    return FileResponse(output_path, media_type="video/mp4", filename="tracked_output.mp4")


@app.websocket("/track/stream")
async def track_stream(websocket: WebSocket):
    """
    Live tracking over a websocket: client sends JPEG-encoded frame bytes,
    server responds with JSON detections+track IDs for that frame.
    Used by the Streamlit demo for a live webcam feed.
    """
    await websocket.accept()
    engine = get_engine()
    tracking_cfg = _cfg["tracking"]
    tracker = ByteTracker(
        track_thresh=tracking_cfg["track_thresh"],
        match_thresh=tracking_cfg["match_thresh"],
        track_buffer=tracking_cfg["track_buffer"],
        min_box_area=tracking_cfg["min_box_area"],
    )

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "could not decode frame"})
                continue

            detections = engine.detect(frame)
            tracks = tracker.update(detections)

            await websocket.send_json({
                "tracks": [
                    {
                        "id": t["id"],
                        "box": t["box"],
                        "class_id": t["class_id"],
                        "class_name": engine.class_name(t["class_id"]),
                        "score": t["score"],
                    }
                    for t in tracks
                ]
            })
    except WebSocketDisconnect:
        pass

"""
Streamlit demo UI.

Two modes:
  - "Upload image": calls POST /detect, draws boxes on the image
  - "Upload video":  calls POST /track/video, shows the annotated result

Run:
    streamlit run src/streamlit_app.py
(with the API already running: uvicorn src.api:app --port 8000)
"""
import io

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

API_URL = st.sidebar.text_input("API base URL", value="http://localhost:8000")

st.set_page_config(page_title="Detect & Track Demo", layout="wide")
st.title("Real-Time Object Detection & Tracking")
st.caption("YOLOv8 (fine-tuned) + ByteTrack, served via FastAPI + ONNX Runtime")

mode = st.sidebar.radio("Mode", ["Image detection", "Video tracking", "API status"])

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
]


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = COLORS[det["class_id"] % len(COLORS)]
        label = f"{det['class_name']} {det['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 16), label, fill=(255, 255, 255))
    return image


if mode == "API status":
    st.subheader("Health & metrics")
    col1, col2 = st.columns(2)
    if st.button("Refresh"):
        pass
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        col1.metric("Model ready", "Yes" if health.get("model_ready") else "No")
        metrics = requests.get(f"{API_URL}/metrics", timeout=5).json()
        col2.metric("Avg detect latency (ms)", metrics.get("avg_detect_latency_ms"))
        st.json(metrics)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")

elif mode == "Image detection":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    conf_note = st.caption("Confidence/IoU thresholds are controlled server-side via config.yaml")

    if uploaded is not None:
        image_bytes = uploaded.read()
        st.image(image_bytes, caption="Input", use_column_width=True)

        if st.button("Run detection"):
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/detect",
                        files={"file": (uploaded.name, image_bytes, uploaded.type)},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            st.success(f"{result['count']} detection(s) in {result['inference_ms']} ms")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            annotated = draw_detections(image, result["detections"])
            st.image(annotated, caption="Detections", use_column_width=True)

            if result["detections"]:
                st.subheader("Raw detections")
                st.dataframe(result["detections"])

elif mode == "Video tracking":
    uploaded = st.file_uploader("Upload a short video clip", type=["mp4", "mov", "avi"])

    if uploaded is not None:
        st.video(uploaded)

        if st.button("Run detection + tracking"):
            with st.spinner("Processing video — this can take a while for longer clips..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/track/video",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=600,
                    )
                    resp.raise_for_status()
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            st.success("Done — annotated video below")
            st.video(resp.content)
            st.download_button(
                "Download annotated video", data=resp.content,
                file_name="tracked_output.mp4", mime="video/mp4",
            )

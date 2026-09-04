"""
Exports a fine-tuned YOLOv8 (.pt) model to ONNX and benchmarks inference
latency of PyTorch vs ONNX Runtime, mirroring the ONNX-speedup measurement
pattern used in the CropShield project (~340ms -> ~21ms there).

Usage:
    python src/export_onnx.py --weights models/best.pt --imgsz 640
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from config_loader import load_config


def benchmark_pytorch(model: YOLO, dummy_image: np.ndarray, n_runs: int = 50, device: str = "cpu") -> list[float]:
    # Warmup
    for _ in range(5):
        model.predict(dummy_image, verbose=False, device=device)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(dummy_image, verbose=False, device=device)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    return latencies


def benchmark_onnx(onnx_path: str, imgsz: int, n_runs: int = 50) -> list[float]:
    # Forced to CPU to match benchmark_pytorch's device — apples-to-apples
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def summarize(name: str, latencies: list[float]) -> dict:
    arr = np.array(latencies)
    summary = {
        "engine": name,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
    print(f"{name:>12}: mean={summary['mean_ms']:.2f}ms  p50={summary['p50_ms']:.2f}ms  p95={summary['p95_ms']:.2f}ms")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights to ONNX and benchmark.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default=None, help="Override config['export']['weights_path']")
    parser.add_argument("--imgsz", type=int, default=None, help="Override config['export']['imgsz']")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_cfg = cfg["export"]

    weights_path = args.weights or export_cfg["weights_path"]
    imgsz = args.imgsz or export_cfg["imgsz"]

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"{weights_path} not found. Run src/train.py first.")

    print(f"Loading {weights_path} ...")
    model = YOLO(weights_path)

    print(f"Exporting to ONNX (imgsz={imgsz}, opset={export_cfg['opset']}) ...")
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=export_cfg["opset"],
        dynamic=export_cfg["dynamic"],
        simplify=export_cfg["simplify"],
    )

    target_path = Path(export_cfg["onnx_path"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if str(exported_path) != str(target_path):
        Path(exported_path).replace(target_path)
    print(f"ONNX model written to {target_path}")

    print(f"\nBenchmarking ({args.runs} runs each, {imgsz}x{imgsz})...")
    dummy_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    pt_latencies = benchmark_pytorch(model, dummy_image, args.runs, device="cpu")
    onnx_latencies = benchmark_onnx(str(target_path), imgsz, args.runs)

    pt_summary = summarize("PyTorch", pt_latencies)
    onnx_summary = summarize("ONNX Runtime", onnx_latencies)

    speedup = pt_summary["mean_ms"] / onnx_summary["mean_ms"]
    print(f"\nONNX Runtime is {speedup:.1f}x faster than PyTorch on mean latency.")


if __name__ == "__main__":
    main()

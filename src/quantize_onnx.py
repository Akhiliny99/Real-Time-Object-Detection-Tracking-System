"""
Quantizes the FP32 ONNX model (from export_onnx.py) to dynamic INT8 and
benchmarks latency against the FP32 version, both on CPU.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from config_loader import load_config


def benchmark_onnx(onnx_path, imgsz, n_runs=50):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(5):
        session.run(None, {input_name: dummy_input})
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)
    arr = np.array(latencies)
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_cfg = cfg["export"]
    imgsz = export_cfg["imgsz"]

    fp32_path = Path(export_cfg["onnx_path"])
    if not fp32_path.exists():
        raise FileNotFoundError(f"{fp32_path} not found. Run src/export_onnx.py first.")

    int8_path = fp32_path.with_name(fp32_path.stem + "_int8.onnx")

    print(f"Quantizing {fp32_path} -> {int8_path} ...")
    quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path),
                      weight_type=QuantType.QUInt8)
    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)
    print(f"FP32: {fp32_size:.2f} MB -> INT8: {int8_size:.2f} MB ({fp32_size/int8_size:.2f}x smaller)")

    print(f"\nBenchmarking ({args.runs} runs, {imgsz}x{imgsz}, CPU) ...")
    fp32_summary = benchmark_onnx(fp32_path, imgsz, args.runs)
    int8_summary = benchmark_onnx(int8_path, imgsz, args.runs)
    print(f"  FP32 ONNX: mean={fp32_summary['mean_ms']:.2f}ms")
    print(f"  INT8 ONNX: mean={int8_summary['mean_ms']:.2f}ms")
    speedup = fp32_summary["mean_ms"] / int8_summary["mean_ms"]
    print(f"\nINT8 is {speedup:.2f}x the speed of FP32 ({'faster' if speedup>1 else 'slower'}).")


if __name__ == "__main__":
    main()

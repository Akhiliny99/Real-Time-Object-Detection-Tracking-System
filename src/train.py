"""
Fine-tunes a YOLOv8 model (PyTorch) on a custom dataset.

Usage:
    python src/train.py --config config.yaml
    python src/train.py --config config.yaml --epochs 50 --device cpu   # overrides

After training, the best checkpoint is copied to the path set in
config['export']['weights_path'] so export_onnx.py can pick it up directly.
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a custom dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config['training']['epochs']")
    parser.add_argument("--device", default=None, help="Override config['training']['device']")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint of this run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]

    epochs = args.epochs or train_cfg["epochs"]
    device = args.device or train_cfg["device"]

    if device != "cpu" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU. Training will be slow.")
        device = "cpu"

    data_yaml = Path(dataset_cfg["data_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run src/data_prep.py first to generate it."
        )

    print(f"Loading base model: {train_cfg['base_model']}")
    model = YOLO(train_cfg["base_model"])

    print(
        f"Fine-tuning for {epochs} epochs on {data_yaml} "
        f"(imgsz={train_cfg['imgsz']}, batch={train_cfg['batch']}, device={device})"
    )
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        patience=train_cfg["patience"],
        device=device,
        project=train_cfg["project_dir"],
        name=train_cfg["run_name"],
        lr0=train_cfg["lr0"],
        optimizer=train_cfg["optimizer"],
        augment=train_cfg["augment"],
        resume=args.resume,
        exist_ok=True,
    )

    # Ultralytics writes best.pt under <project_dir>/<run_name>/weights/best.pt
    run_dir = Path(train_cfg["project_dir"]) / train_cfg["run_name"]
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(f"Expected best weights at {best_weights} but none were found.")

    export_target = Path(cfg["export"]["weights_path"])
    export_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, export_target)
    print(f"\nBest weights copied to {export_target}")

    # Quick validation summary
    metrics = model.val(data=str(data_yaml), device=device)
    print("\n=== Validation summary ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Full training run + plots saved under: {run_dir}")


if __name__ == "__main__":
    main()

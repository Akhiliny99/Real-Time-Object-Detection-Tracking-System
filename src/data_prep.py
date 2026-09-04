"""
Prepares a YOLOv8-format dataset for training.

Expects a Roboflow-style export (or anything already in YOLOv8 layout):

    source/
      train/images/*.jpg   train/labels/*.txt
      valid/images/*.jpg   valid/labels/*.txt
      test/images/*.jpg    test/labels/*.txt   (optional)
      data.yaml

This script:
  1. Validates every image has a matching label file (and vice versa)
  2. Checks label files are well-formed (class_id x y w h, all in [0, 1])
  3. Copies/links the dataset into `output/` in a clean layout
  4. Writes a `data.yaml` pointing at the new layout, ready for train.py

Usage:
    python src/data_prep.py --source data/raw --output data/processed
"""
import argparse
import shutil
from pathlib import Path

import yaml


def validate_split(images_dir: Path, labels_dir: Path, split_name: str) -> list[str]:
    """Returns a list of human-readable problems found in this split."""
    problems = []
    if not images_dir.exists():
        problems.append(f"[{split_name}] missing images dir: {images_dir}")
        return problems
    if not labels_dir.exists():
        problems.append(f"[{split_name}] missing labels dir: {labels_dir}")
        return problems

    image_files = {p.stem: p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    label_files = {p.stem: p for p in labels_dir.glob("*.txt")}

    missing_labels = set(image_files) - set(label_files)
    orphan_labels = set(label_files) - set(image_files)

    for stem in missing_labels:
        problems.append(f"[{split_name}] image with no label: {image_files[stem].name}")
    for stem in orphan_labels:
        problems.append(f"[{split_name}] label with no image: {label_files[stem].name}")

    for stem, label_path in label_files.items():
        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"[{split_name}] {label_path.name}:{line_no} expected 5 fields, got {len(parts)}")
                continue
            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                problems.append(f"[{split_name}] {label_path.name}:{line_no} non-numeric field")
                continue
            if cls_id < 0:
                problems.append(f"[{split_name}] {label_path.name}:{line_no} negative class id")
            if not all(0.0 <= c <= 1.0 for c in coords):
                problems.append(f"[{split_name}] {label_path.name}:{line_no} coords out of [0,1] range")

    print(f"  {split_name}: {len(image_files)} images, {len(label_files)} labels, "
          f"{len(missing_labels)} missing labels, {len(orphan_labels)} orphan labels")
    return problems


def copy_split(source_root: Path, output_root: Path, split_name: str) -> None:
    src_images = source_root / split_name / "images"
    src_labels = source_root / split_name / "labels"
    if not src_images.exists():
        return
    dst_images = output_root / split_name / "images"
    dst_labels = output_root / split_name / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for f in src_images.iterdir():
        shutil.copy2(f, dst_images / f.name)
    if src_labels.exists():
        for f in src_labels.iterdir():
            shutil.copy2(f, dst_labels / f.name)


def main():
    parser = argparse.ArgumentParser(description="Validate and prepare a YOLOv8 dataset.")
    parser.add_argument("--source", required=True, help="Path to raw Roboflow-format export")
    parser.add_argument("--output", required=True, help="Where to write the cleaned dataset")
    parser.add_argument("--class-names", nargs="*", default=None,
                         help="Override class names; otherwise read from source data.yaml")
    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Validating dataset at {source_root} ...")
    all_problems = []
    for split in ("train", "valid", "test"):
        all_problems += validate_split(source_root / split / "images", source_root / split / "labels", split)

    if all_problems:
        print(f"\nFound {len(all_problems)} issue(s):")
        for p in all_problems[:50]:
            print(f"  - {p}")
        if len(all_problems) > 50:
            print(f"  ... and {len(all_problems) - 50} more")
        print("\nFix these before training, or proceed knowingly (bad labels will hurt mAP).")
    else:
        print("No structural issues found.")

    print(f"\nCopying dataset into {output_root} ...")
    for split in ("train", "valid", "test"):
        copy_split(source_root, output_root, split)

    # Class names: prefer explicit override, then source data.yaml, else config default
    class_names = args.class_names
    src_yaml = source_root / "data.yaml"
    if class_names is None and src_yaml.exists():
        src_config = yaml.safe_load(src_yaml.read_text())
        class_names = src_config.get("names")

    if class_names is None:
        raise ValueError(
            "No class names found. Pass --class-names, or ensure source/data.yaml has a 'names' field."
        )

    data_yaml_content = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images" if (output_root / "test").exists() else "valid/images",
        "nc": len(class_names),
        "names": list(class_names),
    }
    out_yaml_path = output_root / "data.yaml"
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml_content, f, sort_keys=False)

    print(f"\nDone. Wrote {out_yaml_path} — ready for src/train.py")


if __name__ == "__main__":
    main()

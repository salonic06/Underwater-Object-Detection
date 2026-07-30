"""
Prepare URPC 2019 (reduced subset) for YOLO training at 640x640.

Reads the local reduced dataset (train/Images + train/Labels layout),
letterboxes images to 640x640, fixes/validates YOLO labels, and writes
an Ultralytics-compatible folder:

    output/
      data.yaml
      train/images  train/labels
      val/images    val/labels
      test/images   test/labels

Usage:
    py -3 prepare_urpc640.py
    py -3 prepare_urpc640.py --source "path/to/URPC2019_Reduced" --output "./data/urpc2019-640"
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

CLASSES = ["echinus", "starfish", "holothurian", "scallop", "waterweeds"]
NUM_CLASSES = len(CLASSES)
IMGSZ = 640
PAD_COLOR = (114, 114, 114)
SPLITS = ("train", "val", "test")


def default_source() -> Path:
    return Path(__file__).resolve().parent.parent / "BTP" / "URPC2019"


def default_output() -> Path:
    return Path(__file__).resolve().parent / "data" / "urpc2019-640"


def letterbox(image: Image.Image, size: int = IMGSZ):
    """Resize with aspect ratio preserved and gray padding."""
    w, h = image.size
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def transform_labels(
    lines: list[str],
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    size: int = IMGSZ,
) -> tuple[list[str], int]:
    """Convert normalized YOLO boxes from original image to letterboxed 640 image."""
    valid: list[str] = []
    skipped = 0

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            skipped += 1
            continue

        # URPC source labels are 1-indexed: 1=echinus ... 5=waterweeds.
        # YOLO / Ultralytics require 0-indexed IDs matching CLASSES above.
        cls_id = int(float(parts[0]))
        if 1 <= cls_id <= NUM_CLASSES:
            cls_id -= 1
        else:
            skipped += 1
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        x1 = (xc - bw / 2) * orig_w
        y1 = (yc - bh / 2) * orig_h
        x2 = (xc + bw / 2) * orig_w
        y2 = (yc + bh / 2) * orig_h

        x1 = x1 * scale + pad_x
        x2 = x2 * scale + pad_x
        y1 = y1 * scale + pad_y
        y2 = y2 * scale + pad_y

        x1 = max(0.0, min(size, x1))
        x2 = max(0.0, min(size, x2))
        y1 = max(0.0, min(size, y1))
        y2 = max(0.0, min(size, y2))

        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        nxc = ((x1 + x2) / 2) / size
        nyc = ((y1 + y2) / 2) / size
        nbw = (x2 - x1) / size
        nbh = (y2 - y1) / size

        valid.append(f"{cls_id} {nxc:.6f} {nyc:.6f} {nbw:.6f} {nbh:.6f}")

    return valid, skipped


def find_split_dirs(source: Path, split: str) -> tuple[Path, Path]:
    """Support Images/Labels or images/labels folder naming."""
    split_dir = source / split
    for img_name, lbl_name in (("Images", "Labels"), ("images", "labels")):
        img_dir = split_dir / img_name
        lbl_dir = split_dir / lbl_name
        if img_dir.is_dir() and lbl_dir.is_dir():
            return img_dir, lbl_dir
    raise FileNotFoundError(f"Could not find image/label folders under {split_dir}")


def sample_image_files(
    image_files: list[Path], fraction: float, seed: int, split: str
) -> list[Path]:
    """Randomly sample a fraction of images per split (reproducible)."""
    if fraction >= 1.0:
        return image_files

    rng = random.Random(seed + hash(split) % 10000)
    n = max(1, int(round(len(image_files) * fraction)))
    if n >= len(image_files):
        return image_files
    return sorted(rng.sample(image_files, n))


def process_split(
    source: Path, output: Path, split: str, fraction: float = 1.0, seed: int = 88
) -> dict:
    img_dir, lbl_dir = find_split_dirs(source, split)
    out_img = output / split / "images"
    out_lbl = output / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    images_ok = 0
    labels_skipped = 0
    missing_labels = 0

    all_images = sorted(
        f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_files = sample_image_files(all_images, fraction, seed, split)

    for img_path in image_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_labels += 1
            continue

        with lbl_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
            boxed, scale, pad_x, pad_y = letterbox(im)

        new_lines, skipped = transform_labels(
            lines, orig_w, orig_h, scale, pad_x, pad_y
        )
        labels_skipped += skipped

        if not new_lines:
            continue

        boxed.save(out_img / img_path.name, quality=95)
        with (out_lbl / f"{img_path.stem}.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        for line in new_lines:
            class_counts[int(line.split()[0])] += 1
        images_ok += 1

    return {
        "split": split,
        "images": images_ok,
        "source_images": len(all_images),
        "sampled_images": len(image_files),
        "instances": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "labels_skipped": labels_skipped,
        "missing_labels": missing_labels,
    }


def write_data_yaml(output: Path) -> None:
    data = {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": NUM_CLASSES,
        "names": CLASSES,
    }
    with (output / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare URPC reduced dataset at 640x640.")
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source(),
        help="Path to URPC2019 folder (contains train/val/test folders)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output(),
        help="Output directory for Ultralytics-ready dataset",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of each split to sample (e.g. 0.15 for 15%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=88,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before writing",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source not found: {args.source}")

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Source   : {args.source}")
    print(f"Output   : {args.output}")
    print(f"Fraction : {args.fraction:.0%} per split (seed={args.seed})")
    print(f"Image size: {IMGSZ}x{IMGSZ} (letterbox)")
    print()

    stats = []
    for split in SPLITS:
        stats.append(
            process_split(args.source, args.output, split, args.fraction, args.seed)
        )

    write_data_yaml(args.output)

    print("Done.\n")
    print(
        f"{'split':<6} {'sampled':>8} {'of':>6} {'total':>8} "
        f"{'instances':>10} {'skipped':>8}"
    )
    print("-" * 58)
    for row in stats:
        print(
            f"{row['split']:<6} {row['sampled_images']:>8} {'of':>6} "
            f"{row['source_images']:>8} {row['instances']:>10} "
            f"{row['labels_skipped']:>8}"
        )

    print("\nClass names:", CLASSES)
    # Aggregate class counts across splits (sanity: echinus must be non-zero)
    total_cls: Counter = Counter()
    for row in stats:
        total_cls.update(row.get("class_counts", {}))
    print("Class instance counts (all splits):")
    for i, name in enumerate(CLASSES):
        print(f"  {i} {name:<12} {total_cls.get(i, 0)}")
    if total_cls.get(0, 0) == 0:
        print("WARNING: echinus count is 0 — check that source labels are 1-indexed URPC.")
    print(f"\ndata.yaml written to: {args.output / 'data.yaml'}")
    print("\nNext steps:")
    print("  1. Zip the output folder and upload to Kaggle as dataset urpc2019-640-15pct-v2")
    print("  2. Open underwater-yolo-v2 on Kaggle, attach the NEW dataset, enable GPU")
    print("  3. Confirm class counts show echinus > 0, then full training")
    print("  4. Save Version and download yolov9c_urpc640_15pct_best.pt")


if __name__ == "__main__":
    main()

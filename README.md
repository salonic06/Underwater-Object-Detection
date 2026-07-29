# Underwater Object Detection with YOLOv9

Fine-tuned **YOLOv9c** for marine object detection on the URPC 2019 underwater dataset using Ultralytics.

**Kaggle notebook:** https://www.kaggle.com/code/salonichippa/underwater-yolo-v2  
**Executed run (with outputs):** see saved versions on the [Kaggle notebook](https://www.kaggle.com/code/salonichippa/underwater-yolo-v2) — metrics and plots are also in [`results/RESULTS.md`](results/RESULTS.md) and [`assets/`](assets/).

## Results (test set)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|----------|
| **YOLOv9c** | **71.95%** | 38.30% | 77.43% | 62.39% |
| YOLO11s (baseline) | 66.71% | 35.58% | 73.00% | 60.88% |

**Dataset:** 15% stratified subset of URPC 2019 — 508 train / 141 val / 56 test images at 640×640  
**Classes:** echinus, starfish, holothurian, scallop, waterweeds

See [results/RESULTS.md](results/RESULTS.md) for per-class breakdown and experiment history.

### Visualizations

| Training metrics (all epochs) | Precision–Recall curve |
|---|---|
| ![Training results](assets/training_results.png) | ![PR curve](assets/pr_curve.png) |

| Confusion matrix (normalized) |
|---|
| ![Confusion matrix](assets/confusion_matrix_normalized.png) |

Detection batch grids were omitted — raw validation overlays are cluttered due to dense underwater scenes and low-confidence waterweeds detections. Metric plots better represent model performance.

## Project highlights

- Built preprocessing pipeline (`prepare_urpc640.py`) — letterbox resize, label validation, stratified sampling
- Compared **YOLOv9c vs YOLO11s** on the same subset (v9c +5.2% test mAP)
- Scaled from 5% to 15% data: **+9.2 pts** test mAP improvement
- End-to-end workflow: preprocess locally → train on Kaggle GPU → evaluate → visualize predictions

## Repository structure

```
├── prepare_urpc640.py          # Create 640×640 YOLO dataset from URPC 2019
├── notebooks/
│   └── underwater-yolo-v2.ipynb  # Training and evaluation notebook
├── results/
│   └── RESULTS.md              # Full metrics tables
└── assets/                     # Training plots and validation batch visuals
```

## Quick start

### 1. Preprocess dataset (local)

```bash
pip install -r requirements-preprocess.txt
python prepare_urpc640.py --fraction 0.15 --output ./data/urpc2019-640-15pct --clean
```

Upload the output folder to [Kaggle Datasets](https://www.kaggle.com/datasets) as `urpc2019-640-15pct`.

### 2. Train on Kaggle

1. Open the [Kaggle notebook](https://www.kaggle.com/code/salonichippa/underwater-yolo-v2)
2. Attach dataset `urpc2019-640-15pct`
3. Enable **GPU T4**, Internet **On**
4. Run all cells

Or upload `notebooks/underwater-yolo-v2.ipynb` to your own Kaggle notebook.

## Method

| Component | Choice |
|-----------|--------|
| Model | YOLOv9c (Ultralytics) |
| Input size | 640×640 (letterbox) |
| Optimizer | AdamW + cosine LR |
| Augmentation | HSV (underwater color), mixup, copy-paste |
| Hardware | Tesla T4 GPU (Kaggle) |

## Limitations

- Trained on 15% of URPC 2019 (~705 images), not the full benchmark
- Scallop detection remains challenging due to class imbalance
- Not SOTA vs papers using full URPC + custom architectures (~81%+ mAP)

## References

- URPC 2019: Underwater Robot Picking Contest dataset
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- YOLOv9: [arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)

## Author

Saloni Chippa — B.Tech research project, IIITP

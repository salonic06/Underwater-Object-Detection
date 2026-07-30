# Results Summary

## Final test metrics (corrected labels, 15% URPC 2019, 640×640)

Dataset rebuilt with **0-indexed class IDs** (`prepare_urpc640.py` converts URPC’s 1-based labels).  
Kaggle dataset: `urpc2019-640-15pct-v2`

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|----------|
| **YOLOv9c** (primary) | **71.67%** | 39.18% | 71.50% | 71.51% |
| YOLO11s (baseline) | 63.57% | 32.23% | 71.59% | 56.62% |

**Dataset:** 15% stratified subset — test split 56 images / 499 instances  
**Classes:** echinus, starfish, holothurian, scallop, waterweeds

## Per-class test mAP@0.5 (YOLOv9c)

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| Echinus | 48 | 280 | 68.9% | 82.9% | **80.1%** | 39.6% |
| Starfish | 32 | 80 | 86.1% | 85.5% | **88.5%** | 51.2% |
| Holothurian | 32 | 83 | 66.8% | 60.5% | 60.9% | 32.7% |
| Scallop | 11 | 56 | 64.2% | 57.1% | 57.2% | 33.2% |

Waterweeds: very few / none in this test split after correct remapping (rare in source as class 5).

## Per-class test mAP@0.5 (YOLO11s baseline)

| Class | Instances | mAP@0.5 |
|-------|-----------|---------|
| Echinus | 280 | 81.3% |
| Starfish | 80 | 77.8% |
| Holothurian | 83 | 51.2% |
| Scallop | 56 | 44.0% |

## Training progression

| Experiment | Labels | Data | Model | Test mAP@0.5 |
|------------|--------|------|-------|--------------|
| Phase A | shifted (bug) | ~5% | YOLOv9c | 62.8%* |
| Phase B | shifted (bug) | 15% | YOLOv9c | 72.0%* |
| **Phase C (final)** | **corrected 0-index** | **15% v2** | **YOLOv9c** | **71.67%** |
| Phase C | corrected 0-index | 15% v2 | YOLO11s | 63.57% |

\*Phase A/B headline mAP was similar, but **per-class names were wrong** (echinus labeled as starfish, etc.) because source IDs were 1-based and never decremented.

## Training config (Phase C)

- Model: `yolov9c.pt` via Ultralytics 8.4
- Epochs: early-stopped (~37 from 80 max), batch=4, imgsz=640
- Optimizer: AdamW, cos_lr=True
- Augmentation: HSV (underwater), mixup=0.05, copy_paste=0.1
- Hardware: Kaggle Tesla T4 GPU

## Known limitations

- 15% subset of URPC 2019 (~705 images), not the full benchmark
- Scallop / holothurian harder than echinus / starfish
- Waterweeds under-represented after correct label mapping
- Not SOTA vs full-URPC papers (~81%+ mAP)

# Results Summary

## Final test metrics (15% URPC 2019 subset, 640×640)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|----------|
| **YOLOv9c** (primary) | **71.95%** | 38.30% | 77.43% | 62.39% |
| YOLO11s (baseline) | 66.71% | 35.58% | 73.00% | 60.88% |

**Dataset:** 508 train / 141 val / 56 test images (15% stratified subset of [URPC 2019](https://github.com/xiaoHaiSea/URPC2019))

## Per-class test mAP@0.5 (YOLOv9c)

| Class | Instances | mAP@0.5 |
|-------|-----------|---------|
| Starfish | 262 | 90.2% |
| Holothurian | 85 | 90.0% |
| Scallop | 37 | 55.4% |
| Waterweeds | 59 | 52.3% |

## Training progression

| Experiment | Data | Model | Val mAP@0.5 | Test mAP@0.5 |
|------------|------|-------|-------------|--------------|
| Phase A | ~5% (234 img) | YOLOv9c | 69.3% | 62.8% |
| Phase A | ~5% | YOLO11s | 61.2% | — |
| **Phase B** | **15% (705 img)** | **YOLOv9c** | **76.3%** | **72.0%** |
| Phase B | 15% | YOLO11s | 69.1% | 66.7% |

## Training config (best run)

- Model: `yolov9c.pt` via Ultralytics 8.4
- Epochs: 55 (early stop from 80 max), batch=4, imgsz=640
- Optimizer: AdamW, cos_lr=True
- Augmentation: HSV (underwater), mixup=0.05, copy_paste=0.1
- Hardware: Kaggle Tesla T4 GPU

## Known limitations

- 15% subset of URPC 2019 (not full 4,700-image benchmark)
- Scallop class has fewest instances → lowest mAP
- Echinus rare/absent in test split

"""
Gradio demo for underwater object detection (YOLOv9c on URPC 2019).

Run locally:
  pip install -r requirements-deploy.txt
  python app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from ultralytics import YOLO

CLASSES = ["echinus", "starfish", "holothurian", "scallop", "waterweeds"]
DEFAULT_CONF = 0.50
IMGSZ = 640

ROOT = Path(__file__).resolve().parent
WEIGHT_CANDIDATES = [
    Path(os.environ["WEIGHTS_PATH"]) if os.environ.get("WEIGHTS_PATH") else None,
    ROOT / "weights" / "yolov9c_urpc640_15pct_best.pt",
    ROOT / "yolov9c_urpc640_15pct_best.pt",
]

_model: Optional[YOLO] = None


def resolve_weights() -> Path:
    for path in WEIGHT_CANDIDATES:
        if path is not None and path.is_file():
            return path
    raise FileNotFoundError(
        "Model weights not found. Place "
        "weights/yolov9c_urpc640_15pct_best.pt next to app.py "
        "or set WEIGHTS_PATH."
    )


def get_model() -> YOLO:
    global _model
    if _model is None:
        weights = resolve_weights()
        print(f"Loading model from {weights} ...", flush=True)
        _model = YOLO(str(weights))
        print("Model ready.", flush=True)
    return _model


def detect(image, conf_threshold: float) -> Tuple[Optional[np.ndarray], str]:
    if image is None:
        return None, "Upload an underwater image to run detection."

    model = get_model()
    results = model.predict(
        source=image,
        conf=float(conf_threshold),
        imgsz=IMGSZ,
        verbose=False,
    )
    result = results[0]
    annotated = result.plot()  # BGR
    annotated_rgb = annotated[:, :, ::-1]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated_rgb, f"No detections above confidence {conf_threshold:.2f}."

    lines = [f"Detections (conf >= {conf_threshold:.2f}):", ""]
    counts = {}
    for box in boxes:
        cls_id = int(box.cls.item())
        name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
        score = float(box.conf.item())
        counts[name] = counts.get(name, 0) + 1
        lines.append(f"- {name}: {score:.2f}")

    lines.append("")
    lines.append("Counts: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    return annotated_rgb, "\n".join(lines)


DESCRIPTION = (
    "Upload an underwater image to detect marine objects with fine-tuned **YOLOv9c** "
    "(URPC 2019, 15% subset, corrected 0-indexed labels). "
    "Test mAP@0.5: **71.67%**. "
    "Classes: echinus, starfish, holothurian, scallop, waterweeds. "
    "Tip: use confidence 0.50–0.60 for cleaner boxes."
)

examples_dir = ROOT / "examples"
example_paths = sorted(
    str(p)
    for p in examples_dir.glob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Input image"),
        gr.Slider(0.15, 0.90, value=DEFAULT_CONF, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Detections"),
        gr.Textbox(label="Summary", lines=8),
    ],
    title="Underwater Object Detection (YOLOv9c)",
    description=DESCRIPTION,
    examples=[[p, DEFAULT_CONF] for p in example_paths] or None,
    allow_flagging="never",
)


if __name__ == "__main__":
    print("Starting Gradio (model loads on first request)...", flush=True)
    # Avoid Gradio 4.x + newer FastAPI schema bug on /info
    gr.blocks.Blocks.get_api_info = lambda self: {"named_endpoints": {}, "unnamed_endpoints": {}}
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_api=False)

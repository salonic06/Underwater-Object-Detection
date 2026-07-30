"""
Streamlit demo for underwater object detection (YOLOv9c / URPC 2019).

Local:
  streamlit run streamlit_app.py

Streamlit Cloud: set main file to streamlit_app.py
Weights: local weights/*.pt or auto-download from GitHub Release (WEIGHTS_URL).
"""

from __future__ import annotations

import os
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Writable config dir on Streamlit Cloud
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

CLASSES = ["echinus", "starfish", "holothurian", "scallop", "waterweeds"]
DEFAULT_CONF = 0.50
IMGSZ = 640
WEIGHT_NAME = "yolov9c_urpc640_15pct_best_v2.pt"

# Create a GitHub Release named v1.0 and attach the .pt file (see README).
DEFAULT_WEIGHTS_URL = (
    "https://github.com/salonic06/Underwater-Object-Detection/releases/download/"
    f"v1.0/{WEIGHT_NAME}"
)

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
LOCAL_WEIGHTS = WEIGHTS_DIR / WEIGHT_NAME
# Also accept the local Gradio filename if present
LOCAL_WEIGHTS_ALIASES = [
    LOCAL_WEIGHTS,
    WEIGHTS_DIR / "yolov9c_urpc640_15pct_best.pt",
]


def _weights_url() -> str:
    return os.environ.get("WEIGHTS_URL", DEFAULT_WEIGHTS_URL)


def ensure_weights() -> Path:
    """Return path to weights file, downloading from GitHub Release if needed."""
    for path in LOCAL_WEIGHTS_ALIASES:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path

    # Streamlit Cloud: download into /tmp (writable)
    cache = Path(os.environ.get("WEIGHTS_CACHE", "/tmp")) / WEIGHT_NAME
    if cache.is_file() and cache.stat().st_size > 1_000_000:
        return cache

    url = _weights_url()
    dest = cache if os.access("/tmp", os.W_OK) else LOCAL_WEIGHTS
    dest.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner(f"Downloading model weights (~50 MB)…\n{url}"):
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not download weights from:\n{url}\n\n"
                "Fix: create GitHub Release tag v1.0 and upload "
                f"{WEIGHT_NAME}, or place the file at weights/{WEIGHT_NAME}.\n"
                f"Details: {exc}"
            ) from exc

    if not dest.is_file() or dest.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Downloaded weights look invalid: {dest}")
    return dest


@st.cache_resource(show_spinner="Loading YOLOv9c model…")
def load_model():
    from ultralytics import YOLO

    path = ensure_weights()
    return YOLO(str(path))


def run_detect(image: np.ndarray, conf: float):
    model = load_model()
    results = model.predict(
        source=image,
        conf=float(conf),
        imgsz=IMGSZ,
        verbose=False,
    )
    result = results[0]
    annotated = result.plot()[:, :, ::-1]  # BGR → RGB

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, f"No detections above confidence {conf:.2f}."

    lines = [f"**Detections** (conf ≥ {conf:.2f})", ""]
    counts: Counter[str] = Counter()
    for box in boxes:
        cls_id = int(box.cls.item())
        name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
        score = float(box.conf.item())
        counts[name] += 1
        lines.append(f"- {name}: `{score:.2f}`")

    lines.append("")
    lines.append(
        "**Counts:** " + ", ".join(f"{k} ×{v}" for k, v in sorted(counts.items()))
    )
    return annotated, "\n".join(lines)


def main() -> None:
    st.set_page_config(
        page_title="Underwater Object Detection",
        page_icon="🌊",
        layout="wide",
    )

    st.title("Underwater Object Detection (YOLOv9c)")
    st.markdown(
        "Fine-tuned on **URPC 2019** (15% subset, corrected labels). "
        "Test **mAP@0.5 = 71.67%**. "
        "Classes: echinus · starfish · holothurian · scallop · waterweeds."
    )
    st.caption(
        "Repo: [salonic06/Underwater-Object-Detection](https://github.com/salonic06/Underwater-Object-Detection) · "
        "Kaggle: [underwater-yolo-v2](https://www.kaggle.com/code/salonichippa/underwater-yolo-v2)"
    )

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.15, 0.90, DEFAULT_CONF, 0.05)
        st.markdown(
            "Tip: **0.50–0.60** for cleaner boxes; lower to catch more objects."
        )
        st.divider()
        st.markdown("### Examples")
        example_files = sorted(
            (ROOT / "examples").glob("*")
        )
        example_files = [
            p for p in example_files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        chosen_example = st.selectbox(
            "Load example image",
            options=["(none)"] + [p.name for p in example_files],
        )

    col_in, col_out = st.columns(2)

    image = None
    uploaded = st.file_uploader(
        "Upload an underwater image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        image = np.array(Image.open(uploaded).convert("RGB"))
    elif chosen_example != "(none)":
        image = np.array(Image.open(ROOT / "examples" / chosen_example).convert("RGB"))

    with col_in:
        st.subheader("Input")
        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.info("Upload an image or pick an example in the sidebar.")

    with col_out:
        st.subheader("Detections")
        if image is None:
            st.info("Results will appear here.")
        else:
            try:
                annotated, summary = run_detect(image, conf)
                st.image(annotated, use_container_width=True)
                st.markdown(summary)
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()

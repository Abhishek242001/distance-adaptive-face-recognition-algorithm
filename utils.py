"""
utils.py
--------
Shared helpers used by every other script in this toolkit:
face detection + embedding extraction (InsightFace, ArcFace 512-dim),
and folder-naming parsing so every script agrees on the same convention.

Folder convention:

    dataset/
        person_01/
            depth_001m/{front,left,right,top}.jpg   <- REAL captures (close range)
            depth_003m/{front,left,right,top}.jpg   <- REAL captures (close range)
            depth_020m/{front,left,right,top}.jpg   <- SYNTHETIC (generated)
            depth_080m/{front,left,right,top}.jpg   <- SYNTHETIC (generated)
            depth_150m/{front,left,right,top}.jpg   <- SYNTHETIC (generated)
        person_02/
            ...

The lowest real captured depth folder for a person is treated as their
enrollment reference (E_ref) for each view.
"""

import re
from pathlib import Path

import cv2
import numpy as np

VIEWS = ["front", "left", "right", "top"]

_face_app = None


def get_face_app():
    """Lazily load the InsightFace app (buffalo_l, ArcFace 512-dim) once.

    Provider preference order: CUDA > DirectML > CPU.

    - CUDAExecutionProvider requires the `onnxruntime-gpu` package AND
      a matching NVIDIA CUDA Toolkit + cuDNN installed system-wide
      (not just a CUDA-capable driver). Recent onnxruntime-gpu builds
      (1.27+) target CUDA 13 specifically -- if you don't have that
      exact runtime installed, provider creation silently fails and
      falls back to CPU.
    - DmlExecutionProvider (DirectML) requires the `onnxruntime-directml`
      package instead, and works with any DirectX12-capable GPU on
      Windows (NVIDIA, AMD, Intel) with NO separate toolkit install --
      much simpler to get working reliably.
    - Falls back to CPU automatically if neither is available, so this
      is always safe to call regardless of what's installed.
    """
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def get_active_provider():
    """Returns the execution provider actually running the recognition
    model right now (not just what's installed/available) -- e.g.
    'CUDAExecutionProvider', 'DmlExecutionProvider', or
    'CPUExecutionProvider'. Call this AFTER get_face_app() so the
    model is already loaded."""
    app = get_face_app()
    try:
        return app.models["recognition"].session.get_providers()[0]
    except Exception:
        return "unknown"


def extract_embedding(image_or_path):
    app = get_face_app()
    if isinstance(image_or_path, (str, Path)):
        img = cv2.imread(str(image_or_path))
    else:
        img = image_or_path
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].normed_embedding


def extract_embedding_and_width(image_or_path):
    app = get_face_app()
    if isinstance(image_or_path, (str, Path)):
        img = cv2.imread(str(image_or_path))
    else:
        img = image_or_path
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    face = faces[0]
    width_px = float(face.bbox[2] - face.bbox[0])
    return face.normed_embedding, width_px


def estimate_depth_from_facewidth(live_width_px, ref_width_px, ref_depth_m):
    """Pinhole camera model: apparent width is inversely proportional
    to distance. depth_est = ref_depth * (ref_width / live_width)."""
    if live_width_px is None or live_width_px <= 0:
        return None
    return ref_depth_m * (ref_width_px / live_width_px)


def parse_depth_folder(name: str) -> float:
    """'depth_003m' -> 3.0   'depth_000.50m' -> 0.5   'depth_150m' -> 150.0"""
    m = re.search(r"depth_(\d+(?:\.\d+)?)m", name)
    if not m:
        raise ValueError(f"Could not parse depth from folder name: {name}")
    return float(m.group(1))


def depth_folder_name(depth_m: float) -> str:
    """3 -> 'depth_003m'   20 -> 'depth_020m'   150 -> 'depth_150m'"""
    if depth_m == int(depth_m):
        return f"depth_{int(depth_m):03d}m"
    return f"depth_{depth_m:06.2f}m"


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def list_people(dataset_root: Path):
    return sorted([d for d in Path(dataset_root).iterdir() if d.is_dir()])


def list_depth_folders(person_dir: Path):
    return sorted([d for d in Path(person_dir).iterdir() if d.is_dir()], key=lambda d: parse_depth_folder(d.name))

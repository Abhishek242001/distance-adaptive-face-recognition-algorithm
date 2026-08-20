"""
adar_admin.py
---------------
ONE FILE containing the FULL production pipeline (same methods as
api_server.py + calibrate.py + enroll_capture.py + utils.py combined),
wrapped in a single browser-based admin panel instead of separate
scripts you run one after another.

Nothing about the underlying math is simplified vs the original 4-file
pipeline:

  - Multi-view enrollment: front / left / right / top, per person,
    per distance -- same as enroll_capture.py.
  - Real calibration: fits near_k, near_gamma, near_sigma0 via
    scipy.optimize.curve_fit on your captured (depth, similarity)
    points -- same math as calibrate.py.
  - SNR-gated, distance-corrected matching: alpha(d)=exp(-k*d) decay
    correction, sigma(d)=sigma0*(1+gamma*d) noise growth, SNR gate
    (>=5.0) AND cosine similarity gate (>=0.5) -- same math and same
    thresholds as api_server.py.
  - Detection logging to logs/session_log.csv + an in-app /report page
    -- same as api_server.py.
  - GPU warm-up at startup (throwaway inference calls before serving
    real requests) -- same as api_server.py.

What's different from the 4-file version is only WORKFLOW: instead of
"run enroll_capture.py, then calibrate.py, then api_server.py", this
is one script with an admin panel that does enrollment, calibration,
and live inference all from the same page, and supports enrolling
several people (2-5+) without editing any files by hand.

Usage:
    python adar_admin.py
    (asks you, right in the terminal, whether to use your webcam or
    an RTSP URL -- or pass --source to skip the prompt)

    python adar_admin.py --source "rtsp://user:pass@192.168.1.50:554/stream1"
    python adar_admin.py --source 0
    python adar_admin.py --port 8000

Then open http://<this-machine's-ip>:8000 in a browser.

WORKFLOW ON THE PAGE:
    1. Enroll: set a person's name and a distance in meters, then tap
       Capture 4 times (front/left/right/top -- the page tells you
       which pose to make each time, same as enroll_capture.py).
       Capture at MORE THAN ONE real distance per person if you can
       (e.g. 0.5m and then 1.5m) -- one distance alone cannot fit a
       meaningful gamma (noise-growth) curve; this is a documented,
       inherent limitation of single-distance calibration, not a bug.
    2. Repeat for every person.
    3. Tap "Calibrate" once everyone is enrolled (and again any time
       you add more people or more distances). This fits k/gamma/sigma0
       from everything captured so far.
    4. Start live inference -- now SNR-gated and distance-corrected,
       exactly like api_server.py's /detect and live stream.
"""

import argparse
import csv
import json
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "adar-admin"
# Enrollment status/actions (person, distance, capture) now travel over
# a WebSocket instead of the old 2s HTTP poll of /api/enroll/status.
# threading async_mode needs no extra dependency beyond flask-socketio
# itself and works fine alongside the existing background camera thread.
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

SNR_ACCEPT_THRESHOLD = 5.0
SIM_ACCEPT_THRESHOLD = 0.5
NEAR_RANGE_MAX_M = 10.0  # matches calibrate.py's near-range cutoff for indoor real-data fitting

VIEWS = ["front", "left", "right", "top"]
VIEW_PROMPTS = {
    "front": "Look straight at the camera (FRONT view)",
    "left": "Turn your head to show your LEFT profile",
    "right": "Turn your head to show your RIGHT profile",
    "top": "Tilt your head down slightly / raise the camera above eye level (TOP view)",
}

GALLERY_PATH = Path("gallery_dataset.json")
CALIBRATION_PATH = Path("calibration.json")
LOG_CSV_PATH = Path("logs") / "session_log.csv"
LOG_CSV_FIELDS = ["timestamp", "source", "employee_detected", "decision",
                   "distance_m", "similarity", "snr", "view_matched"]

# ---------------------------------------------------------------------
# Face model loading (CUDA -> DirectML -> CPU, automatic)
# ---------------------------------------------------------------------

_face_app = None


def get_face_app():
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
        print(f"[model] Loading InsightFace buffalo_l with providers: {providers}")
        _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def get_active_provider():
    fa = get_face_app()
    try:
        return fa.models["recognition"].session.get_providers()[0]
    except Exception:
        return "unknown"


def is_gpu_provider(provider: str) -> bool:
    return any(marker in provider for marker in ("CUDA", "Tensorrt", "Dml", "DML"))


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def estimate_depth_from_facewidth(live_width_px, ref_width_px, ref_depth_m):
    """Pinhole camera model: apparent face width is inversely
    proportional to distance. Same formula as ALGORITHM.md section 4 /
    utils.py."""
    if not live_width_px or live_width_px <= 0 or not ref_width_px or not ref_depth_m:
        return None
    return ref_depth_m * (ref_width_px / live_width_px)


def alpha(d, k):
    """Distance decay correction factor. Same as api_server.py."""
    return float(np.exp(-k * d))


def sigma(d, sigma0, gamma):
    """Noise growth with distance. Same as api_server.py."""
    return sigma0 * (1 + gamma * d)


def snr(d, k, sigma0, gamma, ref_norm_sq):
    """Signal-to-noise ratio at relative depth d. Same as api_server.py."""
    a = alpha(d, k)
    s = sigma(d, sigma0, gamma)
    return (a ** 2) * ref_norm_sq / (s ** 2 + 1e-8)


# ---------------------------------------------------------------------
# Gallery: person -> depth (as string key, meters) -> view -> sample.
# Persisted to gallery_dataset.json. Unlike the 4-file pipeline, images
# are never written to disk here -- only the extracted embedding +
# detected face width are kept, since that's all calibration and
# matching actually need.
# ---------------------------------------------------------------------

gallery_lock = threading.Lock()
gallery = {}  # person -> {depth_str: {view: {"embedding": [...], "width_px": float}}}


def load_gallery():
    global gallery
    if GALLERY_PATH.exists():
        with open(GALLERY_PATH) as f:
            gallery = json.load(f)
        print(f"[gallery] Loaded {len(gallery)} people from {GALLERY_PATH}")
    else:
        gallery = {}


def save_gallery():
    with gallery_lock:
        with open(GALLERY_PATH, "w") as f:
            json.dump(gallery, f)


# ---------------------------------------------------------------------
# Calibration state: near_k / near_gamma / near_sigma0 + per-person
# reference depth/width, plus the flattened matching gallery built
# from each person's reference-depth embeddings (same structure as
# calibrate.py's gallery.npz). Persisted to calibration.json.
# ---------------------------------------------------------------------

CALIB = None  # dict, set once /api/calibrate has been run (or loaded from disk)
MATCH_GALLERY = {}  # f"{person}__{view}" -> embedding (np.ndarray), reference-depth only


def load_calibration():
    global CALIB, MATCH_GALLERY
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            CALIB = json.load(f)
        _rebuild_match_gallery()
        print(f"[calibration] Loaded existing calibration from {CALIBRATION_PATH}")


def save_calibration():
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(CALIB, f, indent=2)


def _rebuild_match_gallery():
    """Rebuilds the flattened person__view -> embedding matching gallery
    from each person's reference (smallest captured) depth, per CALIB's
    recorded reference depths. Mirrors calibrate.py's gallery.npz."""
    global MATCH_GALLERY
    MATCH_GALLERY = {}
    if CALIB is None:
        return
    ref_depths = CALIB.get("reference_depths_per_person", {})
    with gallery_lock:
        for person, ref_depth in ref_depths.items():
            depths = gallery.get(person, {})
            ref_key = _find_depth_key(depths, ref_depth)
            if ref_key is None:
                continue
            for view, sample in depths[ref_key].items():
                MATCH_GALLERY[f"{person}__{view}"] = np.asarray(sample["embedding"], dtype=np.float32)


def _find_depth_key(depths_dict, target_depth, tol=1e-6):
    for key in depths_dict:
        if abs(float(key) - target_depth) < tol:
            return key
    return None


def decay_model(x, k):
    return np.exp(-k * np.asarray(x))


def noise_model(x, sigma0, gamma):
    return sigma0 * (1 + gamma * np.asarray(x))


def fit_k(rows):
    """Same as calibrate.py's fit_k -- curve_fit the decay model."""
    from scipy.optimize import curve_fit
    xs = np.array([r[0] for r in rows], dtype=float)
    sims = np.array([r[1] for r in rows], dtype=float)
    popt, pcov = curve_fit(decay_model, xs, sims, p0=[0.015], bounds=(0, 1))
    k_fit = float(popt[0])
    k_stderr = float(np.sqrt(pcov[0][0]))
    return k_fit, k_stderr


def estimate_gamma(rows, k_fit):
    """Same as calibrate.py's estimate_gamma -- fit noise growth from
    the residuals of the decay fit."""
    from scipy.optimize import curve_fit
    xs = np.array([r[0] for r in rows], dtype=float)
    sims = np.array([r[1] for r in rows], dtype=float)
    predicted = decay_model(xs, k_fit)
    residual = np.abs(predicted - sims)
    try:
        popt, _ = curve_fit(noise_model, xs, residual, p0=[0.05, 0.02], bounds=(0, [1, 1]))
        return float(popt[0]), float(popt[1])
    except RuntimeError:
        return 0.05, 0.02


def calibrate_now():
    """Full calibration pass over everything currently in `gallery`.
    Same method as calibrate.py: each person's SMALLEST captured depth
    is their reference; every other captured depth/view is compared
    back to the same-view reference via cosine similarity, and a decay
    curve (k) plus a noise-growth curve (sigma0, gamma) are fit through
    the resulting (depth-offset, similarity) points. Returns a dict
    with either the new calibration or an error explaining what's
    missing."""
    with gallery_lock:
        snapshot = json.loads(json.dumps(gallery))  # deep copy, avoid holding the lock during fitting

    if len(snapshot) < 1:
        return {"success": False, "error": "No one is enrolled yet. Capture at least one person first."}

    reference_depths = {}
    reference_widths = {}
    rows = []          # (depth_offset, similarity, absolute_depth) -- ALL data
    near_rows = []      # same, but only absolute_depth < NEAR_RANGE_MAX_M
    per_person_k = {}

    for person, depths in snapshot.items():
        if not depths:
            continue
        depth_floats = sorted(float(d) for d in depths.keys())
        ref_depth = depth_floats[0]
        ref_key = _find_depth_key(depths, ref_depth)
        ref_views = depths[ref_key]
        reference_depths[person] = ref_depth

        widths_at_ref = [v["width_px"] for v in ref_views.values() if v.get("width_px")]
        if widths_at_ref:
            reference_widths[person] = float(np.mean(widths_at_ref))

        person_rows = []
        for depth_key, views in depths.items():
            depth_val = float(depth_key)
            if depth_val == ref_depth:
                continue
            for view, sample in views.items():
                if view not in ref_views:
                    continue
                sim = cosine_sim(sample["embedding"], ref_views[view]["embedding"])
                offset = depth_val - ref_depth
                rows.append((offset, sim, depth_val))
                person_rows.append((offset, sim))
                if depth_val < NEAR_RANGE_MAX_M:
                    near_rows.append((offset, sim, depth_val))

        if len(person_rows) >= 3:
            try:
                k_i, k_i_stderr = fit_k(person_rows)
                per_person_k[person] = {"k": k_i, "stderr": k_i_stderr}
            except Exception:
                pass

    if len(rows) < 5:
        return {
            "success": False,
            "error": (
                f"Only {len(rows)} usable data point(s) so far -- not enough to fit a decay curve. "
                f"Enroll at least one more distance per person (e.g. capture at 0.5m, then again at "
                f"1.5m or 3m), or enroll more people, then calibrate again."
            ),
        }

    global_k, global_k_stderr = fit_k(rows)
    global_sigma0, global_gamma = estimate_gamma(rows, global_k)

    near_k = near_k_stderr = near_sigma0 = near_gamma = None
    if len(near_rows) >= 5:
        near_rows_2 = [(o, s) for (o, s, ad) in near_rows]
        near_k, near_k_stderr = fit_k(near_rows_2)
        near_sigma0, near_gamma = estimate_gamma(near_rows_2, near_k)

    warning = None
    if near_k is None:
        warning = (
            f"Only {len(near_rows)} near-range (<{NEAR_RANGE_MAX_M:.0f}m) datapoints found -- "
            f"falling back to the global fit for now. Capture at least 2-3 real distances per "
            f"person for a meaningful near-range fit."
        )
    elif all(len(depths) < 2 for depths in snapshot.values()):
        warning = (
            "Every person only has ONE captured distance so far. A single distance point cannot "
            "distinguish 'similarity decays with distance' from 'similarity decays with anything' -- "
            "near_gamma will likely come out as ~0.0. Capture a second distance per person (e.g. "
            "walk back to 1.5m or 3m) for a genuinely meaningful fit. This mirrors the exact same "
            "known limitation documented for the original calibrate.py."
        )

    global CALIB
    CALIB = {
        "global_k": global_k,
        "global_k_stderr": global_k_stderr,
        "sigma0": global_sigma0,
        "gamma": global_gamma,
        "near_k": near_k if near_k is not None else global_k,
        "near_k_stderr": near_k_stderr,
        "near_sigma0": near_sigma0 if near_sigma0 is not None else global_sigma0,
        "near_gamma": near_gamma if near_gamma is not None else global_gamma,
        "num_near_datapoints": len(near_rows),
        "num_datapoints": len(rows),
        "near_range_max_m": NEAR_RANGE_MAX_M,
        "reference_depths_per_person": reference_depths,
        "reference_widths_px_per_person": reference_widths,
        "per_person_k": per_person_k,
        "calibrated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    save_calibration()
    _rebuild_match_gallery()

    return {"success": True, "calibration": CALIB, "warning": warning}


# ---------------------------------------------------------------------
# SNR-gated, distance-corrected matching -- same math as api_server.py
# ---------------------------------------------------------------------

def match_face(live_emb, live_width_px):
    """Returns a dict: person, sim, depth_est, snr, view. person is
    None if no calibration is loaded or nothing matched well enough."""
    if CALIB is None:
        return {"person": None, "sim": -1.0, "depth_est": None, "snr": None, "view": None,
                "reason": "not_calibrated"}

    k = CALIB.get("near_k", CALIB["global_k"])
    sigma0 = CALIB.get("near_sigma0", CALIB["sigma0"])
    gamma = CALIB.get("near_gamma", CALIB["gamma"])
    ref_depths = CALIB["reference_depths_per_person"]
    ref_widths = CALIB["reference_widths_px_per_person"]

    best = {"person": None, "sim": -1.0, "depth_est": None, "snr": None, "view": None}

    for person, ref_depth in ref_depths.items():
        ref_width = ref_widths.get(person)
        if ref_width is None:
            continue
        depth_est = estimate_depth_from_facewidth(live_width_px, ref_width, ref_depth)
        if depth_est is None:
            continue
        relative_depth = max(depth_est - ref_depth, 0.0)

        a = alpha(relative_depth, k)
        corrected = live_emb / max(a, 1e-6)

        for key, ref_emb in MATCH_GALLERY.items():
            if not key.startswith(person + "__"):
                continue
            sim = cosine_sim(corrected, ref_emb)
            if sim > best["sim"]:
                ref_norm_sq = float(np.dot(ref_emb, ref_emb))
                best = {
                    "person": person,
                    "sim": sim,
                    "depth_est": depth_est,
                    "snr": snr(relative_depth, k, sigma0, gamma, ref_norm_sq),
                    "view": key.split("__")[1],
                }

    return best


def run_detection(frame):
    """Shared detection+matching logic used by BOTH /api/detect_once and
    the live stream loop -- same structure as api_server.py's
    run_detection()."""
    face_app = get_face_app()
    faces = sorted(face_app.get(frame), key=lambda f: f.bbox[0])

    results = []
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        live_width_px = float(x2 - x1)
        best = match_face(face.normed_embedding, live_width_px)

        if best["person"] is None:
            results.append({
                "name": "unknown",
                "similarity": round(best["sim"], 4) if best["sim"] > -1 else None,
                "distance_m": None,
                "snr": None,
                "view_matched": None,
                "bounding_box": [x1, y1, x2, y2],
                "decision": "REJECT",
                "reason": best.get("reason"),
            })
            continue

        accept = (best["snr"] is not None and best["snr"] >= SNR_ACCEPT_THRESHOLD
                  and best["sim"] >= SIM_ACCEPT_THRESHOLD)

        results.append({
            "name": best["person"] if accept else "unknown",
            "similarity": round(best["sim"], 4),
            "distance_m": round(best["depth_est"], 3) if best["depth_est"] is not None else None,
            "snr": round(best["snr"], 3) if best["snr"] is not None else None,
            "view_matched": best["view"] if accept else None,
            "bounding_box": [x1, y1, x2, y2],
            "decision": "ACCEPT" if accept else "REJECT",
        })
    return results


# ---------------------------------------------------------------------
# Detection event log -- same as api_server.py
# ---------------------------------------------------------------------

EVENT_LOG = deque(maxlen=5000)
LOG_LOCK = threading.Lock()


def ensure_log_file():
    LOG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_CSV_PATH.exists():
        with open(LOG_CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_CSV_FIELDS).writeheader()


def log_event(source, name, decision, distance, sim, snr_val, view):
    row = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": source,
        "employee_detected": name,
        "decision": decision,
        "distance_m": distance,
        "similarity": sim,
        "snr": snr_val,
        "view_matched": view,
    }
    with LOG_LOCK:
        EVENT_LOG.append(row)
        try:
            with open(LOG_CSV_PATH, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_CSV_FIELDS).writerow(row)
        except Exception:
            pass


# ---------------------------------------------------------------------
# Camera: opened ONCE, read continuously by a background thread.
# Shared by enrollment preview, single-shot capture, and live inference.
# ---------------------------------------------------------------------

CAMERA_STATE = {
    "source": None,
    "cap": None,
    "latest_frame": None,
    "lock": threading.Lock(),
    "running": False,
    "last_error": None,
}


def open_capture(source):
    if isinstance(source, str) and source.strip().lower().startswith(("rtsp://", "http://", "https://")):
        print(f"[camera] Opening network source: {source}")
        return cv2.VideoCapture(source)
    else:
        print(f"[camera] Opening local camera index: {source}")
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)


def camera_reader_loop():
    cap = open_capture(CAMERA_STATE["source"])
    if not cap.isOpened():
        CAMERA_STATE["last_error"] = f"Could not open source: {CAMERA_STATE['source']}"
        print(f"[camera] ERROR: {CAMERA_STATE['last_error']}")
        return
    CAMERA_STATE["cap"] = cap
    CAMERA_STATE["running"] = True
    print("[camera] Reader thread started")
    while CAMERA_STATE["running"]:
        ok, frame = cap.read()
        if ok:
            with CAMERA_STATE["lock"]:
                CAMERA_STATE["latest_frame"] = frame
        else:
            time.sleep(0.1)
    cap.release()
    print("[camera] Reader thread stopped")


def get_latest_frame():
    with CAMERA_STATE["lock"]:
        return None if CAMERA_STATE["latest_frame"] is None else CAMERA_STATE["latest_frame"].copy()


# ---------------------------------------------------------------------
# Enrollment state: multi-view (front/left/right/top) capture flow for
# the CURRENT person+depth, same UX as enroll_capture.py.
# ---------------------------------------------------------------------

ENROLL_STATE = {
    "person": "person_01",
    "depth": 1.0,
    "view_index": 0,
    "done": False,
    "lock": threading.Lock(),
}


def _enroll_refresh():
    ENROLL_STATE["view_index"] = 0
    ENROLL_STATE["done"] = False


def enroll_set_person(new_person):
    with ENROLL_STATE["lock"]:
        ENROLL_STATE["person"] = new_person
        _enroll_refresh()


def enroll_set_depth(new_depth):
    with ENROLL_STATE["lock"]:
        ENROLL_STATE["depth"] = new_depth
        _enroll_refresh()


def enroll_capture_view():
    """Captures the CURRENT view for the CURRENT person+depth from the
    live camera frame, extracts embedding+width, stores it, and
    advances to the next view. Returns a result dict."""
    with ENROLL_STATE["lock"]:
        if ENROLL_STATE["done"]:
            return {"success": False, "error": "All 4 views already captured for this person/distance. Change distance or person to continue."}
        person = ENROLL_STATE["person"]
        depth = ENROLL_STATE["depth"]
        view = VIEWS[ENROLL_STATE["view_index"]]

    frame = get_latest_frame()
    if frame is None:
        return {"success": False, "error": "No camera frame available yet."}

    try:
        face_app = get_face_app()
        faces = face_app.get(frame)
    except Exception as e:
        return {"success": False, "error": f"Detection failed: {e}"}

    if not faces:
        return {"success": False, "error": "No face detected in the current frame. Face the camera and try again."}

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    width_px = float(face.bbox[2] - face.bbox[0])
    sample = {"embedding": face.normed_embedding.tolist(), "width_px": width_px}

    depth_key = str(float(depth))
    with gallery_lock:
        gallery.setdefault(person, {}).setdefault(depth_key, {})[view] = sample
    save_gallery()

    with ENROLL_STATE["lock"]:
        ENROLL_STATE["view_index"] += 1
        done = ENROLL_STATE["view_index"] >= len(VIEWS)
        ENROLL_STATE["done"] = done

    return {"success": True, "person": person, "depth": depth, "view_saved": view, "done": done}


def enroll_status():
    with ENROLL_STATE["lock"]:
        idx = ENROLL_STATE["view_index"]
        return {
            "person": ENROLL_STATE["person"],
            "depth": ENROLL_STATE["depth"],
            "view_index": idx,
            "prompt": VIEW_PROMPTS[VIEWS[idx]] if idx < len(VIEWS) else "",
            "done": ENROLL_STATE["done"],
        }


def people_payload():
    with gallery_lock:
        return {"people": [{"name": name, "depths": sorted(float(d) for d in depths.keys())}
                            for name, depths in gallery.items()]}


def _push_enroll_status():
    """Broadcast the current enrollment status to every connected
    client. Only called right after a real change (set person/depth,
    capture) -- there is no periodic timer driving this anymore, which
    is what stops it from overwriting the Person/Distance boxes while
    someone is mid-typing."""
    socketio.emit("enroll_status", enroll_status())


def _push_people():
    socketio.emit("people_update", people_payload())


# ---------------------------------------------------------------------
# Live inference stream (annotated MJPEG) -- SNR-gated, same visual
# convention as api_server.py: green = ACCEPT, red = REJECT.
# ---------------------------------------------------------------------

def mjpeg_generator_raw():
    """Plain preview (used during enrollment), with the current view
    prompt overlaid -- same as enroll_capture.py's preview."""
    while True:
        frame = get_latest_frame()
        if frame is not None:
            status = enroll_status()
            if not status["done"] and status["prompt"]:
                cv2.putText(frame, status["prompt"], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


def mjpeg_generator_annotated():
    frame_count = 0
    last_results = []
    while True:
        frame = get_latest_frame()
        if frame is not None:
            frame_count += 1
            if frame_count % 3 == 0:
                try:
                    last_results = run_detection(frame)
                    for r in last_results:
                        log_event("stream", r["name"], r["decision"], r["distance_m"],
                                  r["similarity"], r["snr"], r["view_matched"])
                except Exception:
                    pass

            annotated = frame.copy()
            for r in last_results:
                x1, y1, x2, y2 = r["bounding_box"]
                accepted = r["decision"] == "ACCEPT"
                color = (0, 200, 0) if accepted else (0, 0, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                dist_txt = f'{r["distance_m"]}m' if r["distance_m"] is not None else ''
                label = f'{r["name"]} ({r["similarity"]}) {dist_txt}' if r["similarity"] is not None else r["name"]
                cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.1)  # ~10fps over the network, same cap as api_server.py


# ---------------------------------------------------------------------
# GPU warm-up -- same as api_server.py
# ---------------------------------------------------------------------

def warm_up():
    print("[startup] Warming up model (first inference is slow, especially on GPU -- doing it now)...")
    t0 = time.perf_counter()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    try:
        for _ in range(2):
            get_face_app().get(img)
    except Exception:
        pass
    t1 = time.perf_counter()
    print(f"[startup] Warm-up complete ({(t1 - t0) * 1000:.0f} ms) -- server is now fully ready for fast requests")


# ---------------------------------------------------------------------
# Admin panel HTML
# ---------------------------------------------------------------------

ADMIN_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAR Admin - Full Pipeline</title>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 0 auto; padding: 16px; background: #111; color: #eee; }
  h2 { font-size: 18px; margin-top: 24px; }
  a { color: #4fa3ff; }
  .nav { font-size: 13px; margin-bottom: 12px; }
  .card { background: #1e1e1e; border-radius: 8px; padding: 14px; margin: 10px 0; }
  input[type=text], input[type=number] { font-size: 16px; padding: 10px; width: 100%; box-sizing: border-box;
    border-radius: 6px; border: 1px solid #444; background: #111; color: #eee; margin: 6px 0; }
  button { font-size: 15px; padding: 10px 18px; border-radius: 6px; border: none; margin: 4px 4px 4px 0; cursor: pointer; }
  .btn-primary { background: #2ecc71; color: #fff; }
  .btn-secondary { background: #3498db; color: #fff; }
  .btn-danger { background: #e53935; color: #fff; }
  .btn-warn { background: #f39c12; color: #111; }
  img#enrollPreview, img#inferFeed { width: 100%; border-radius: 8px; margin: 8px 0; background: #000; }
  #inferFeed { display: none; }
  .person-card { display: flex; justify-content: space-between; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid #333; font-size: 14px; }
  .status-line { font-size: 14px; color: #ccc; min-height: 20px; margin: 6px 0; }
  .warn { color: #f39c12; }
  .ok { color: #2ecc71; }
  pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; background: #000; padding: 10px; border-radius: 6px; }
  .landing-card { background: #1e1e1e; border-radius: 10px; padding: 22px; margin: 14px 0; text-align: center; }
  .landing-btn { display: block; width: 100%; font-size: 17px; padding: 16px; margin: 10px 0; }
  .back-link { display: inline-block; margin: 10px 0; font-size: 13px; cursor: pointer; color: #4fa3ff; }
  #enrollView, #inferView, #landingView { display: none; }
  #connIndicator { position: fixed; top: 8px; right: 10px; font-size: 12px; padding: 4px 10px;
                    border-radius: 10px; background: #333; z-index: 10; }
  #connIndicator.connected { background: #1e5c33; color: #6fe38f; }
  #connIndicator.disconnected { background: #5c1e1e; color: #ff8a8a; }
</style>
</head>
<body>
<div id="connIndicator">Connecting...</div>
<div class="nav"><a href="/report">Detection report</a></div>
<h1 style="font-size:20px;">ADAR Admin -- Full Pipeline</h1>
<div id="sourceInfo" class="status-line"></div>

<div id="landingView">
  <div class="landing-card">
    <p style="font-size:14px;color:#999;margin-top:0;">What do you want to do?</p>
    <button class="btn-primary landing-btn" onclick="showEnrollMode()">Enroll a face</button>
    <button class="btn-secondary landing-btn" onclick="showInferMode()">Start live inference</button>
  </div>
</div>

<div id="enrollView">
  <span class="back-link" onclick="showLanding()">&larr; Back</span>

  <h2>1. Enroll a person</h2>
  <div class="card">
    <input type="text" id="personInput" placeholder="Person name (e.g. Abhishek)">
    <input type="number" step="0.1" min="0" id="distanceInput" placeholder="Distance from camera in meters (e.g. 0.5)">
    <div>
      <button class="btn-secondary" onclick="applyPerson()">Set person</button>
      <button class="btn-secondary" onclick="applyDepth()">Set distance</button>
    </div>
    <img id="enrollPreview" src="/video_feed">
    <div id="enrollStatus" class="status-line">Loading...</div>
    <button class="btn-primary" id="capBtn" onclick="doCapture()">CAPTURE</button>
    <div id="confirm" class="status-line"></div>
    <p style="font-size:12px;color:#888;">All 4 views (front, left, right, top) are required at every distance before that person-distance pair counts as done. Capture at more than one real distance per person if you can (e.g. 0.5m, then walk back and capture again at 1.5m or 3m) -- one distance alone can't fit a meaningful decay curve.</p>
  </div>

  <h2>Enrolled people</h2>
  <div class="card"><div id="peopleList">Loading...</div></div>

  <h2>2. Calibrate</h2>
  <div class="card">
    <p style="font-size:13px;color:#999;">Fits the real decay curve (k, gamma, sigma0) from everyone captured so far. Run this once everyone is enrolled, and again any time you add more people or distances.</p>
    <button class="btn-warn" onclick="runCalibrate()">Calibrate now</button>
    <div id="calibStatus" class="status-line"></div>
  </div>
</div>

<div id="inferView">
  <span class="back-link" onclick="showLanding()">&larr; Back</span>

  <h2>Live inference (SNR-gated)</h2>
  <div class="card">
    <p style="font-size:13px;color:#999;">Uses the calibration and gallery already on record. Green box = ACCEPT (SNR &ge; 5.0 and similarity &ge; 0.5), red = REJECT.</p>
    <button class="btn-primary" onclick="startInfer()">Start</button>
    <button class="btn-danger" onclick="stopInfer()">Stop</button>
    <div id="inferStatus" class="status-line"></div>
    <img id="inferFeed">
    <button class="btn-secondary" onclick="detectOnce()">Run single detection</button>
    <pre id="detectResult"></pre>
  </div>
</div>

<script>
let currentMode = null;

function showLanding() {
  currentMode = null;
  document.getElementById('landingView').style.display = 'block';
  document.getElementById('enrollView').style.display = 'none';
  document.getElementById('inferView').style.display = 'none';
  stopInfer();
}

function showEnrollMode() {
  currentMode = 'enroll';
  document.getElementById('landingView').style.display = 'none';
  document.getElementById('enrollView').style.display = 'block';
  document.getElementById('inferView').style.display = 'none';
}

function showInferMode() {
  currentMode = 'infer';
  document.getElementById('landingView').style.display = 'none';
  document.getElementById('enrollView').style.display = 'none';
  document.getElementById('inferView').style.display = 'block';
  startInfer();
}

const socket = io();
const personEl = document.getElementById('personInput');
const depthEl = document.getElementById('distanceInput');
const connIndicator = document.getElementById('connIndicator');

socket.on('connect', () => {
  connIndicator.innerText = 'Connected';
  connIndicator.className = 'connected';
});

socket.on('disconnect', () => {
  connIndicator.innerText = 'Reconnecting...';
  connIndicator.className = 'disconnected';
});

// This is the ONLY place enrollment status updates the UI from --
// pushed by the server right after a real change (capture /
// enroll_set_person / enroll_set_depth), never on a timer. That's what
// stops it from clobbering the Person/Distance boxes while you're
// typing: there is nothing to race against, unlike the old 2s poll.
socket.on('enroll_status', (j) => {
  // Still guard against overwriting a field you're actively typing in,
  // in case an update arrives from elsewhere (e.g. a second device).
  if (document.activeElement !== personEl) personEl.value = j.person;
  if (document.activeElement !== depthEl) depthEl.value = j.depth;
  const statusEl = document.getElementById('enrollStatus');
  if (j.done) {
    statusEl.innerText = `All 4 views captured for ${j.person} at ${j.depth}m. Change person and/or distance to keep going.`;
    document.getElementById('capBtn').disabled = true;
  } else {
    statusEl.innerText = `${j.person} @ ${j.depth}m -- View ${j.view_index + 1} of 4: ${j.prompt}`;
    document.getElementById('capBtn').disabled = false;
  }
});

socket.on('people_update', (j) => renderPeople(j.people));

async function loadMeta() {
  const r = await fetch('/api/source');
  const j = await r.json();
  document.getElementById('sourceInfo').innerText =
    `Source: ${j.source} | Provider: ${j.active_provider} | GPU: ${j.gpu_used}` +
    (j.camera_error ? ` | CAMERA ERROR: ${j.camera_error}` : '');
}

function applyPerson() {
  const name = personEl.value.trim();
  if (!name) return;
  socket.emit('enroll_set_person', {person: name});
}

function applyDepth() {
  const depth = parseFloat(depthEl.value);
  if (isNaN(depth)) return;
  socket.emit('enroll_set_depth', {depth});
}

function doCapture() {
  document.getElementById('capBtn').disabled = true;
  socket.emit('enroll_capture', {}, (j) => {
    document.getElementById('confirm').innerText = j.success ? `Saved: ${j.view_saved}` : ('Error: ' + j.error);
  });
}

function renderPeople(people) {
  const el = document.getElementById('peopleList');
  if (!people.length) { el.innerHTML = '<span style="color:#888;">No one enrolled yet.</span>'; return; }
  el.innerHTML = people.map(p => `
    <div class="person-card">
      <span><b>${p.name}</b><br><span style="color:#888;font-size:12px;">${p.depths.join('m, ')}m captured</span></span>
      <button class="btn-danger" onclick="deletePerson('${p.name}')">Delete</button>
    </div>
  `).join('');
}

async function refreshPeople() {
  const r = await fetch('/api/people');
  const j = await r.json();
  renderPeople(j.people);
}

async function deletePerson(name) {
  if (!confirm(`Delete all samples for ${name}?`)) return;
  await fetch('/api/people/' + encodeURIComponent(name), {method: 'DELETE'});
  refreshPeople();
}

async function runCalibrate() {
  document.getElementById('calibStatus').innerText = 'Calibrating...';
  const r = await fetch('/api/calibrate', {method: 'POST'});
  const j = await r.json();
  if (!j.success) {
    document.getElementById('calibStatus').innerHTML = '<span class="warn">' + j.error + '</span>';
    return;
  }
  const c = j.calibration;
  let html = `<span class="ok">Calibrated.</span> near_k=${c.near_k.toFixed(5)}, near_gamma=${c.near_gamma.toFixed(4)}, near_sigma0=${c.near_sigma0.toFixed(4)} (${c.num_near_datapoints} near-range datapoints)`;
  if (j.warning) html += `<br><span class="warn">${j.warning}</span>`;
  document.getElementById('calibStatus').innerHTML = html;
}

function startInfer() {
  const feed = document.getElementById('inferFeed');
  feed.src = '/api/stream?' + Date.now();
  feed.style.display = 'block';
  document.getElementById('inferStatus').innerText = 'Live inference running.';
}

function stopInfer() {
  const feed = document.getElementById('inferFeed');
  feed.src = '';
  feed.style.display = 'none';
  document.getElementById('inferStatus').innerText = 'Stopped.';
}

async function detectOnce() {
  document.getElementById('detectResult').innerText = 'Running...';
  const r = await fetch('/api/detect_once');
  const j = await r.json();
  document.getElementById('detectResult').innerText = JSON.stringify(j, null, 2);
}

loadMeta();
showLanding();
</script>
</body>
</html>"""


REPORT_PAGE_STYLE = """
  body { font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 16px; background: #111; color: #eee; }
  h2 { font-size: 18px; }
  a { color: #4fa3ff; }
  .card { background: #1e1e1e; border-radius: 8px; padding: 12px; margin: 8px 0; }
  .accept { border-left: 4px solid #4caf50; }
  .reject { border-left: 4px solid #e53935; }
  .row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 14px; }
  .row span:first-child { color: #999; }
  .big { font-size: 18px; font-weight: bold; }
"""


@app.route("/")
def admin_page():
    return Response(ADMIN_PAGE_HTML, mimetype="text/html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/source")
def api_source():
    provider = get_active_provider()
    return jsonify({
        "source": str(CAMERA_STATE["source"]),
        "active_provider": provider,
        "gpu_used": is_gpu_provider(provider),
        "camera_error": CAMERA_STATE["last_error"],
    })


@app.route("/api/people")
def api_people():
    with gallery_lock:
        people = [{"name": name, "depths": sorted(float(d) for d in depths.keys())}
                  for name, depths in gallery.items()]
    return jsonify({"people": people})


@app.route("/api/people/<name>", methods=["DELETE"])
def api_delete_person(name):
    with gallery_lock:
        if name in gallery:
            del gallery[name]
    save_gallery()
    _push_people()
    return jsonify({"success": True})


@socketio.on("connect")
def on_connect():
    """Push current enrollment status + people list immediately to a
    newly-connected/reconnected client, so the page is never stuck on
    'Loading...'."""
    socketio.emit("enroll_status", enroll_status(), to=request.sid)
    socketio.emit("people_update", people_payload(), to=request.sid)


@socketio.on("enroll_set_person")
def on_enroll_set_person(payload):
    name = str((payload or {}).get("person", "")).strip()
    if not name:
        return {"success": False, "error": "Empty name."}
    enroll_set_person(name)
    _push_enroll_status()
    return {"success": True}


@socketio.on("enroll_set_depth")
def on_enroll_set_depth(payload):
    try:
        depth = float((payload or {}).get("depth"))
    except (TypeError, ValueError):
        return {"success": False, "error": "Invalid distance."}
    enroll_set_depth(depth)
    _push_enroll_status()
    return {"success": True}


@socketio.on("enroll_capture")
def on_enroll_capture():
    result = enroll_capture_view()
    _push_enroll_status()
    if result.get("success"):
        _push_people()
    return result


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    result = calibrate_now()
    return jsonify(result), (200 if result["success"] else 400)


@app.route("/api/calibration")
def api_calibration():
    return jsonify({"calibrated": CALIB is not None, "calibration": CALIB})


@app.route("/api/detect_once")
def api_detect_once():
    if CALIB is None:
        return jsonify({"success": False, "error": "Not calibrated yet. Enroll people and tap Calibrate first."}), 400
    frame = get_latest_frame()
    if frame is None:
        return jsonify({"success": False, "error": "No camera frame available yet."}), 503
    try:
        results = run_detection(frame)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    for r in results:
        log_event("api", r["name"], r["decision"], r["distance_m"], r["similarity"], r["snr"], r["view_matched"])
    return jsonify({
        "success": True,
        "num_faces_detected": len(results),
        "faces": results,
        "compute_device": "GPU" if is_gpu_provider(get_active_provider()) else "CPU",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    })


@app.route("/api/stream")
def api_stream():
    if CALIB is None:
        return jsonify({"success": False, "error": "Not calibrated yet. Enroll people and tap Calibrate first."}), 400
    return Response(mjpeg_generator_annotated(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/report")
def report():
    with LOG_LOCK:
        events = list(EVENT_LOG)
    total = len(events)
    accepts = [e for e in events if e["decision"] == "ACCEPT"]
    rejects = [e for e in events if e["decision"] == "REJECT"]
    per_person = {}
    for e in accepts:
        per_person.setdefault(e["employee_detected"], []).append(e)

    rows_html = ""
    for e in list(reversed(events))[:30]:
        cls = "accept" if e["decision"] == "ACCEPT" else "reject"
        rows_html += f"""<div class="card {cls}">
          <div class="row"><span>{e['timestamp'][11:19]}</span><span>{e['source']}</span></div>
          <div class="big">{e['employee_detected']}</div>
          <div class="row"><span>Decision</span><span>{e['decision']}</span></div>
          <div class="row"><span>Distance</span><span>{e['distance_m']} m</span></div>
          <div class="row"><span>Similarity</span><span>{e['similarity']}</span></div>
          <div class="row"><span>SNR</span><span>{e['snr']}</span></div>
        </div>"""

    per_person_html = "".join(
        f'<div class="row"><span>{name}</span><span>{len(evs)} accepted detections</span></div>'
        for name, evs in per_person.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAR Admin - Report</title>
<style>{REPORT_PAGE_STYLE}</style>
</head>
<body>
<p><a href="/">&larr; Back to admin panel</a></p>
<h2>Detection Report</h2>
<div class="card">
  <div class="row"><span>Total events logged</span><span>{total}</span></div>
  <div class="row"><span>Accepted</span><span>{len(accepts)}</span></div>
  <div class="row"><span>Rejected</span><span>{len(rejects)}</span></div>
</div>
<div class="card">
  <div class="big">Per person (accepted)</div>
  {per_person_html or '<div class="row"><span>No accepted detections yet</span></div>'}
</div>
<p><a href="/report/csv">Download full CSV log</a></p>
<h3 style="font-size:15px;">Last {min(30, total)} events</h3>
{rows_html or '<div class="card">No events logged yet.</div>'}
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/report/csv")
def report_csv():
    if not LOG_CSV_PATH.exists():
        return jsonify({"success": False, "error": "No log file yet."}), 404
    return send_file(LOG_CSV_PATH, as_attachment=True, download_name="adar_session_log.csv")


@app.route("/health")
def health():
    return jsonify({
        "success": True,
        "camera_running": CAMERA_STATE["running"],
        "camera_error": CAMERA_STATE["last_error"],
        "people_enrolled": len(gallery),
        "calibrated": CALIB is not None,
        "active_provider": get_active_provider(),
        "events_logged": len(EVENT_LOG),
    })


# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------

def prompt_for_source():
    print("\nWhich camera source do you want to use?")
    print("  1) Local webcam")
    print("  2) RTSP URL (real CCTV / IP camera)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "2":
        url = input("Paste the full RTSP URL (e.g. rtsp://user:pass@192.168.1.50:554/stream1): ").strip()
        return url
    else:
        idx = input("Webcam index (press Enter for default 0): ").strip()
        return idx if idx else "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                         help="Camera index (e.g. 0) or RTSP/HTTP URL. If omitted, you'll be asked at startup.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no_warmup", action="store_true", help="Skip startup warm-up (not recommended on GPU)")
    args = parser.parse_args()

    source = args.source if args.source is not None else prompt_for_source()
    CAMERA_STATE["source"] = source

    ensure_log_file()
    load_gallery()

    print("\nLoading face recognition model (this can take a few seconds)...")
    get_face_app()
    print(f"Active execution provider: {get_active_provider()}")

    load_calibration()

    if not args.no_warmup:
        warm_up()

    t = threading.Thread(target=camera_reader_loop, daemon=True)
    t.start()
    time.sleep(1.0)
    if CAMERA_STATE["last_error"]:
        print(f"\n[WARNING] {CAMERA_STATE['last_error']}")
        print("The admin panel will still start, but the camera feed won't work until this is fixed.\n")

    print(f"\nAdmin panel ready. Open this in a browser:")
    print(f"   http://<this-machine-ip>:{args.port}")
    print(f"   (or http://localhost:{args.port} on this machine)\n")
    if CALIB is None:
        print("[startup] No calibration found yet -- enroll people and tap 'Calibrate now' before using live inference.\n")
    print("[startup] Enrollment status now travels over a WebSocket, pushed only on real changes --")
    print("          no more background polling clobbering the boxes while you type.\n")

    socketio.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""
api_server.py
--------------
V3 -- ArcFace + SNR pipeline, served as a production HTTP API, PLUS a
live webcam stream and detection report viewable from your phone.

Routes:
    POST /detect        raw image bytes in, JSON detection result out
    GET  /health         liveness/readiness check
    GET  /test            mobile page: take/pick a photo, see /detect result
    POST /stream/start    starts the laptop's webcam + live detection loop
    POST /stream/stop     stops it
    GET  /stream           MJPEG video feed (open in an <img> tag or browser)
    GET  /stream_page      mobile page: Start/Stop buttons + live feed
    GET  /report            mobile page: summary of every detection logged
    GET  /report/csv        downloads the full detection log as CSV

Every detection -- from a /detect POST OR from the live stream -- is
logged to logs/session_log.csv (created automatically) and kept in
memory for the /report page, so you have a persistent record of who
was seen, when, at what distance, and whether they were accepted.

Run it:
    python api_server.py --calibration_dir calibration_output --port 8000
"""

import argparse
import csv
import glob
import json
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_file

from utils import get_face_app, get_active_provider, cosine_sim, estimate_depth_from_facewidth

SNR_ACCEPT_THRESHOLD = 5.0
DEFAULT_SIM_ACCEPT_THRESHOLD = 0.5

NAME_MAP = {
    "person_01": "Abhishek",
    "person_02": "Manoj",
}

app = Flask(__name__)

STATE = {
    "face_app": None,
    "calib": None,
    "gallery": None,
    "gpu_used": False,
    "calibration_dir": None,
}

# ---------------------------------------------------------------------
# Live stream state (laptop webcam -> detection loop -> MJPEG output)
# ---------------------------------------------------------------------
STREAM_STATE = {
    "running": False,
    "thread": None,
    "latest_jpeg": None,
    "lock": threading.Lock(),
    "camera_index": 0,
    "detect_every_n": 5,
    "last_error": None,
}

# ---------------------------------------------------------------------
# Detection event log -- every /detect call AND every stream detection
# gets appended here, both in-memory (for the fast /report page) and
# to a CSV file on disk (so the record survives a server restart).
# ---------------------------------------------------------------------
EVENT_LOG = deque(maxlen=5000)
LOG_LOCK = threading.Lock()
LOG_CSV_PATH = Path("logs") / "session_log.csv"
LOG_CSV_FIELDS = ["timestamp", "source", "employee_detected", "decision",
                   "distance_m", "similarity", "snr", "view_matched"]


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
            pass  # logging failures should never break real-time detection


TEST_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAR v3 - Live Test</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 16px; background: #111; color: #eee; }
  h2 { font-size: 18px; }
  a { color: #4fa3ff; }
  input[type=file] { width: 100%; padding: 14px; font-size: 16px; margin: 12px 0; }
  #preview { width: 100%; border-radius: 8px; margin: 8px 0; display: none; }
  #status { color: #999; font-size: 14px; }
  .card { background: #1e1e1e; border-radius: 8px; padding: 12px; margin: 8px 0; }
  .accept { border-left: 4px solid #4caf50; }
  .reject { border-left: 4px solid #e53935; }
  .row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 14px; }
  .row span:first-child { color: #999; }
  .big { font-size: 20px; font-weight: bold; }
  .nav { margin-bottom: 12px; font-size: 14px; }
</style>
</head>
<body>
<div class="nav"><a href="/test">Photo test</a> &nbsp;|&nbsp; <a href="/stream_page">Live stream</a> &nbsp;|&nbsp; <a href="/report">Report</a></div>
<h2>ADAR v3 - Live Test</h2>
<input type="file" id="fileInput" accept="image/*" capture="environment">
<img id="preview">
<div id="status"></div>
<div id="results"></div>

<script>
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  statusEl.textContent = 'Sending to server...';
  resultsEl.innerHTML = '';

  try {
    const buf = await file.arrayBuffer();
    const t0 = performance.now();
    const resp = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: buf
    });
    const data = await resp.json();
    const roundtrip = (performance.now() - t0).toFixed(0);

    if (!data.success) {
      statusEl.textContent = 'Error: ' + data.error;
      return;
    }

    statusEl.textContent = `Round-trip: ${roundtrip} ms | Server: ${data.timing_ms.total} ms | ${data.compute_device}`;

    if (data.num_faces_detected === 0) {
      resultsEl.innerHTML = '<div class="card">No face detected.</div>';
      return;
    }

    let html = '';
    for (let i = 0; i < data.num_faces_detected; i++) {
      const cls = data.decision[i] === 'ACCEPT' ? 'accept' : 'reject';
      html += `<div class="card ${cls}">
        <div class="big">${data.employee_detected[i]}</div>
        <div class="row"><span>Decision</span><span>${data.decision[i]}</span></div>
        <div class="row"><span>Distance</span><span>${data.distance_m[i] ?? 'N/A'} m</span></div>
        <div class="row"><span>Similarity</span><span>${data.similarity[i] ?? 'N/A'}</span></div>
        <div class="row"><span>SNR</span><span>${data.snr[i] ?? 'N/A'}</span></div>
        <div class="row"><span>View matched</span><span>${data.view_matched[i] ?? 'N/A'}</span></div>
      </div>`;
    }
    resultsEl.innerHTML = html;
  } catch (err) {
    statusEl.textContent = 'Request failed: ' + err;
  }
});
</script>
</body>
</html>"""


STREAM_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAR v3 - Live Stream</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 16px; background: #111; color: #eee; }
  h2 { font-size: 18px; }
  a { color: #4fa3ff; }
  .nav { margin-bottom: 12px; font-size: 14px; }
  button { padding: 12px 20px; font-size: 16px; margin: 4px; border-radius: 6px; border: none; }
  #startBtn { background: #4caf50; color: white; }
  #stopBtn { background: #e53935; color: white; }
  #feed { width: 100%; border-radius: 8px; margin-top: 12px; display: none; background: #000; }
  #status { color: #999; font-size: 14px; margin-top: 8px; }
</style>
</head>
<body>
<div class="nav"><a href="/test">Photo test</a> &nbsp;|&nbsp; <a href="/stream_page">Live stream</a> &nbsp;|&nbsp; <a href="/report">Report</a></div>
<h2>ADAR v3 - Live Stream</h2>
<p style="font-size:13px;color:#999;">Streams the LAPTOP's webcam with live detection boxes drawn on it. Green box = ACCEPT, red = REJECT.</p>
<button id="startBtn">Start</button>
<button id="stopBtn">Stop</button>
<div id="status"></div>
<img id="feed">

<script>
const feed = document.getElementById('feed');
const statusEl = document.getElementById('status');

document.getElementById('startBtn').addEventListener('click', async () => {
  statusEl.textContent = 'Starting camera...';
  const resp = await fetch('/stream/start', { method: 'POST' });
  const data = await resp.json();
  statusEl.textContent = data.message || '';
  feed.src = '/stream?' + Date.now();
  feed.style.display = 'block';
});

document.getElementById('stopBtn').addEventListener('click', async () => {
  const resp = await fetch('/stream/stop', { method: 'POST' });
  const data = await resp.json();
  statusEl.textContent = data.message || '';
  feed.style.display = 'none';
  feed.src = '';
});
</script>
</body>
</html>"""


def alpha(d, k):
    return float(np.exp(-k * d))


def sigma(d, sigma0, gamma):
    return sigma0 * (1 + gamma * d)


def snr(d, k, sigma0, gamma, ref_norm_sq):
    a = alpha(d, k)
    s = sigma(d, sigma0, gamma)
    return (a ** 2) * ref_norm_sq / (s ** 2 + 1e-8)


def match_face(live_emb, live_width_px, calib, gallery):
    k = calib.get("near_k", calib["global_k"])
    sigma0 = calib.get("near_sigma0", calib["sigma0"])
    gamma = calib.get("near_gamma", calib["gamma"])
    ref_depths = calib["reference_depths_per_person"]
    ref_widths = calib["reference_widths_px_per_person"]

    best = {"person": None, "sim": -1.0, "depth_est": None, "snr": None, "view": None}

    for person_id, ref_depth in ref_depths.items():
        ref_width = ref_widths.get(person_id)
        if ref_width is None:
            continue
        depth_est = estimate_depth_from_facewidth(live_width_px, ref_width, ref_depth)
        if depth_est is None:
            continue
        relative_depth = max(depth_est - ref_depth, 0.0)

        a = alpha(relative_depth, k)
        corrected = live_emb / max(a, 1e-6)

        for key, ref_emb in gallery.items():
            if not key.startswith(person_id + "__"):
                continue
            sim = cosine_sim(corrected, ref_emb)
            if sim > best["sim"]:
                ref_norm_sq = float(np.dot(ref_emb, ref_emb))
                best = {
                    "person": person_id,
                    "sim": sim,
                    "depth_est": depth_est,
                    "snr": snr(relative_depth, k, sigma0, gamma, ref_norm_sq),
                    "view": key.split("__")[1],
                }

    return best


def display_name(person_id):
    return NAME_MAP.get(person_id, person_id)


def run_detection(frame):
    """Shared detection+matching logic used by BOTH /detect and the
    live stream loop, so there's exactly one implementation of the
    matching pipeline instead of two copies that could drift apart."""
    faces = STATE["face_app"].get(frame)
    faces = sorted(faces, key=lambda f: f.bbox[0])

    employee_detected, bounding_boxes, distance_m = [], [], []
    similarity, snr_values, decision, view_matched = [], [], [], []

    calib = STATE["calib"]
    gallery = STATE["gallery"]

    for face in faces:
        live_emb = face.normed_embedding
        live_width_px = float(face.bbox[2] - face.bbox[0])
        x1, y1, x2, y2 = [int(v) for v in face.bbox]

        best = match_face(live_emb, live_width_px, calib, gallery)

        if best["person"] is None:
            employee_detected.append("unknown")
            bounding_boxes.append([x1, y1, x2, y2])
            distance_m.append(None)
            similarity.append(None)
            snr_values.append(None)
            decision.append("REJECT")
            view_matched.append(None)
            continue

        accept = (best["snr"] is not None and best["snr"] >= SNR_ACCEPT_THRESHOLD
                  and best["sim"] >= DEFAULT_SIM_ACCEPT_THRESHOLD)

        employee_detected.append(display_name(best["person"]) if accept else "unknown")
        bounding_boxes.append([x1, y1, x2, y2])
        distance_m.append(round(best["depth_est"], 3) if best["depth_est"] is not None else None)
        similarity.append(round(best["sim"], 4))
        snr_values.append(round(best["snr"], 3) if best["snr"] is not None else None)
        decision.append("ACCEPT" if accept else "REJECT")
        view_matched.append(best["view"])

    return {
        "num_faces_detected": len(faces),
        "employee_detected": employee_detected,
        "bounding_boxes": bounding_boxes,
        "distance_m": distance_m,
        "similarity": similarity,
        "snr": snr_values,
        "decision": decision,
        "view_matched": view_matched,
    }


def load_state(calibration_dir: Path):
    calib_path = calibration_dir / "calibration_results.json"
    gallery_path = calibration_dir / "gallery.npz"
    if not calib_path.exists() or not gallery_path.exists():
        raise RuntimeError(
            f"Missing calibration files in {calibration_dir}. "
            f"Run calibrate.py first (see README.md)."
        )

    with open(calib_path) as f:
        calib = json.load(f)
    gallery_npz = np.load(gallery_path)
    gallery = {k: gallery_npz[k] for k in gallery_npz.files}

    face_app = get_face_app()

    active_provider = get_active_provider()
    gpu_used = any(marker in active_provider for marker in ("CUDA", "Tensorrt", "Dml", "DML"))

    STATE["face_app"] = face_app
    STATE["calib"] = calib
    STATE["gallery"] = gallery
    STATE["gpu_used"] = gpu_used
    STATE["active_provider"] = active_provider
    STATE["calibration_dir"] = str(calibration_dir)

    using_near = "near_k" in calib
    print(f"[startup] Loaded calibration from {calibration_dir}")
    print(f"[startup] Using {'NEAR-RANGE' if using_near else 'GLOBAL'} fit: "
          f"k={calib.get('near_k', calib['global_k']):.5f}, "
          f"gamma={calib.get('near_gamma', calib['gamma']):.4f}, "
          f"sigma0={calib.get('near_sigma0', calib['sigma0']):.4f}")
    print(f"[startup] Gallery: {len(gallery)} reference embeddings")
    print(f"[startup] Active execution provider: {active_provider}")
    print(f"[startup] GPU actually in use: {gpu_used}")


def warm_up(face_app, dataset_root: str = "dataset"):
    print("[startup] Warming up model (first inference is slow, especially on GPU -- doing it now, not on your first real request)...")
    t0 = time.perf_counter()

    sample_path = None
    try:
        matches = glob.glob(str(Path(dataset_root) / "**" / "*.jpg"), recursive=True)
        if matches:
            sample_path = matches[0]
    except Exception:
        pass

    img = cv2.imread(sample_path) if sample_path else None
    if img is None:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        for _ in range(2):
            STATE["face_app"].get(img)
    except Exception:
        pass

    t1 = time.perf_counter()
    print(f"[startup] Warm-up complete ({(t1 - t0) * 1000:.0f} ms) -- server is now fully ready for fast requests")


def error_response(message: str, status_code: int = 400):
    body = {
        "success": False,
        "error": message,
        "num_faces_detected": 0,
        "employee_detected": [],
        "bounding_boxes": [],
        "distance_m": [],
        "similarity": [],
        "snr": [],
        "decision": [],
        "view_matched": [],
        "timing_ms": None,
        "compute_device": "GPU" if STATE["gpu_used"] else "CPU",
        "gpu_used": STATE["gpu_used"],
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    return jsonify(body), status_code


# ---------------------------------------------------------------------
# Live webcam stream: background thread reads the LAPTOP's camera,
# runs detection every N frames (full detection every frame would be
# too slow to sustain smooth video), draws boxes on EVERY frame using
# the most recent detection result, and publishes the latest JPEG for
# the MJPEG route to serve.
# ---------------------------------------------------------------------

def camera_worker():
    cap = cv2.VideoCapture(STREAM_STATE["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        STREAM_STATE["last_error"] = f"Could not open camera index {STREAM_STATE['camera_index']}"
        STREAM_STATE["running"] = False
        print(f"[stream] ERROR: {STREAM_STATE['last_error']}")
        return

    print("[stream] Camera worker started")
    frame_count = 0
    last_result = None

    while STREAM_STATE["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % STREAM_STATE["detect_every_n"] == 0 or last_result is None:
            try:
                last_result = run_detection(frame)
                for i in range(last_result["num_faces_detected"]):
                    log_event("stream", last_result["employee_detected"][i], last_result["decision"][i],
                              last_result["distance_m"][i], last_result["similarity"][i],
                              last_result["snr"][i], last_result["view_matched"][i])
            except Exception as e:
                STREAM_STATE["last_error"] = str(e)

        annotated = frame.copy()
        if last_result:
            for i in range(last_result["num_faces_detected"]):
                x1, y1, x2, y2 = last_result["bounding_boxes"][i]
                accepted = last_result["decision"][i] == "ACCEPT"
                color = (0, 200, 0) if accepted else (0, 0, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                dist_txt = f'{last_result["distance_m"][i]}m' if last_result["distance_m"][i] is not None else ''
                label = f'{last_result["employee_detected"][i]} {dist_txt}'
                cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ok2, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok2:
            with STREAM_STATE["lock"]:
                STREAM_STATE["latest_jpeg"] = jpeg.tobytes()

    cap.release()
    print("[stream] Camera worker stopped")


def mjpeg_generator():
    boundary = b"--frame"
    while STREAM_STATE["running"]:
        with STREAM_STATE["lock"]:
            jpeg = STREAM_STATE["latest_jpeg"]
        if jpeg is not None:
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.1)  # cap the feed to ~10fps over the network


@app.route("/stream/start", methods=["POST"])
def stream_start():
    if STREAM_STATE["running"]:
        return jsonify({"success": True, "message": "Stream already running."}), 200
    STREAM_STATE["running"] = True
    STREAM_STATE["last_error"] = None
    t = threading.Thread(target=camera_worker, daemon=True)
    STREAM_STATE["thread"] = t
    t.start()
    time.sleep(0.5)  # give the camera a moment to open before responding
    if STREAM_STATE["last_error"]:
        STREAM_STATE["running"] = False
        return jsonify({"success": False, "message": STREAM_STATE["last_error"]}), 500
    return jsonify({"success": True, "message": "Stream started."}), 200


@app.route("/stream/stop", methods=["POST"])
def stream_stop():
    STREAM_STATE["running"] = False
    return jsonify({"success": True, "message": "Stream stopped."}), 200


@app.route("/stream")
def stream():
    if not STREAM_STATE["running"]:
        return error_response("Stream not running. POST /stream/start first (or use /stream_page).", 400)
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream_page")
def stream_page():
    return Response(STREAM_PAGE_HTML, mimetype="text/html")


@app.route("/test", methods=["GET"])
def test_page():
    return Response(TEST_PAGE_HTML, mimetype="text/html")


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

    per_person_html = ""
    for name, evs in per_person.items():
        per_person_html += f'<div class="row"><span>{name}</span><span>{len(evs)} accepted detections</span></div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAR v3 - Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 16px; background: #111; color: #eee; }}
  h2 {{ font-size: 18px; }}
  a {{ color: #4fa3ff; }}
  .nav {{ margin-bottom: 12px; font-size: 14px; }}
  .card {{ background: #1e1e1e; border-radius: 8px; padding: 12px; margin: 8px 0; }}
  .accept {{ border-left: 4px solid #4caf50; }}
  .reject {{ border-left: 4px solid #e53935; }}
  .row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 14px; }}
  .row span:first-child {{ color: #999; }}
  .big {{ font-size: 18px; font-weight: bold; }}
  .summary {{ font-size: 15px; }}
</style>
</head>
<body>
<div class="nav"><a href="/test">Photo test</a> &nbsp;|&nbsp; <a href="/stream_page">Live stream</a> &nbsp;|&nbsp; <a href="/report">Report</a></div>
<h2>Detection Report</h2>
<div class="card summary">
  <div class="row"><span>Total events logged</span><span>{total}</span></div>
  <div class="row"><span>Accepted</span><span>{len(accepts)}</span></div>
  <div class="row"><span>Rejected</span><span>{len(rejects)}</span></div>
</div>
<div class="card">
  <div class="big">Per person (accepted)</div>
  {per_person_html or '<div class="row"><span>No accepted detections yet</span></div>'}
</div>
<p><a href="/report/csv">Download full CSV log</a></p>
<h3 style="font-size:15px;">Last {min(30,total)} events</h3>
{rows_html or '<div class="card">No events logged yet. Try the photo test or live stream.</div>'}
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/report/csv")
def report_csv():
    if not LOG_CSV_PATH.exists():
        return error_response("No log file yet -- no detections have been run.", 404)
    return send_file(LOG_CSV_PATH, as_attachment=True, download_name="adar_session_log.csv")


@app.route("/detect", methods=["POST"])
def detect():
    t_start = time.perf_counter()

    raw_bytes = request.get_data()
    if not raw_bytes:
        return error_response("No image data received in request body.", 400)

    try:
        np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    except Exception:
        return error_response(f"Failed to decode image bytes: {traceback.format_exc(limit=1)}", 400)

    if frame is None:
        return error_response("Image data could not be decoded (not a valid image).", 400)

    if STATE["face_app"] is None or STATE["calib"] is None or STATE["gallery"] is None:
        return error_response("Server not fully initialized (models/calibration not loaded).", 500)

    t_detect_start = time.perf_counter()
    try:
        result = run_detection(frame)
    except Exception:
        return error_response(f"Face detection/embedding failed: {traceback.format_exc(limit=1)}", 500)
    t_detect_end = time.perf_counter()

    for i in range(result["num_faces_detected"]):
        log_event("api", result["employee_detected"][i], result["decision"][i],
                  result["distance_m"][i], result["similarity"][i],
                  result["snr"][i], result["view_matched"][i])

    t_end = time.perf_counter()

    response = {
        "success": True,
        "error": None,
        **result,
        "timing_ms": {
            "detection_and_embedding": round((t_detect_end - t_detect_start) * 1000, 2),
            "matching": 0.0,
            "total": round((t_end - t_start) * 1000, 2),
        },
        "compute_device": "GPU" if STATE["gpu_used"] else "CPU",
        "gpu_used": STATE["gpu_used"],
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    return jsonify(response), 200


@app.route("/health", methods=["GET"])
def health():
    ready = STATE["face_app"] is not None and STATE["calib"] is not None and STATE["gallery"] is not None
    return jsonify({
        "success": True,
        "error": None,
        "ready": ready,
        "gallery_size": len(STATE["gallery"]) if STATE["gallery"] else 0,
        "calibration_dir": STATE["calibration_dir"],
        "gpu_used": STATE["gpu_used"],
        "compute_device": "GPU" if STATE["gpu_used"] else "CPU",
        "active_provider": STATE.get("active_provider", "unknown"),
        "stream_running": STREAM_STATE["running"],
        "events_logged": len(EVENT_LOG),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }), 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_dir", default="calibration_output")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam index for /stream (default 0)")
    parser.add_argument("--no_warmup", action="store_true", help="Skip startup warm-up (not recommended on GPU)")
    args = parser.parse_args()

    ensure_log_file()
    STREAM_STATE["camera_index"] = args.camera_index

    load_state(Path(args.calibration_dir))

    if not args.no_warmup:
        warm_up(STATE["face_app"], args.dataset_root)

    print(f"\n[startup] Serving on http://{args.host}:{args.port}")
    print(f"[startup]   POST raw image bytes to  /detect")
    print(f"[startup]   GET                       /health")
    print(f"[startup]   GET                       /test          <-- photo test, open on phone")
    print(f"[startup]   GET                       /stream_page   <-- live webcam, open on phone")
    print(f"[startup]   GET                       /report        <-- detection report, open on phone")
    print()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

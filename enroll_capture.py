"""
enroll_capture.py
--------------------
Two-phase remote enrollment flow, driven entirely from your phone.

WHY TWO PHASES:
Typing a name and a handful of distances while a live camera preview
is streaming in the background is exactly the situation that caused
the old input-field-gets-clobbered bug. The fix here isn't just a
smarter client-side guard -- it's removing the camera from the
picture entirely while you're typing. The camera process/thread is
not even started until you've finished entering everything and tap
"Start capture".

HOW IT WORKS:
  PHASE 1 -- SETUP (no camera):
    On your phone: type the person's name, choose how many distances
    you're capturing at (a stepper, 1-8), then a text box appears for
    each distance (in meters). Nothing here touches the webcam.

  PHASE 2 -- GUIDED CAPTURE (camera opens once, stays open):
    Tapping "Start capture" opens the laptop's webcam and walks you
    through your distances in ascending order. At each distance you
    manually tap CAPTURE four times (front / left / right / top) --
    capture is never automatic, since you have to physically walk to
    each mark and frame yourself. After the 4th view the app advances
    to the next distance on its own. When every distance is done you
    land on a completion screen with a button to enroll another
    person, which returns you to Phase 1 (and closes the camera again).

Saved files land in the same folder convention every other script in
this toolkit expects:
    dataset/<person>/depth_XXXm/{front,left,right,top}.jpg

SETUP:
1. Same WiFi network for phone + laptop.
2. Find your laptop's local IP (ipconfig / ifconfig).
3. pip install flask-socketio
4. python enroll_capture.py
5. On your phone: http://<that-ip>:5000

Usage (run ONCE, covers every person you enroll this session):
    python enroll_capture.py

Press Ctrl+C in the terminal only when fully done with everyone.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, request
from flask_socketio import SocketIO

from utils import VIEWS, depth_folder_name

VIEW_PROMPTS = {
    "front": "Look straight at the camera (FRONT view)",
    "left": "Turn your head to show your LEFT profile",
    "right": "Turn your head to show your RIGHT profile",
    "top": "Tilt your head down slightly / raise the camera above eye level (TOP view)",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = "adar-enroll-capture"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ---------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------
state = {
    "phase": "setup",       # "setup" | "capture" | "done"
    "person": None,
    "distances": [],        # sorted list of floats, set once at "Start capture"
    "dist_index": 0,        # which distance we're currently capturing
    "view_index": 0,        # which of VIEWS within the current distance
    "last_saved": None,
    "frame": None,           # latest raw camera frame (BGR numpy array)
    "lock": threading.Lock(),
}

# Camera is only ever running during the "capture" phase -- started when
# Phase 1 is submitted, stopped when a session finishes or is reset.
camera = {
    "cap": None,
    "thread": None,
    "running": False,
    "index": 0,
}


def _camera_loop():
    cap = camera["cap"]
    while camera["running"]:
        ret, frame = cap.read()
        if ret:
            with state["lock"]:
                state["frame"] = frame
        time.sleep(0.03)  # ~30fps


def start_camera(camera_index):
    if camera["running"]:
        return
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera_index (0, 1, 2...).")
    camera["cap"] = cap
    camera["running"] = True
    t = threading.Thread(target=_camera_loop, daemon=True)
    camera["thread"] = t
    t.start()


def stop_camera():
    camera["running"] = False
    if camera["cap"] is not None:
        camera["cap"].release()
        camera["cap"] = None
    with state["lock"]:
        state["frame"] = None


def _current_out_dir():
    person = state["person"]
    depth = state["distances"][state["dist_index"]]
    return Path(app.config["DATASET_ROOT"]) / person / depth_folder_name(depth)


def _status_payload():
    with state["lock"]:
        payload = {
            "phase": state["phase"],
            "person": state["person"],
            "distances": state["distances"],
            "dist_index": state["dist_index"],
            "view_index": state["view_index"],
            "last_saved": state["last_saved"],
        }
    if payload["phase"] == "capture" and payload["dist_index"] < len(payload["distances"]):
        payload["current_distance"] = payload["distances"][payload["dist_index"]]
        payload["prompt"] = VIEW_PROMPTS[VIEWS[payload["view_index"]]]
    else:
        payload["current_distance"] = None
        payload["prompt"] = ""
    return payload


def _push_status():
    socketio.emit("status", _status_payload())


def mjpeg_generator():
    while True:
        with state["lock"]:
            frame = None if state["frame"] is None else state["frame"].copy()
        if frame is not None:
            if state["phase"] == "capture" and state["dist_index"] < len(state["distances"]):
                view = VIEWS[state["view_index"]]
                cv2.putText(frame, VIEW_PROMPTS[view], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ADAR Enrollment</title>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <style>
    :root {{
      --bg: #0c0d10; --panel: #16181d; --panel-2: #1e2127; --line: #2a2e37;
      --text: #e9ebf0; --dim: #8b909c; --accent: #4fd1a5; --accent-dim: #2a5f4d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            display: flex; justify-content: center; padding: 20px 14px 60px; }}
    .phone {{ width: 100%; max-width: 420px; }}
    .eyebrow {{ font-family: monospace; font-size: 11px; letter-spacing: .12em;
                color: var(--accent); text-transform: uppercase; margin-bottom: 4px; }}
    h1 {{ font-size: 20px; margin: 0 0 18px; font-weight: 600; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
             padding: 18px; margin-bottom: 14px; }}
    label {{ display: block; font-size: 12px; color: var(--dim); margin-bottom: 6px; font-family: monospace; }}
    input[type=text], input[type=number] {{
      width: 100%; background: var(--panel-2); border: 1px solid var(--line); color: var(--text);
      font-size: 16px; padding: 12px 14px; border-radius: 9px; outline: none; }}
    input:focus {{ border-color: var(--accent); }}
    .field {{ margin-bottom: 16px; }}
    .field:last-child {{ margin-bottom: 0; }}
    .distance-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px; }}
    .distance-grid .field {{ margin-bottom: 0; }}
    .distance-grid label {{ font-size: 11px; }}
    button {{ font: inherit; border: none; border-radius: 10px; cursor: pointer; }}
    .btn-primary {{ width: 100%; background: var(--accent); color: #06231a; font-weight: 700;
                    font-size: 16px; padding: 15px; margin-top: 18px; }}
    .btn-primary:disabled {{ background: #34383f; color: #6b6f78; cursor: not-allowed; }}
    .btn-secondary {{ background: var(--panel-2); color: var(--text); border: 1px solid var(--line);
                      font-size: 14px; padding: 9px 14px; width: 100%; margin-top: 8px; }}
    .stepper {{ display: flex; align-items: center; justify-content: space-between;
                background: var(--panel-2); border: 1px solid var(--line); border-radius: 9px; padding: 4px; }}
    .stepper button {{ background: transparent; color: var(--accent); font-size: 20px; width: 40px; height: 40px; }}
    .stepper .count {{ font-family: monospace; font-size: 20px; font-weight: 700; }}
    .hidden {{ display: none; }}
    .progress-row {{ display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }}
    .dist-pill {{ flex: 1; min-width: 54px; text-align: center; font-family: monospace; font-size: 12px;
                  padding: 8px 4px; border-radius: 8px; background: var(--panel-2); color: var(--dim);
                  border: 1px solid var(--line); }}
    .dist-pill.active {{ background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }}
    .dist-pill.done {{ color: var(--dim); text-decoration: line-through; }}
    img.preview {{ width: 100%; border-radius: 12px; border: 1px solid var(--line); display: block; }}
    .views-row {{ display: flex; gap: 8px; margin: 14px 0; }}
    .view-chip {{ flex: 1; text-align: center; padding: 10px 4px; border-radius: 8px;
                  background: var(--panel-2); border: 1px solid var(--line); font-size: 12px; color: var(--dim); }}
    .view-chip.current {{ border-color: var(--accent); color: var(--accent); }}
    .view-chip.done {{ background: var(--accent-dim); color: var(--accent); }}
    .status-line {{ text-align: center; font-size: 14px; color: var(--dim); margin: 10px 0 4px; }}
    .status-line b {{ color: var(--text); }}
    .btn-capture {{ width: 100%; background: var(--accent); color: #06231a; font-weight: 700;
                    font-size: 17px; padding: 18px; margin-top: 14px; }}
    .btn-capture:disabled {{ background: #34383f; color: #6b6f78; }}
    .note {{ font-size: 12px; color: var(--dim); text-align: center; margin-top: 10px; line-height: 1.5; }}
    #confirm {{ color: var(--accent); font-size: 12px; text-align: center; min-height: 16px; margin-top: 6px; }}
    .done-screen {{ text-align: center; padding: 30px 10px; }}
    .done-screen .check {{ width: 56px; height: 56px; border-radius: 50%; background: var(--accent-dim);
                           color: var(--accent); font-size: 28px; display: flex; align-items: center;
                           justify-content: center; margin: 0 auto 16px; }}
    #connIndicator {{ position: fixed; top: 8px; right: 10px; font-size: 12px; padding: 4px 10px;
                       border-radius: 10px; background: #333; }}
    #connIndicator.connected {{ background: #1e5c33; color: #6fe38f; }}
    #connIndicator.disconnected {{ background: #5c1e1e; color: #ff8a8a; }}
  </style>
</head>
<body>
<div id="connIndicator">Connecting...</div>
<div class="phone">

  <div id="phase-setup">
    <div class="eyebrow">Step 1 of 2 -- no camera</div>
    <h1>Enrollment setup</h1>
    <div class="card">
      <div class="field">
        <label for="nameInput">Person name</label>
        <input type="text" id="nameInput" placeholder="e.g. person_01">
      </div>
      <div class="field">
        <label>Number of distances to capture</label>
        <div class="stepper">
          <button type="button" onclick="changeCount(-1)">-</button>
          <span class="count" id="countDisplay">3</span>
          <button type="button" onclick="changeCount(1)">+</button>
        </div>
      </div>
      <div class="field">
        <label>Distance values (meters)</label>
        <div class="distance-grid" id="distanceGrid"></div>
      </div>
    </div>
    <button class="btn-primary" id="startBtn" onclick="startCapture()">Start capture &rarr;</button>
    <div class="note">The camera stays off until every field here is filled in and submitted.</div>
    <div id="setupError" class="note" style="color:#ff8a6b;"></div>
  </div>

  <div id="phase-capture" class="hidden">
    <div class="eyebrow" id="personLabel"></div>
    <h1>Guided capture</h1>
    <div class="progress-row" id="progressRow"></div>
    <div class="card">
      <img class="preview" src="/video_feed">
      <div class="views-row" id="viewsRow">
        <div class="view-chip" data-view="front">FRONT</div>
        <div class="view-chip" data-view="left">LEFT</div>
        <div class="view-chip" data-view="right">RIGHT</div>
        <div class="view-chip" data-view="top">TOP</div>
      </div>
      <div class="status-line" id="statusLine">Loading...</div>
      <button class="btn-capture" id="captureBtn" onclick="doCapture()">CAPTURE</button>
      <div id="confirm"></div>
    </div>
  </div>

  <div id="phase-done" class="hidden">
    <div class="done-screen">
      <div class="check">&#10003;</div>
      <h1>All distances captured</h1>
      <p class="note" id="doneSummary"></p>
      <button class="btn-secondary" onclick="startOver()">Enroll another person</button>
    </div>
  </div>

</div>

<script>
const socket = io();
const connIndicator = document.getElementById('connIndicator');
const setupDiv = document.getElementById('phase-setup');
const captureDiv = document.getElementById('phase-capture');
const doneDiv = document.getElementById('phase-done');

socket.on('connect', () => {{ connIndicator.innerText = 'Connected'; connIndicator.className = 'connected'; }});
socket.on('disconnect', () => {{ connIndicator.innerText = 'Reconnecting...'; connIndicator.className = 'disconnected'; }});

let distCount = 3;
function renderDistanceGrid() {{
  const grid = document.getElementById('distanceGrid');
  grid.innerHTML = '';
  for (let i = 0; i < distCount; i++) {{
    const field = document.createElement('div');
    field.className = 'field';
    field.innerHTML = `<label>Distance ${{i + 1}}</label><input type="number" step="0.1" min="0" class="distInput">`;
    grid.appendChild(field);
  }}
}}
function changeCount(delta) {{
  distCount = Math.max(1, Math.min(8, distCount + delta));
  document.getElementById('countDisplay').innerText = distCount;
  renderDistanceGrid();
}}
renderDistanceGrid();

function startCapture() {{
  const errEl = document.getElementById('setupError');
  const name = document.getElementById('nameInput').value.trim();
  const inputs = Array.from(document.querySelectorAll('.distInput'));
  const vals = inputs.map(i => parseFloat(i.value));
  errEl.innerText = '';
  if (!name) {{ errEl.innerText = 'Enter a person name.'; return; }}
  if (vals.some(v => isNaN(v) || v < 0)) {{ errEl.innerText = 'Fill in every distance box with a valid number.'; return; }}
  document.getElementById('startBtn').disabled = true;
  socket.emit('start_capture', {{person: name, distances: vals}});
}}

function doCapture() {{
  document.getElementById('captureBtn').disabled = true;
  socket.emit('capture');
}}

function startOver() {{
  socket.emit('reset');
}}

socket.on('setup_error', (j) => {{
  document.getElementById('setupError').innerText = j.message;
  document.getElementById('startBtn').disabled = false;
}});

socket.on('status', (j) => {{
  document.getElementById('startBtn').disabled = false;

  if (j.phase === 'setup') {{
    setupDiv.classList.remove('hidden');
    captureDiv.classList.add('hidden');
    doneDiv.classList.add('hidden');
    return;
  }}

  if (j.phase === 'capture') {{
    setupDiv.classList.add('hidden');
    captureDiv.classList.remove('hidden');
    doneDiv.classList.add('hidden');

    document.getElementById('personLabel').innerText = (j.person || '').toUpperCase();

    const row = document.getElementById('progressRow');
    row.innerHTML = '';
    j.distances.forEach((d, i) => {{
      const pill = document.createElement('div');
      pill.className = 'dist-pill' + (i === j.dist_index ? ' active' : '') + (i < j.dist_index ? ' done' : '');
      pill.innerText = d + 'm';
      row.appendChild(pill);
    }});

    document.querySelectorAll('.view-chip').forEach((chip, i) => {{
      chip.classList.toggle('current', i === j.view_index);
      chip.classList.toggle('done', i < j.view_index);
    }});

    document.getElementById('statusLine').innerHTML =
      j.person + ' @ <b>' + j.current_distance + 'm</b> -- view ' + (j.view_index + 1) + ' of 4: ' + j.prompt;
    document.getElementById('captureBtn').disabled = false;

    if (j.last_saved) {{
      document.getElementById('confirm').innerText = 'Saved: ' + j.last_saved;
    }}
    return;
  }}

  if (j.phase === 'done') {{
    setupDiv.classList.add('hidden');
    captureDiv.classList.add('hidden');
    doneDiv.classList.remove('hidden');
    document.getElementById('doneSummary').innerText =
      'Captured 4 views at each of ' + j.distances.length + ' distances (' + j.distances.join('m, ') + 'm) for ' + j.person + '.';
  }}
}});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    # .format() with no args collapses the {{ }} escaping used throughout
    # the embedded CSS/JS back down to literal { } -- there are no actual
    # server-side placeholders left to fill in (person/depth are no longer
    # pre-rendered into the page; Phase 1 collects them entirely client-side).
    return PAGE_TEMPLATE.format()


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("connect")
def on_connect():
    socketio.emit("status", _status_payload(), to=request.sid)


@socketio.on("start_capture")
def on_start_capture(payload):
    person = str(payload.get("person", "")).strip()
    raw_distances = payload.get("distances", [])
    try:
        distances = sorted(float(d) for d in raw_distances)
    except (TypeError, ValueError):
        socketio.emit("setup_error", {"message": "Invalid distance value."}, to=request.sid)
        return

    if not person or not distances:
        socketio.emit("setup_error", {"message": "Person name and at least one distance are required."}, to=request.sid)
        return

    with state["lock"]:
        state["phase"] = "capture"
        state["person"] = person
        state["distances"] = distances
        state["dist_index"] = 0
        state["view_index"] = 0
        state["last_saved"] = None

    try:
        start_camera(camera["index"])
    except RuntimeError as e:
        with state["lock"]:
            state["phase"] = "setup"
        socketio.emit("setup_error", {"message": str(e)}, to=request.sid)
        return

    _push_status()


@socketio.on("capture")
def on_capture():
    with state["lock"]:
        if state["phase"] != "capture" or state["frame"] is None:
            saved = None
        else:
            view = VIEWS[state["view_index"]]
            out_dir = _current_out_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{view}.jpg"
            cv2.imwrite(str(out_path), state["frame"])
            state["last_saved"] = str(out_path)
            saved = str(out_path)

            state["view_index"] += 1
            if state["view_index"] >= len(VIEWS):
                state["view_index"] = 0
                state["dist_index"] += 1
                if state["dist_index"] >= len(state["distances"]):
                    state["phase"] = "done"

    if state["phase"] == "done":
        stop_camera()

    _push_status()
    return {"saved": saved}


@socketio.on("reset")
def on_reset():
    stop_camera()
    with state["lock"]:
        state["phase"] = "setup"
        state["person"] = None
        state["distances"] = []
        state["dist_index"] = 0
        state["view_index"] = 0
        state["last_saved"] = None
    _push_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets your phone reach this over WiFi")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.config["DATASET_ROOT"] = args.dataset_root
    camera["index"] = args.camera_index

    print(f"\nStarting two-phase enrollment server.")
    print("On your PHONE (same WiFi as this laptop), open:")
    print(f"   http://<this-laptop's-local-IP>:{args.port}")
    print("Find the local IP with 'ipconfig' (Windows) or 'ifconfig' (Mac/Linux) if you don't know it.")
    print("Phase 1 (name + distances) never touches the camera.")
    print("Phase 2 opens the camera once 'Start capture' is tapped, and closes it again")
    print("once all distances are done or you tap 'Enroll another person'.")
    print("Press Ctrl+C when fully done with everyone.\n")

    socketio.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

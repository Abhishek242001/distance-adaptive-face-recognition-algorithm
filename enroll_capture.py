"""
enroll_capture.py
--------------------
Remote-shutter version of enroll_capture.py, for when you're the only person
around and can't both stand at a distance AND press a key on the laptop.

HOW IT WORKS:
This starts a tiny local web server on your laptop. Your phone (on the
SAME WiFi network as the laptop) opens a webpage served by that
server, which shows a live preview from the laptop's webcam and one
big CAPTURE button. Both the PERSON name and the DISTANCE are set
right there on the phone page -- you never touch the terminal again
after the one command that starts the server.

WEBSOCKET NOTE (why this version doesn't fight you while typing):
The old version polled GET /status every 2 seconds over plain HTTP and
overwrote the Person/Distance boxes with whatever the server had, which
kept clobbering half-typed values even with a "don't overwrite while
focused" guard (timing between the poll tick and a fast phone tap/typing
race is exactly what caused the visible refresh-while-typing).

This version uses a WebSocket (Flask-SocketIO) instead. There is NO
periodic timer at all -- the server only ever *pushes* a status update
when something actually changes: a capture happens, or you tap
"Set person" / "Set distance". Nothing runs in the background to
interrupt you while you're mid-typing, so the field only updates
when you yourself trigger it.

SETUP:
1. Make sure your phone and laptop are on the same WiFi network.
2. Find your laptop's local IP address:
     Windows:      ipconfig            -> look for "IPv4 Address"
     Mac/Linux:    ifconfig | grep inet
   It will look something like 192.168.1.23
3. Install the one extra dependency this version needs (WebSocket
   support), then run the script once:
     pip install flask-socketio
     python enroll_capture.py
4. On your phone's browser, go to:
     http://<that-ip-address>:5000
5. You'll see a PERSON box, a DISTANCE box, a live preview, and a
   CAPTURE button.
     - Type the person name (e.g. person_01) and tap "Set person".
     - Type the distance in meters (e.g. 0.5) and tap "Set distance".
   Frame yourself for the "front" prompt and tap CAPTURE. The page
   auto-advances through all 4 views (front -> left -> right -> top).
6. For a NEW distance, same person: just walk to the new mark, type
   the new number in the distance box, tap "Set distance" again, and
   keep capturing. This resets the 4-view progress and starts saving
   into that distance's own folder.
7. For a NEW person: type the new name in the person box, tap
   "Set person", then set the distance and continue. All from the
   phone -- the terminal is never touched again.
8. A small indicator in the corner shows "Connected" / "Reconnecting..."
   for the WebSocket link, so you always know if the page has silently
   dropped its connection to the laptop.

Usage (run ONCE, covers your entire session -- every person, every distance):
    python enroll_capture.py

Optional starting values, purely for convenience (both are fully
editable from the phone afterwards, so these are just what the page
shows when it first loads):
    python enroll_capture.py --person person_01 --depth 0.5

Press Ctrl+C in the terminal only when you're completely done with
EVERYONE.
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
# threading async_mode needs no extra dependency beyond flask-socketio
# itself (no eventlet/gevent required) and plays nicely with the
# existing background camera_loop thread below.
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Shared state between the camera-reading thread and the Flask routes.
state = {
    "frame": None,          # latest raw frame (BGR numpy array)
    "view_index": 0,        # which of VIEWS we're currently capturing
    "done": False,          # all 4 views captured for this person+depth
    "last_saved": None,     # path of the most recently saved file, for on-page confirmation
    "person": None,         # current person folder name -- changeable from the UI
    "depth": None,          # current distance in meters -- changeable from the UI
    "lock": threading.Lock(),
}


def _refresh_out_dir():
    """Recompute the save folder from the current person+depth and reset
    view progress. Called whenever either person or depth changes."""
    out_dir = Path(app.config["DATASET_ROOT"]) / state["person"] / depth_folder_name(state["depth"])
    state["view_index"] = 0
    state["done"] = False
    state["last_saved"] = None
    app.config["OUT_DIR"] = str(out_dir)


def _status_payload():
    """Build the same status dict the old /status endpoint returned --
    now pushed over the socket instead of polled."""
    with state["lock"]:
        idx = state["view_index"]
        done = state["done"]
        depth = state["depth"]
        person = state["person"]
        last_saved = state["last_saved"]
    prompt = VIEW_PROMPTS[VIEWS[idx]] if idx < len(VIEWS) else ""
    return {
        "view_index": idx,
        "prompt": prompt,
        "done": done,
        "depth": depth,
        "person": person,
        "last_saved": last_saved,
    }


def _push_status():
    """Broadcast the current status to every connected client (in
    practice just your phone). Only called right after a real change,
    never on a timer."""
    socketio.emit("status", _status_payload())


def set_depth(new_depth):
    """Switch the active capture distance without restarting the server."""
    with state["lock"]:
        state["depth"] = new_depth
        _refresh_out_dir()


def set_person(new_person):
    """Switch the active person without restarting the server."""
    with state["lock"]:
        state["person"] = new_person
        _refresh_out_dir()


def camera_loop(camera_index):
    """Continuously reads frames from the webcam in the background so the
    preview stream and capture button always see a fresh frame, not a
    stale buffered one."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera_index (0, 1, 2...).")
    while True:
        ret, frame = cap.read()
        if ret:
            with state["lock"]:
                state["frame"] = frame
        time.sleep(0.03)  # ~30fps read loop


def mjpeg_generator():
    """Unchanged from before -- the live preview stays a plain MJPEG HTTP
    stream (an <img> tag re-requesting frames), which is a completely
    separate mechanism from the status WebSocket and was never the
    source of the typing-interrupt problem."""
    while True:
        with state["lock"]:
            frame = None if state["frame"] is None else state["frame"].copy()
        if frame is not None:
            view = VIEWS[state["view_index"]] if state["view_index"] < len(VIEWS) else None
            if view:
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
  <title>ADAR Remote Capture</title>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <style>
    body {{ font-family: sans-serif; text-align: center; background: #111; color: #eee; margin: 0; padding: 10px; }}
    img {{ width: 100%; max-width: 480px; border-radius: 8px; }}
    #status {{ font-size: 20px; margin: 12px 0; }}
    button {{ font-size: 24px; padding: 20px 50px; border-radius: 12px; border: none;
              background: #2ecc71; color: white; margin-top: 10px; }}
    button:disabled {{ background: #555; }}
    #confirm {{ color: #2ecc71; min-height: 24px; margin-top: 10px; }}
    .field {{ margin: 10px 0; }}
    .field label {{ font-size: 16px; }}
    .field input {{ font-size: 22px; width: 140px; text-align: center; padding: 6px;
                     border-radius: 8px; border: none; margin-top: 6px; }}
    .field button {{ font-size: 16px; padding: 10px 20px; background: #3498db; margin-left: 8px; }}
    #connIndicator {{ position: fixed; top: 8px; right: 10px; font-size: 12px; padding: 4px 10px;
                       border-radius: 10px; background: #333; }}
    #connIndicator.connected {{ background: #1e5c33; color: #6fe38f; }}
    #connIndicator.disconnected {{ background: #5c1e1e; color: #ff8a8a; }}
  </style>
</head>
<body>
  <div id="connIndicator">Connecting...</div>
  <h2>ADAR Remote Capture</h2>

  <div class="field">
    <label for="personInput">Person:</label><br>
    <input id="personInput" type="text" value="{person}">
    <button onclick="applyPerson()">Set person</button>
  </div>

  <div class="field">
    <label for="depthInput">Distance (m):</label><br>
    <input id="depthInput" type="number" step="0.1" min="0" value="{depth}">
    <button onclick="applyDepth()">Set distance</button>
  </div>

  <div id="status">Loading...</div>
  <img src="/video_feed">
  <br>
  <button id="capBtn" onclick="doCapture()">CAPTURE</button>
  <div id="confirm"></div>

<script>
const socket = io();
const personEl = document.getElementById('personInput');
const depthEl = document.getElementById('depthInput');
const statusEl = document.getElementById('status');
const confirmEl = document.getElementById('confirm');
const capBtn = document.getElementById('capBtn');
const connIndicator = document.getElementById('connIndicator');

socket.on('connect', () => {{
  connIndicator.innerText = 'Connected';
  connIndicator.className = 'connected';
}});

socket.on('disconnect', () => {{
  connIndicator.innerText = 'Reconnecting...';
  connIndicator.className = 'disconnected';
}});

// This is the ONLY place the UI updates from -- pushed by the server
// right after a real change (capture / set_person / set_depth), never
// on a timer. That's what stops it from clobbering the boxes while
// you're typing: there is nothing to race against.
socket.on('status', (j) => {{
  // Still guard against overwriting a field you're actively typing in,
  // in case a change arrives from elsewhere (e.g. a second phone/tab).
  if (document.activeElement !== personEl) personEl.value = j.person;
  if (document.activeElement !== depthEl) depthEl.value = j.depth;

  if (j.done) {{
    statusEl.innerText =
      "All 4 views captured for " + j.person + " at " + j.depth + "m! Change person and/or distance above and keep going, no restart needed.";
    capBtn.disabled = true;
  }} else {{
    statusEl.innerText =
      j.person + " @ " + j.depth + "m -- View " + (j.view_index + 1) + " of 4: " + j.prompt;
    capBtn.disabled = false;
  }}

  if (j.last_saved) {{
    confirmEl.innerText = "Saved: " + j.last_saved;
  }}
}});

function applyPerson() {{
  const newPerson = personEl.value.trim();
  if (!newPerson) return;
  confirmEl.innerText = "";
  socket.emit('set_person', {{person: newPerson}});
}}

function applyDepth() {{
  const newDepth = parseFloat(depthEl.value);
  if (isNaN(newDepth)) return;
  confirmEl.innerText = "";
  socket.emit('set_depth', {{depth: newDepth}});
}}

function doCapture() {{
  capBtn.disabled = true;
  socket.emit('capture');
}}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    with state["lock"]:
        person = state["person"]
        depth = state["depth"]
    return PAGE_TEMPLATE.format(person=person, depth=depth)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("connect")
def on_connect():
    """Send the current status immediately to a newly-connected/
    reconnected client, so the page is never stuck on 'Loading...'."""
    socketio.emit("status", _status_payload(), to=request.sid)


@socketio.on("set_person")
def on_set_person(payload):
    new_person = str(payload["person"]).strip()
    if not new_person:
        return
    set_person(new_person)
    _push_status()


@socketio.on("set_depth")
def on_set_depth(payload):
    try:
        new_depth = float(payload["depth"])
    except (TypeError, ValueError):
        return
    set_depth(new_depth)
    _push_status()


@socketio.on("capture")
def on_capture():
    with state["lock"]:
        if state["done"] or state["frame"] is None:
            saved = None
        else:
            view = VIEWS[state["view_index"]]
            out_dir = Path(app.config["OUT_DIR"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{view}.jpg"
            cv2.imwrite(str(out_path), state["frame"])
            state["last_saved"] = str(out_path)
            state["view_index"] += 1
            if state["view_index"] >= len(VIEWS):
                state["done"] = True
            saved = str(out_path)
    _push_status()
    return {"saved": saved}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default="person_01",
                         help="STARTING person name (e.g. person_01). You can change this later "
                              "from the phone UI without restarting the script.")
    parser.add_argument("--depth", type=float, default=1.0,
                         help="STARTING distance in meters (e.g. 0.5). You can change this later "
                              "from the phone UI without restarting the script.")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets your phone reach this over WiFi")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.config["DATASET_ROOT"] = args.dataset_root

    with state["lock"]:
        state["person"] = args.person
        state["depth"] = args.depth
        _refresh_out_dir()

    t = threading.Thread(target=camera_loop, args=(args.camera_index,), daemon=True)
    t.start()

    print(f"\nStarting remote capture server (starting person='{args.person}', starting depth={args.depth}m)")
    print("On your PHONE (same WiFi as this laptop), open:")
    print(f"   http://<this-laptop's-local-IP>:{args.port}")
    print("Find the local IP with 'ipconfig' (Windows) or 'ifconfig' (Mac/Linux) if you don't know it.")
    print("Both PERSON and DISTANCE are editable right from the phone page from now on --")
    print("you do NOT need to touch this terminal again until you're fully done. Press Ctrl+C then.\n")
    print("(Status updates now travel over a WebSocket, pushed only on real changes --")
    print("no more background polling clobbering the boxes while you type.)\n")

    socketio.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

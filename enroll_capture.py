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

SETUP:
1. Make sure your phone and laptop are on the same WiFi network.
2. Find your laptop's local IP address:
     Windows:      ipconfig            -> look for "IPv4 Address"
     Mac/Linux:    ifconfig | grep inet
   It will look something like 192.168.1.23
3. Run this script ONCE (see Usage below) -- no arguments are required.
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
from flask import Flask, Response, jsonify, request

from utils import VIEWS, depth_folder_name

VIEW_PROMPTS = {
    "front": "Look straight at the camera (FRONT view)",
    "left": "Turn your head to show your LEFT profile",
    "right": "Turn your head to show your RIGHT profile",
    "top": "Tilt your head down slightly / raise the camera above eye level (TOP view)",
}

app = Flask(__name__)

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
  </style>
</head>
<body>
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
async function refreshStatus() {{
  const r = await fetch('/status');
  const j = await r.json();
  // Only overwrite these inputs if the user is NOT currently typing in
  // them -- otherwise the 2-second poll below stomps on whatever they're
  // mid-typing (e.g. deleting "1" to type "5") before they get a chance
  // to tap "Set". document.activeElement is the element currently
  // focused/being typed into.
  const personEl = document.getElementById('personInput');
  const depthEl = document.getElementById('depthInput');
  if (document.activeElement !== personEl) {{
    personEl.value = j.person;
  }}
  if (document.activeElement !== depthEl) {{
    depthEl.value = j.depth;
  }}
  if (j.done) {{
    document.getElementById('status').innerText =
      "All 4 views captured for " + j.person + " at " + j.depth + "m! Change person and/or distance above and keep going, no restart needed.";
    document.getElementById('capBtn').disabled = true;
  }} else {{
    document.getElementById('status').innerText =
      j.person + " @ " + j.depth + "m -- View " + (j.view_index+1) + " of 4: " + j.prompt;
    document.getElementById('capBtn').disabled = false;
  }}
}}
async function applyPerson() {{
  const newPerson = document.getElementById('personInput').value.trim();
  if (!newPerson) return;
  document.getElementById('confirm').innerText = "";
  await fetch('/set_person', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{person: newPerson}})
  }});
  await refreshStatus();
}}
async function applyDepth() {{
  const newDepth = document.getElementById('depthInput').value;
  document.getElementById('confirm').innerText = "";
  await fetch('/set_depth', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{depth: parseFloat(newDepth)}})
  }});
  await refreshStatus();
}}
async function doCapture() {{
  document.getElementById('capBtn').disabled = true;
  const r = await fetch('/capture', {{method: 'POST'}});
  const j = await r.json();
  document.getElementById('confirm').innerText = j.saved ? ("Saved: " + j.saved) : "No face-safe frame yet, try again";
  await refreshStatus();
  document.getElementById('capBtn').disabled = j.done;
}}
refreshStatus();
setInterval(refreshStatus, 2000);
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


@app.route("/status")
def status():
    with state["lock"]:
        idx = state["view_index"]
        done = state["done"]
        depth = state["depth"]
        person = state["person"]
    prompt = VIEW_PROMPTS[VIEWS[idx]] if idx < len(VIEWS) else ""
    return jsonify({"view_index": idx, "prompt": prompt, "done": done, "depth": depth, "person": person})


@app.route("/set_depth", methods=["POST"])
def set_depth_route():
    """Called when you tap 'Set distance' on the phone. Lets you change
    distance on the fly -- no need to Ctrl+C and re-run the script."""
    payload = request.get_json(force=True)
    new_depth = float(payload["depth"])
    set_depth(new_depth)
    return jsonify({"ok": True, "depth": new_depth})


@app.route("/set_person", methods=["POST"])
def set_person_route():
    """Called when you tap 'Set person' on the phone. Lets you switch to
    a new person on the fly -- no need to Ctrl+C and re-run the script."""
    payload = request.get_json(force=True)
    new_person = str(payload["person"]).strip()
    set_person(new_person)
    return jsonify({"ok": True, "person": new_person})


@app.route("/capture", methods=["POST"])
def capture():
    with state["lock"]:
        if state["done"] or state["frame"] is None:
            return jsonify({"saved": None, "done": state["done"]})
        view = VIEWS[state["view_index"]]
        out_dir = Path(app.config["OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{view}.jpg"
        cv2.imwrite(str(out_path), state["frame"])
        state["last_saved"] = str(out_path)
        state["view_index"] += 1
        if state["view_index"] >= len(VIEWS):
            state["done"] = True
        return jsonify({"saved": str(out_path), "done": state["done"]})


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

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

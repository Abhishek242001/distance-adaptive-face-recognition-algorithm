# Distance Adaptive Face Recognition Algorithm (ADAR)

**Developed by Abhishek Kumar Shukla**

A distance-adaptive face recognition toolkit that explores how face
recognition accuracy degrades as the distance between a person and a
camera increases, and builds a principled, mathematically-grounded
correction for it — rather than treating every detection at every
distance as equally trustworthy.

The core idea: a face recognition embedding captured at 8 meters is
**not** as reliable as one captured at 0.5 meters, even if the raw
cosine similarity number looks similar. This project builds a
calibrated Signal-to-Noise Ratio (SNR) gate that adjusts for this,
so the system says "I don't know" instead of confidently guessing
wrong once someone is too far away to be reliably recognized.

See **`ALGORITHM.md`** for the full mathematical explanation of how
this works, with worked numeric examples at every step.

## Repository structure

```
api_server.py         Flask HTTP API — the production entry point
calibrate.py           Fits the decay curve (k, gamma, sigma0) from captured photos
enroll_capture.py       Phone-based remote-shutter capture tool for enrollment (WebSocket-based)
utils.py                 Shared face detection / embedding / geometry helpers
adar_admin.py              ONE-FILE all-in-one admin panel: enroll + calibrate + live
                             inference from a single browser page (WebSocket-based)
requirements.txt
README.md                 (this file)
ALGORITHM.md               Full mathematical walkthrough of the algorithm
V3_README.md                 Detailed setup/usage notes and worked examples
```

Two ways to use this project:

- **4-script pipeline** (`enroll_capture.py` → `calibrate.py` → `api_server.py`) —
  run each step separately, closer to the original research workflow.
- **`adar_admin.py`** — the same underlying math and enrollment method in
  one script with a single browser-based admin panel (enroll, calibrate,
  and run live inference all from one page, no separate steps).

Both the remote enrollment page (`enroll_capture.py`) and the admin
panel (`adar_admin.py`) push live status over a **WebSocket**
(Flask-SocketIO) instead of polling on a timer, so the Person/Distance
fields never get overwritten while you're mid-typing on a phone.

## Served routes (`api_server.py`)

| Route | Method | Purpose |
|---|---|---|
| `/detect` | POST | Raw image bytes in → JSON detection result out |
| `/health` | GET | Readiness/liveness check |
| `/test` | GET | Mobile-friendly photo test page |
| `/stream/start` | POST | Starts the laptop's webcam + live detection loop |
| `/stream/stop` | POST | Stops it |
| `/stream` | GET | Live MJPEG video feed |
| `/stream_page` | GET | Mobile-friendly live viewer (Start/Stop + feed) |
| `/report` | GET | Detection history + summary |
| `/report/csv` | GET | Downloadable full detection log |

## Quick start

```bash
pip install -r requirements.txt
```

**Option A — 4-script pipeline:**

```bash
# 1. Enroll people (phone becomes the remote shutter)
python enroll_capture.py

# 2. Fit the distance-decay calibration from captured photos
python calibrate.py --dataset_root dataset --output_dir calibration_output

# 3. Run the live API server
python api_server.py --calibration_dir calibration_output --port 8000
```

Then open `http://<your-laptop-ip>:8000/test` on your phone (same WiFi)
to try a live detection, or `/stream_page` to view the laptop's webcam
with live detection boxes.

**Option B — single-file admin panel:**

```bash
python adar_admin.py
```

Then open `http://<your-laptop-ip>:8000` (or the port you pass with
`--port`) on your phone or laptop. You'll land on a choice screen —
**Enroll a face** or **Start live inference** — and everything (enroll,
calibrate, live view, report) happens from that one page.

`requirements.txt` includes `flask-socketio`, needed by both
`enroll_capture.py` and `adar_admin.py` for their WebSocket-based
status updates.

## GPU acceleration

The pipeline automatically prefers, in order: **CUDA** (NVIDIA GPUs
with the full CUDA Toolkit installed) → **DirectML** (any DirectX12
GPU on Windows, no separate toolkit needed) → **CPU** (always works,
slowest). See `ALGORITHM.md` section 9 for details and real
benchmark numbers from testing.

## A note on data

`dataset/` (real and synthetic face photos), `calibration_output/`
(fitted calibration numbers), and `logs/` (detection history) are
intentionally excluded from version control via `.gitignore`. They
contain real face photos and per-deployment calibration results, and
are meant to be captured/generated locally on each machine, not
versioned or shared through this repository.

## License / usage note

This is a research and prototyping project. It has not been audited
for production security, adversarial robustness, or bias/fairness
across diverse populations, and should not be deployed for access
control, surveillance, or any decision affecting a real person's
safety, employment, or rights without substantially more validation,
real-world testing across diverse individuals and conditions, and a
formal accuracy/fairness assessment.

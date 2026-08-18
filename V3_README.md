# ADAR v3 -- ArcFace + SNR Pipeline, Served as an API

This is v3 of the ADAR toolkit. Scope, on purpose: **ArcFace embeddings
+ the SNR-gated distance correction, and nothing else.** The from-scratch
distance-adaptive CNN backbone from v2 (`6_distance_adaptive_cnn.py`)
is parked for later, once real training data exists -- it is not part
of this package.

What changed vs v2: instead of five separate numbered scripts you run
by hand and a webcam window you watch, v3 is a small, complete,
production-style pipeline:

```
enroll_capture.py   -> capture reference photos (phone remote-shutter)
calibrate.py         -> fit k, gamma, sigma0 from those photos
api_server.py         -> HTTP API: send an image, get back JSON
```

You run each script once (or re-run `calibrate.py` whenever you add
people), then leave `api_server.py` running and just POST images to it.

---

## 1. Setup

```
pip install -r requirements.txt
```

Same dependencies as v2, plus `flask` (already listed).

### GPU (optional, but worth doing if you have an NVIDIA card)

By default, `pip install onnxruntime` (the plain, non-GPU package) has
**no CUDA support at all**, even on a machine with a capable GPU --
this isn't a bug, it's just a separate package. `get_face_app()` in
`utils.py` will happily run on CPU either way, so nothing breaks if
you skip this section, it'll just be slower.

To actually use your GPU:

1. Confirm you have an NVIDIA GPU + driver installed:
```
   nvidia-smi
```
   If this prints a table with your GPU name and a CUDA version, you're
   good. If the command isn't recognized, you don't have NVIDIA
   drivers set up and should skip this section.

2. Install the GPU-enabled onnxruntime package instead of the plain one:
```
   pip uninstall onnxruntime -y
   pip install onnxruntime-gpu
```
   (`requirements.txt` already lists `onnxruntime-gpu` -- this step
   only matters if you'd previously installed plain `onnxruntime`.)

3. That's it -- no code changes needed. `get_face_app()` in `utils.py`
   automatically checks `onnxruntime.get_available_providers()` at
   startup and requests `CUDAExecutionProvider` first if it's present,
   falling back to CPU otherwise.

**How to confirm it's actually working** (don't just assume): start
`api_server.py` and check the startup log line
`[startup] Active execution provider: ...` -- it should say
`CUDAExecutionProvider`, not `CPUExecutionProvider`. The `/health`
endpoint also reports this live as `active_provider`, and every
`/detect` response includes real `timing_ms` for that specific
request, so you can watch `detection_and_embedding` drop from the
tens-of-milliseconds range (CPU) to low single digits (GPU) on real
traffic.

Note: `gpu_used` reports whether the model is **actually running** on
CUDA right now, not merely whether a GPU is present on the machine --
those are different things. A GPU can be present and CUDA-capable and
the model can still silently fall back to CPU (missing runtime DLLs,
version mismatch, etc), which is exactly why this field checks the
live ONNX Runtime session's active provider rather than just listing
what's installed.

---

## 2. Step-by-step pipeline

### Step A -- Enroll people: `enroll_capture.py`

Identical workflow to v2's `1_capture_remote.py`: run once, your phone
becomes the shutter button, both the person's name and the distance
are editable live from the phone page, no terminal re-entry between
people or distances.

```
python enroll_capture.py
```

Then on your phone: `http://<laptop-ip>:5000`. Set person name, set
distance in meters, capture the 4 views (front/left/right/top), repeat
for every distance and every person. Saves into:

```
dataset/<person_id>/depth_XXXm/<view>.jpg
```

**Capture at multiple real distances per person if you can** (e.g.
0.5m AND 1.5m AND 4m), not just one. The near-range k/gamma fit is
only as good as the spread of real data points behind it -- one
distance point can't distinguish "similarity decays with distance"
from "similarity decays with anything," which is why `near_gamma`
came out as exactly 0.0 in earlier testing (see "Known limitation"
at the bottom).

### Step B -- Calibrate: `calibrate.py`

```
python calibrate.py --dataset_root dataset --output_dir calibration_output
```

Reads every captured image, extracts ArcFace embeddings, and fits the
decay curve described in section 3 below. Writes
`calibration_output/calibration_results.json` and
`calibration_output/gallery.npz` (the actual reference embeddings the
API compares against).

### Step C -- Run the API: `api_server.py`

```
python api_server.py --calibration_dir calibration_output --port 8000
```

Leave this running. It loads the face model and your calibration
**once** at startup, then answers requests.

---

## 3. The math, with real worked numbers

Every one of these is implemented exactly as written here in
`api_server.py` (`alpha()`, `sigma()`, `snr()`, `match_face()`) -- this
isn't a simplified explanation, it's the literal code.

### 3.1 Step 1: estimate distance from face size (pinhole model)

A face has some real width in the physical world. In the camera image,
that face occupies a certain number of pixels, and that pixel width
shrinks the farther away the face is -- specifically, in a simple
pinhole camera model, apparent width is **inversely proportional** to
distance. If we know how wide someone's face appeared (in pixels) at
one known distance (their enrollment distance), we can invert that
relationship to estimate distance from any new pixel width:

```
depth_est = ref_depth_m * (ref_width_px / live_width_px)
```

**Worked example:** Say at enrollment, someone's face was measured at
`ref_width_px = 220` pixels at `ref_depth_m = 0.5` meters. Later, in a
live frame, their face is only `live_width_px = 90` pixels wide
(they've backed away):

```
depth_est = 0.5 * (220 / 90) = 1.222 m
```

That's the number that lands in the API response's `distance_m` field.

### 3.2 Step 2: correct the embedding for distance (alpha decay)

As a face moves farther from the camera, image quality degrades --
fewer pixels on the face, more compression/blur/noise -- and the
resulting ArcFace embedding's usefulness degrades with it. Calibration
(`calibrate.py`) fits a single decay constant `k` describing how fast
similarity-to-reference falls off with distance, using this model:

```
alpha(d) = exp(-k * d)
```

where `d` is the *relative* depth -- distance beyond the person's own
enrollment reference distance, `d = depth_est - ref_depth_m`, clipped
to be non-negative.

**Worked example**, continuing the numbers above, using the real `k`
value this pipeline actually measured in earlier calibration
(`near_k = 0.41436`):

```
relative_depth = 1.222 - 0.5 = 0.722 m
alpha = exp(-0.41436 * 0.722) = 0.7414
```

Interpretation: at this distance, the raw embedding's usable "signal"
is expected to have shrunk to about **74.1%** of what it was at the
reference distance. The pipeline uses this to *correct* the live
embedding back up before comparing it:

```
corrected_embedding = live_embedding / alpha
```

This is why a face at 1.2m can still score a reasonable similarity
against a 0.5m reference photo -- the correction compensates for the
expected degradation before the cosine similarity is even computed.

### 3.3 Step 3: model how noisy that correction is (sigma growth)

The correction above is a point estimate, not a guarantee -- the
*uncertainty* of the correction also grows with distance, because
farther-away detections are individually less reliable (more blur,
worse focus, more compression artifacts). This is modeled as:

```
sigma(d) = sigma0 * (1 + gamma * d)
```

`sigma0` is baseline noise at zero relative depth; `gamma` is how fast
that noise grows per additional meter of distance.

**Worked example**, using representative starting values
`sigma0 = 0.05`, `gamma = 0.02` (your own `calibration_results.json`
will contain the values actually fit from your data):

```
sigma(0.722) = 0.05 * (1 + 0.02 * 0.722) = 0.05072
```

### 3.4 Step 4: combine into a Signal-to-Noise Ratio, and gate on it

The whole point of the SNR gate is: don't just trust cosine
similarity blindly -- weigh it against how noisy the correction was at
that distance. The formula:

```
SNR(d) = alpha(d)^2 * ||ref_embedding||^2 / sigma(d)^2
```

**Worked example (ACCEPT case)**, continuing the numbers above, with a
normalized reference embedding so `||ref||^2 = 1.0`:

```
SNR = (0.7414)^2 * 1.0 / (0.05072)^2
    = 0.5497 / 0.002573
    = 213.6
```

Compared against `SNR_ACCEPT_THRESHOLD = 5.0`: since `213.6 >= 5.0`,
this face passes the SNR side of the gate. (It still separately needs
`similarity >= sim_accept_threshold`, default 0.5, to fully ACCEPT --
SNR alone isn't sufficient, it's a necessary condition.)

**Worked example (REJECT case)** -- same person, but now much farther
away, `live_width_px = 15` (barely visible in frame):

```
depth_est = 0.5 * (220 / 15) = 7.33 m
relative_depth = 7.33 - 0.5 = 6.83 m
alpha = exp(-0.41436 * 6.83) = 0.0589
sigma(6.83) = 0.05 * (1 + 0.02*6.83) = 0.0568
SNR = (0.0589)^2 * 1.0 / (0.0568)^2 = 1.075
```

Since `1.075 < 5.0`, this face is REJECTed on the SNR side alone --
correctly, since at that distance the correction is too unreliable to
trust, independent of whatever raw similarity number comes out.

This is also the formal version of the "blind spot" behavior discussed
separately: past some distance, SNR structurally can never clear the
threshold, and the system correctly reports REJECT / "unknown" rather
than guessing.

---

## 4. The API

### `POST /detect`

**Request:** raw binary image bytes in the request body (NOT a file
path, NOT base64 -- the actual decoded/encoded image bytes, e.g. the
raw contents of a `.jpg` or `.png` file). Content-Type doesn't matter
to the server; it decodes whatever bytes it receives with OpenCV.

Optional query parameter: `?sim_accept_threshold=0.5` (default shown).

```bash
curl -X POST --data-binary @test.jpg \
     -H "Content-Type: application/octet-stream" \
     "http://localhost:8000/detect"
```

```python
import requests
with open("test.jpg", "rb") as f:
    img_bytes = f.read()
r = requests.post("http://localhost:8000/detect", data=img_bytes)
print(r.json())
```

**Response shape.** Every per-face field is a **parallel list** --
same length, same order, across every field. `employee_detected[i]`,
`bounding_boxes[i]`, `distance_m[i]`, etc. all describe the *same*
detected face at index `i`.

```json
{
  "success": true,
  "error": null,
  "num_faces_detected": 2,
  "employee_detected": ["Abhishek", "unknown"],
  "bounding_boxes": [[120, 80, 260, 240], [400, 90, 520, 230]],
  "distance_m": [0.62, 1.85],
  "similarity": [0.81, 0.34],
  "snr": [42.7, 3.1],
  "decision": ["ACCEPT", "REJECT"],
  "view_matched": ["front", "front"],
  "timing_ms": {
    "detection_and_embedding": 47.3,
    "matching": 0.11,
    "total": 47.6
  },
  "compute_device": "CPU",
  "gpu_used": false,
  "timestamp": "2026-08-09T12:00:00+05:30"
}
```

Field notes:

- **`employee_detected`** -- the matched person's display name if
  ACCEPTed, otherwise the literal string `"unknown"`.
- **`bounding_boxes`** -- `[x1, y1, x2, y2]` pixel coordinates in the
  original submitted image.
- **`distance_m`** -- the pinhole depth estimate (section 3.1).
- **`similarity`** -- cosine similarity after the alpha correction
  (section 3.2), between 0 and 1.
- **`snr`** -- the signal-to-noise ratio (section 3.4).
- **`decision`** -- `"ACCEPT"` or `"REJECT"` (requires both
  `snr >= 5.0` AND `similarity >= sim_accept_threshold`).
- **`view_matched`** -- which enrolled view (front/left/right/top)
  produced the best match.
- Zero faces detected in a valid image is a **normal, successful**
  response: `success: true`, all lists empty, `num_faces_detected: 0`.
  This is not an error.

**Error responses** (`success: false`) happen only for genuine
failures -- corrupt/undecodable image bytes, empty request body, model
or calibration not loaded, or an unexpected exception during
detection. All per-face list fields are still present but empty, so
callers never need to special-case a missing key:

```json
{
  "success": false,
  "error": "Image data could not be decoded (not a valid image).",
  "num_faces_detected": 0,
  "employee_detected": [],
  "bounding_boxes": [],
  "distance_m": [],
  "similarity": [],
  "snr": [],
  "decision": [],
  "view_matched": [],
  "timing_ms": null,
  "compute_device": "CPU",
  "gpu_used": false,
  "timestamp": "2026-08-09T12:00:01+05:30"
}
```

### `GET /health`

Liveness/readiness check -- confirms models and calibration are
actually loaded before you start sending traffic:

```json
{
  "success": true,
  "error": null,
  "ready": true,
  "gallery_size": 8,
  "calibration_dir": "calibration_output",
  "gpu_used": false,
  "compute_device": "CPU",
  "active_provider": "CPUExecutionProvider",
  "timestamp": "2026-08-09T12:00:00+05:30"
}
```

---

## 5. Performance notes (500-person gallery)

From earlier capacity analysis, carried forward as a note (not yet
independently re-benchmarked for v3):

- **Memory:** trivial. 500 people x 4 views x 512-dim float32
  embeddings = ~3.9 MB total gallery size.
- **Matching speed:** trivial. Comparing one live embedding against
  2000 stored vectors takes well under 1 ms (matrix-multiply cosine
  similarity).
- **Actual bottleneck:** face detection + ArcFace embedding
  extraction, not matching. On CPU only, expect roughly 35-95 ms per
  frame (~10-28 fps). A GPU execution provider drops this to low
  single-digit milliseconds (100+ fps), which matters once this needs
  to run continuously against multiple camera feeds.
- The `/detect` response's `timing_ms` field reports real, per-request
  timing on your actual hardware, so you don't have to take these
  ballpark numbers on faith -- watch the numbers coming back from real
  traffic.

## 6. Known limitation carried over from v2

Only one real captured distance (0.5m) exists in the current dataset
for `person_01`. This is enough to run the pipeline end-to-end, but
`near_gamma` (the noise-growth rate) cannot be meaningfully
distinguished from zero with only one real distance point -- fitting
"how fast does noise grow with distance" needs at least a few distance
values to see growth happen. **Capturing 2-3 real distances per
person** (e.g. 0.5m, 1.5m, 4m) via `enroll_capture.py`, then re-running
`calibrate.py`, will tighten this fit meaningfully.

## 7. Parked for later (not in this package)

- `6_distance_adaptive_cnn.py`, the from-scratch NumPy distance-adaptive
  CNN backbone (Conv2D + FiLM conditioning) from v2 -- explored as a
  research direction, but currently has untrained random weights and
  needs a real training dataset (see v2's `DATASET_GUIDE.md`) before
  it's usable. ArcFace is what this package uses for real matching
  today.
- Metric-learning loss (triplet / ArcFace-margin) as a drop-in
  replacement for the CNN's current classification loss, once real
  data volume justifies training it.
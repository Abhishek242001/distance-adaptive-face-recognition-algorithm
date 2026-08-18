# ADAR: Distance Adaptive Face Recognition Algorithm

**A complete mathematical and architectural walkthrough**

**Developed by Abhishek Kumar Shukla**

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [System Overview](#2-system-overview)
3. [Face Detection and Embedding Extraction](#3-face-detection-and-embedding-extraction)
4. [The Pinhole Camera Model: Estimating Distance from Face Width](#4-the-pinhole-camera-model-estimating-distance-from-face-width)
5. [The Alpha Decay Function: Modeling Similarity Loss with Distance](#5-the-alpha-decay-function-modeling-similarity-loss-with-distance)
6. [The Sigma Noise Function: Modeling Uncertainty Growth with Distance](#6-the-sigma-noise-function-modeling-uncertainty-growth-with-distance)
7. [The Signal-to-Noise Ratio Gate: Combining Everything into a Decision](#7-the-signal-to-noise-ratio-gate-combining-everything-into-a-decision)
8. [Calibration: Fitting k, gamma, and sigma0 from Real Data](#8-calibration-fitting-k-gamma-and-sigma0-from-real-data)
9. [Synthetic Distance Simulation](#9-synthetic-distance-simulation)
10. [Grounding the Distance Range in Real-World Geometry](#10-grounding-the-distance-range-in-real-world-geometry)
11. [GPU Acceleration: CUDA, DirectML, and CPU Fallback](#11-gpu-acceleration-cuda-directml-and-cpu-fallback)
12. [System Architecture: The Production API](#12-system-architecture-the-production-api)
13. [The Live Streaming Pipeline](#13-the-live-streaming-pipeline)
14. [Detection Logging and Reporting](#14-detection-logging-and-reporting)
15. [A Complete End-to-End Worked Example](#15-a-complete-end-to-end-worked-example)
16. [Known Limitations](#16-known-limitations)
17. [Future Work](#17-future-work)
18. [Formula Reference Sheet](#18-formula-reference-sheet)

---

## 1. Introduction and Motivation

### 1.1 The problem with naive face recognition at a distance

Modern face recognition systems, including the one used in this
project (ArcFace, via the InsightFace `buffalo_l` model bundle),
produce an **embedding** for every detected face: a fixed-length
vector of numbers (512 dimensions, in this case) that represents
that face in a mathematical space where similar faces produce
similar vectors. Recognizing a person is then reduced to a geometry
problem: does this new embedding point in roughly the same direction
as a stored reference embedding for a known person?

This works remarkably well when the input image is clean — good
lighting, the person facing the camera, close range, high resolution
on the face region. The problem is that none of these conditions are
guaranteed to hold in a real deployment, especially a CCTV-style
scenario where a person might be standing 50 centimeters from the
camera or 25 meters from it, in the same building, on the same day.

As distance increases, several things get worse simultaneously:

- **Fewer pixels land on the face.** A face that occupies 300 pixels
  of width at 1 meter might occupy only 15 pixels of width at 20
  meters, assuming a fixed camera field of view. Fine detail — the
  exact curve of an eyebrow, the texture of skin, subtle geometry
  around the eyes and nose that the recognition network relies on —
  is simply not present in a 15-pixel-wide face crop.
- **Optical and sensor noise becomes proportionally larger.** The
  same amount of absolute sensor noise (grain, compression
  artifacts, slight motion blur) represents a much larger fraction
  of the available signal when there are fewer pixels to begin with.
- **Focus and depth-of-field issues compound.** Cameras optimized for
  a certain working distance may have degraded sharpness well outside
  that range.

The naive approach — just compute cosine similarity between the live
embedding and the stored reference, and accept if it's above some
fixed threshold like 0.5 — treats a confident, well-lit, close-up
match exactly the same as a noisy, distant, degraded one that happens
to score above the same number. This is dangerous: a system that
doesn't know the difference between "I am confident this is Person
X" and "I am guessing this might be Person X because the noise
happened to line up in the right direction" will eventually make a
wrong, confident-looking decision.

### 1.2 The core idea of this project

Instead of treating similarity as the only signal, this project
builds a **physically motivated model of how recognition reliability
degrades with distance**, and uses that model to compute a
**Signal-to-Noise Ratio (SNR)** for every single detection. A
detection is only accepted as a positive identification if **both**:

1. The (distance-corrected) cosine similarity clears a threshold, **and**
2. The SNR — which accounts for how much the system should trust that
   similarity number at this specific distance — also clears a
   threshold.

This means the system's behavior automatically becomes more
conservative as distance increases, without anyone having to
hand-tune a different threshold for every possible distance. Past
some distance, the SNR gate will *structurally* never clear the bar,
no matter how the raw similarity number happens to land — and the
system correctly reports "unknown / REJECT" rather than guessing.

### 1.3 Why "distance adaptive"

The word "adaptive" here is precise: every part of the matching
pipeline explicitly incorporates the **estimated distance** of the
detected face as an input, and every downstream number — the
corrected similarity, the noise estimate, the SNR, the final
decision — is a **function of that distance**, not a fixed constant.
The system's confidence genuinely changes shape as a person walks
toward or away from the camera, mirroring how a careful human
observer would naturally trust their own eyes less as someone gets
farther away.

---

## 2. System Overview

The complete pipeline consists of five stages, each covered in its
own section below:

```
 [1] ENROLLMENT
     Capture reference photos of each person at a KNOWN distance,
     from 4 views (front / left / right / top).
         |
         v
 [2] CALIBRATION
     Extract embeddings from every captured (and synthetically
     generated) photo, at every known distance, and fit three
     numbers -- k, gamma, sigma0 -- that describe how similarity
     decays and noise grows with distance, specific to this camera
     and this person's captured data.
         |
         v
 [3] LIVE DETECTION
     For any new frame (from an uploaded photo, or a live webcam
     feed): detect faces, extract embeddings, estimate each face's
     distance from its pixel width (pinhole geometry), and compute
     a distance-corrected similarity against the enrolled gallery.
         |
         v
 [4] SNR GATE
     Combine the corrected similarity with a distance-dependent
     noise estimate to compute an SNR value. Only ACCEPT the match
     if both similarity AND SNR clear their thresholds.
         |
         v
 [5] REPORTING
     Every detection (from a single photo test or a continuous
     live stream) is logged with a timestamp, so a persistent
     record and summary report can be produced.
```

Each of these stages is implemented as a real, runnable piece of
this codebase: `enroll_capture.py` (stage 1), `calibrate.py` (stage
2), and `api_server.py` (stages 3, 4, and 5 together, since they all
happen per-request in the same code path). `utils.py` provides shared
helper functions used across all of them.

---

## 3. Face Detection and Embedding Extraction

### 3.1 The underlying model

This project uses **InsightFace's `buffalo_l` model bundle**, which
internally is a set of five separate ONNX neural network models
working together:

1. **`det_10g`** — a face *detector*. Given a full camera frame, it
   outputs zero or more bounding boxes, each one a rectangle
   `(x1, y1, x2, y2)` in pixel coordinates describing where a face
   was found, along with a confidence score.
2. **`2d106det`** — 2D facial landmark detection (106 points: eyes,
   nose, mouth contour, jawline, etc.), used to align the face crop
   before recognition.
3. **`1k3d68`** — 3D facial landmarks (68 points), used for pose
   estimation.
4. **`genderage`** — estimates apparent gender and age (not used by
   this project's matching logic, but part of the bundle).
5. **`w600k_r50`** — the actual *recognition* network. This is a
   ResNet-50-based architecture trained with an **ArcFace loss
   function**, and it is the model that actually produces the
   512-dimensional embedding vector used for identity matching.

### 3.2 What ArcFace optimizes for

ArcFace (Additive Angular Margin Loss) is a training objective
specifically designed to make embeddings of the *same* person cluster
tightly together in angular (directional) space, while pushing
embeddings of *different* people apart by an explicit angular margin.
Without going into the full training-time derivation (this project
uses a pretrained model and does not retrain the recognition network
itself), the practical consequence is: **the angle between two
embedding vectors is a meaningful measure of identity similarity**,
and specifically, **cosine similarity** between two ArcFace
embeddings is the standard, correct way to compare them.

### 3.3 Cosine similarity, precisely

Given two embedding vectors `a` and `b`, each a 512-dimensional
vector of real numbers, cosine similarity is defined as:

```
cosine_sim(a, b) = (a . b) / (||a|| * ||b||)
```

where `a . b` is the dot product (element-wise multiply, then sum
all 512 products), and `||a||` and `||b||` are the Euclidean norms
(lengths) of each vector:

```
||a|| = sqrt(a_1^2 + a_2^2 + ... + a_512^2)
```

This produces a number between -1 and 1:

- **1.0** means the two vectors point in exactly the same direction
  (identical identity, as far as the model can tell).
- **0.0** means the vectors are orthogonal (unrelated).
- **-1.0** means they point in exactly opposite directions (this is
  rare in practice for face embeddings, but can happen for
  spurious/false detections).

In this codebase, this is implemented directly in `utils.py`:

```python
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
```

The `+ 1e-8` is a small numerical safety term to avoid division by
zero in the pathological case where either vector has zero length
(which should not happen for a real embedding, but guards against a
crash if it ever did).

### 3.4 Embedding normalization

InsightFace's `.normed_embedding` property (used throughout this
codebase, e.g. `face.normed_embedding` in `api_server.py`) already
returns an L2-normalized embedding — meaning `||a|| = 1.0` for every
embedding coming out of the model. This matters for two reasons:

1. It means the cosine similarity formula above simplifies to just
   the dot product for two already-normalized vectors, since
   dividing by 1 does nothing — though the code still computes the
   full formula defensively, since some downstream steps (the
   distance correction described in Section 5) modify the embedding
   and can change its length.
2. It ensures that similarity comparisons are about *direction*
   (identity) and not *magnitude* (which could otherwise be
   influenced by irrelevant factors like overall image brightness).

---

## 4. The Pinhole Camera Model: Estimating Distance from Face Width

### 4.1 Why we need a distance estimate at all

Everything in Sections 5 through 7 depends on knowing, or at least
estimating, how far away a detected face is from the camera. This
system has no depth sensor, no stereo camera, no LiDAR — only a
single 2D RGB image. So distance has to be *inferred* from the image
itself.

### 4.2 The physical principle

A pinhole camera (the simplified physical model that describes how
essentially all conventional cameras behave, ignoring lens
distortion) projects a 3D object onto a 2D image plane such that the
size of the object's projection is **inversely proportional** to its
distance from the camera. Formally, if an object has a real-world
width `W` (e.g., the width of someone's face, roughly constant for a
given person), and it is at distance `D` from the camera, its
apparent width in the image, in pixels, is:

```
w_pixels = (f * W) / D
```

where `f` is the camera's focal length (in consistent units,
converted to pixels via the sensor's pixel pitch). The exact value of
`f` and `W` individually doesn't matter for this system — what
matters is that their *product*, `f * W`, is a constant for a given
camera and a given person's face. This means we can eliminate the
need to know `f` or `W` separately by using a **known reference
point**.

### 4.3 Deriving the working formula

Suppose at enrollment, a person's face was measured to be
`ref_width_px` pixels wide, at a known distance `ref_depth_m`. Since
`f * W` is constant for that person and that camera:

```
f * W = ref_width_px * ref_depth_m
```

Later, at some unknown live distance `depth_est`, the same face
appears with pixel width `live_width_px`:

```
f * W = live_width_px * depth_est
```

Since the left-hand sides are equal (same camera, same person's face
width):

```
ref_width_px * ref_depth_m = live_width_px * depth_est
```

Solving for `depth_est`:

```
depth_est = ref_depth_m * (ref_width_px / live_width_px)
```

This is exactly the formula implemented in `utils.py`:

```python
def estimate_depth_from_facewidth(live_width_px, ref_width_px, ref_depth_m):
    if live_width_px is None or live_width_px <= 0:
        return None
    return ref_depth_m * (ref_width_px / live_width_px)
```

### 4.4 Worked numeric example

Suppose during enrollment, a person's face was measured at
`ref_width_px = 220` pixels wide, at a known reference distance of
`ref_depth_m = 0.5` meters (they stood half a meter from the camera
and this was recorded).

Later, in a live frame, the same person's detected face bounding box
has width `live_width_px = 90` pixels (they've moved farther away).
Applying the formula:

```
depth_est = 0.5 * (220 / 90)
          = 0.5 * 2.444...
          = 1.222 meters
```

So the system estimates this person is now standing approximately
1.22 meters from the camera — without any depth sensor, purely from
how much smaller their face appears compared to the known reference.

### 4.5 Sources of error in this estimate

This estimate is not perfectly precise, and it's worth being honest
about why:

- **Face width varies between individuals.** Different people have
  genuinely different face widths, but this system estimates depth
  **per enrolled person**, using that specific person's own
  reference width — so this source of error is largely canceled out,
  as long as the same person's face width doesn't change
  meaningfully between enrollment and live detection (which is
  approximately true for a given adult, barring extreme cases like
  significant weight change).
- **Bounding box jitter.** The face detector's bounding box is not
  perfectly pixel-precise frame to frame; a few pixels of jitter in
  a live video feed translate to small fluctuations in the estimated
  distance.
- **Head pose.** A face turned significantly to the side presents a
  narrower apparent width than the same face viewed frontally, at
  the same true distance — this system captures 4 reference views
  (front/left/right/top) specifically to give the matching logic a
  reference for multiple poses, though the distance estimate itself
  currently always uses whichever view produced the best-matching
  embedding, not necessarily the geometrically "correct" one for
  that head angle.
- **Lens distortion at the edges of the frame.** The pinhole model
  assumes no distortion; real lenses, especially wide-angle ones,
  introduce some barrel or pincushion distortion that is most
  pronounced away from the image center. This is not currently
  corrected for.

None of these make the estimate useless — it is precise enough to
usefully distinguish "close" from "far" and to feed into the decay
and noise models below — but it is an *estimate*, not a laser
rangefinder measurement, and the system's design (the SNR gate in
particular) is built with this uncertainty in mind rather than
pretending the distance number is exact.

---

## 5. The Alpha Decay Function: Modeling Similarity Loss with Distance

### 5.1 The physical intuition

As established in Section 1, image quality on a face genuinely
degrades with distance — fewer pixels, more relative noise, less
fine detail for the recognition network to use. The consequence is
that even for the *same* person, the cosine similarity between a
live embedding captured at some distance and a clean, close-range
reference embedding will tend to be lower than 1.0, and will tend to
decrease further as distance increases — **not because the person is
different, but because the measurement itself has degraded.**

If we don't correct for this, a genuine positive match at long range
might be wrongly rejected simply because its raw similarity dropped
below a fixed threshold, even though the underlying identity is
correct.

### 5.2 Choosing an exponential decay model

This project models this expected decay using a single-parameter
exponential decay function:

```
alpha(d) = exp(-k * d)
```

where:

- `d` is the **relative depth**: how much farther than the person's
  own enrollment reference distance they currently are, i.e.
  `d = max(depth_est - ref_depth_m, 0)`. It's clipped to be
  non-negative because a person standing *closer* than their
  enrollment distance shouldn't be treated as having "negative
  decay" — the model only corrects for degradation beyond the clean
  reference condition.
- `k` is a **decay rate constant**, fit from real calibration data
  (see Section 8) — larger `k` means similarity is expected to drop
  off faster with each additional meter of distance.

Exponential decay was chosen (rather than, say, a linear or
polynomial decay) because it is the standard functional form for
processes where the *rate* of degradation is proportional to the
current signal strength — a very common and physically reasonable
assumption for signal-quality degradation processes, and it has the
convenient properties of always staying positive and asymptotically
approaching (but never reaching) zero as distance grows, which
matches the intuitive expectation that similarity should degrade
smoothly and never becomes negative purely from a distance effect.

### 5.3 Using alpha to correct the live embedding

The raw live embedding was captured under degraded conditions, so
its magnitude/reliability is expected to have shrunk by a factor of
`alpha(d)`. To compensate, before comparing it against the clean
reference embedding, this system divides the live embedding by
`alpha(d)`:

```
corrected_embedding = live_embedding / alpha(d)
```

This is implemented directly in `api_server.py`'s `match_face()`
function:

```python
a = alpha(relative_depth, k)
corrected = live_emb / max(a, 1e-6)
```

(The `max(a, 1e-6)` guards against division by an extremely small
number at very large distances, which would otherwise blow up the
corrected embedding's magnitude arbitrarily.)

The effect: at zero relative depth (the person is at exactly their
enrollment distance), `alpha(0) = exp(0) = 1`, so the correction does
nothing — as expected, since no correction is needed under the clean
reference condition. As relative depth grows, `alpha(d)` shrinks
toward zero, and the correction proportionally amplifies the live
embedding before comparison, compensating for the expected
degradation.

### 5.4 Worked numeric example

Continuing the distance estimate from Section 4.4 (`depth_est =
1.222m`, `ref_depth_m = 0.5m`), and using a real `k` value this
project actually measured during calibration (`near_k = 0.41436`,
from an earlier calibration run in this project's development):

```
relative_depth = depth_est - ref_depth_m
               = 1.222 - 0.5
               = 0.722 meters

alpha(0.722) = exp(-0.41436 * 0.722)
             = exp(-0.29917)
             = 0.7414
```

Interpretation: at this distance, the model expects the raw
embedding's usable signal to have shrunk to about **74.1%** of what
it would be at the clean reference distance. The correction step then
divides the live embedding by 0.7414, effectively amplifying it by
about 35% (`1 / 0.7414 ≈ 1.349`) before the cosine similarity
comparison — compensating for the expected degradation.

This is precisely why a face at 1.2 meters can still score a
reasonable similarity against a 0.5-meter reference photo in this
system, whereas an uncorrected raw comparison might have scored
noticeably lower purely due to the distance effect, potentially
causing a false rejection of a genuine match.

### 5.5 What alpha does NOT do

It is worth being explicit about a limitation here: `alpha(d)` is a
**population-level average correction**, fit from calibration data,
not a per-frame, per-pixel measurement of exactly how much this
*specific* frame degraded. Two frames at the same estimated distance
could have genuinely different real image quality (one slightly
blurred by motion, one perfectly still), and `alpha(d)` applies the
same correction to both, because it only knows the estimated
distance, not the actual measured blur or noise level of that
specific frame. This is a deliberate simplification — measuring
true per-frame image quality would require additional signal
processing (blur detection, local contrast metrics, etc.) that this
version of the system does not implement, and is noted as a
potential future improvement in Section 17.

---

## 6. The Sigma Noise Function: Modeling Uncertainty Growth with Distance

### 6.1 Why a correction factor alone isn't enough

Section 5's `alpha(d)` correction adjusts the *expected value* of
the degraded similarity back toward what a clean comparison would
show. But a correction based on an *average* expected degradation
does not account for how much that degradation *varies* from
instance to instance. At longer distances, not only does the average
signal shrink, but the **spread** — the noisiness, the unpredictable
frame-to-frame variation — also tends to grow. A distant, blurry,
low-resolution face crop is not just "shrunk," it is also
**noisier and less consistent** than a close, sharp one.

This matters because a similarity score corrected by `alpha(d)`
alone doesn't tell you *how much to trust that corrected number*. A
corrected similarity of 0.75 from a clean 1-meter shot means
something very different from a corrected similarity of 0.75 derived
from a noisy, heavily-corrected 20-meter shot — even if the two
numbers are numerically identical after correction, their underlying
reliability is not.

### 6.2 The noise growth model

This project models the *uncertainty* (standard deviation) of the
comparison as growing linearly with relative depth:

```
sigma(d) = sigma0 * (1 + gamma * d)
```

where:

- `sigma0` is the **baseline noise** — how much uncertainty exists
  even at zero relative depth (i.e., even under the clean reference
  condition, there is always *some* baseline measurement noise: minor
  lighting variation, small pose changes, compression artifacts,
  etc.).
- `gamma` is the **noise growth rate** — how much additional
  uncertainty is added per meter of relative depth beyond the
  baseline. This is also fit from real calibration data (Section 8).

A **linear** growth model was chosen here (rather than, say,
exponential, which was used for the decay function) because there is
no strong physical reason to expect noise variance to compound
multiplicatively the way signal decay does — a simple, interpretable
linear growth in uncertainty per additional meter is a reasonable
and parsimonious starting assumption, consistent with how measurement
uncertainty is often modeled in signal processing and physical
measurement contexts more generally.

This is implemented in `api_server.py`:

```python
def sigma(d, sigma0, gamma):
    return sigma0 * (1 + gamma * d)
```

### 6.3 Worked numeric example

Continuing the running example (`relative_depth = 0.722` meters),
using calibrated values `sigma0 = 0.0998` and `gamma = 0.6270` (these
are real values this project measured after recalibrating with a
denser, more realistic spread of distance points — see Section 8.5
for the full story of how these numbers changed over the course of
this project's development):

```
sigma(0.722) = 0.0998 * (1 + 0.6270 * 0.722)
             = 0.0998 * (1 + 0.4527)
             = 0.0998 * 1.4527
             = 0.1450
```

So at this distance, the model expects a standard deviation of
roughly 0.145 in the comparison — meaningfully larger than the
baseline `sigma0 = 0.0998` at zero relative depth, reflecting the
genuinely higher noise expected at this distance.

### 6.4 Why gamma being zero (for a long time, in this project's actual development) was a real problem

It is worth documenting an honest, real issue that occurred during
this project's actual development, because it illustrates an
important statistical point about calibration data requirements.

For an extended period, this project's calibration runs consistently
produced `gamma = 0.0000` exactly, no matter how the fit was
attempted. The reason turned out to be simple and instructive: with
only a **single real captured reference distance** (0.5 meters) and
otherwise widely-spaced synthetic depths, there was not enough
*variety* in the real-world noise pattern at different distances for
a fitting procedure to distinguish "noise grows with distance" from
"there just happens to be some noise, unrelated to distance." A
noise-growth parameter cannot be meaningfully estimated from a single
distance point — fitting *any* trend, growth or otherwise, requires
observations across a *range* of the independent variable.

This was resolved once the project moved to (a) capturing real data
at **two** real distances (0.5m and 1.5m) instead of one, and (b)
regenerating the synthetic far-range data with a denser, more
realistic spread of distances (5m through 35m in 5-meter steps,
replacing an earlier unrealistic and sparse 30/60/100/150m spread —
see Sections 9 and 10 for the full reasoning behind this change).
After this change, `gamma` fit to a genuinely non-zero value
(`0.6270`) for the first time in this project's development,
confirming that the noise-growth effect is real and detectable once
there is enough data spread to detect it.

---

## 7. The Signal-to-Noise Ratio Gate: Combining Everything into a Decision

### 7.1 Why not just threshold on corrected similarity alone

At this point, one might ask: why not just accept a match whenever
the `alpha`-corrected similarity clears some fixed threshold, since
that correction already accounts for distance? The answer is that
the correction in Section 5 adjusts the *expected value*, but does
nothing to communicate *how confident* the system should be in that
corrected value. A correction can push a noisy, unreliable, distant
measurement's similarity number up to look numerically similar to a
clean, close, reliable one — but the two measurements are not equally
trustworthy, even after correction. Blindly thresholding on the
corrected number alone would throw away exactly the information that
matters most: *how much noise was involved in producing this number
in the first place.*

### 7.2 The SNR formula

Signal-to-Noise Ratio is a standard concept in signal processing:
the ratio of the strength of a genuine signal to the strength of the
noise corrupting it. Here, "signal" is represented by the *expected*
squared magnitude of a correctly-corrected reference-quality
embedding component (`alpha(d)^2 * ||ref_embedding||^2`), and "noise"
is represented by the *variance* of the measurement uncertainty at
this distance (`sigma(d)^2`):

```
SNR(d) = alpha(d)^2 * ||ref_embedding||^2 / sigma(d)^2
```

This is implemented in `api_server.py`:

```python
def snr(d, k, sigma0, gamma, ref_norm_sq):
    a = alpha(d, k)
    s = sigma(d, sigma0, gamma)
    return (a ** 2) * ref_norm_sq / (s ** 2 + 1e-8)
```

Since embeddings from this model are L2-normalized (Section 3.4),
`||ref_embedding||^2 = 1.0` for a genuine reference embedding, which
simplifies the practical calculation, though the code computes it
explicitly (`ref_norm_sq = float(np.dot(ref_emb, ref_emb))`) rather
than hard-coding 1.0, both for clarity and as a defensive measure in
case a non-normalized embedding is ever passed through the pipeline
for some reason.

Squaring both `alpha(d)` and `sigma(d)` follows the standard
convention for SNR in power/energy terms (as opposed to amplitude
terms) — this is the same convention used broadly in signal
processing and electrical engineering, where SNR is typically
expressed as a ratio of *powers* (proportional to the square of
signal/noise amplitudes), which is also mathematically convenient
here since it keeps the SNR value always non-negative and makes its
scale behavior predictable as either the signal shrinks or the noise
grows.

### 7.3 Worked example — an ACCEPT case

Continuing the running example: `alpha(0.722) = 0.7414`,
`sigma(0.722) = 0.1450`, `||ref_embedding||^2 = 1.0` (normalized
reference):

```
SNR = (0.7414)^2 * 1.0 / (0.1450)^2
    = 0.5497 / 0.02103
    = 26.14
```

Compared against this project's chosen threshold,
`SNR_ACCEPT_THRESHOLD = 5.0`: since `26.14 >= 5.0`, this detection
passes the SNR side of the gate. (It still separately needs the
corrected cosine similarity to also clear
`DEFAULT_SIM_ACCEPT_THRESHOLD = 0.5` — SNR alone is a *necessary*
condition for acceptance, not a *sufficient* one on its own; both
gates must pass.)

### 7.4 Worked example — a REJECT case

Now consider the same enrolled person, but at a much greater
distance — say their detected face bounding box is only 15 pixels
wide (barely visible in the frame), against the same
`ref_width_px = 220` at `ref_depth_m = 0.5`:

```
depth_est = 0.5 * (220 / 15) = 7.333 meters
relative_depth = 7.333 - 0.5 = 6.833 meters

alpha(6.833) = exp(-0.41436 * 6.833) = exp(-2.8317) = 0.0589

sigma(6.833) = 0.0998 * (1 + 0.6270 * 6.833)
             = 0.0998 * (1 + 4.284)
             = 0.0998 * 5.284
             = 0.5273

SNR = (0.0589)^2 * 1.0 / (0.5273)^2
    = 0.00347 / 0.2780
    = 0.01248
```

Since `0.01248 < 5.0`, this detection is **REJECTed** on the SNR
side alone, decisively — the SNR at this distance is nowhere close to
the threshold. This is exactly the intended "blind spot" behavior:
past some distance, the SNR structurally can never clear the
threshold, no matter what the raw similarity number happens to be,
and the system correctly reports "unknown / REJECT" rather than
risking a false positive from a measurement that's too degraded to
trust.

### 7.5 The final decision rule

Putting it together, the exact acceptance rule implemented in
`api_server.py`'s `detect()` route is:

```python
accept = (best["snr"] is not None
          and best["snr"] >= SNR_ACCEPT_THRESHOLD
          and best["sim"] >= sim_accept_threshold)
```

A detection is only labeled `ACCEPT` — and only then is the person's
real name returned instead of `"unknown"` — if **both** the SNR gate
and the similarity gate pass. This dual-gate design means a
low-noise, high-similarity match passes easily; a high-similarity but
very-high-noise (i.e., very distant, or otherwise degraded) match is
still correctly rejected even if the raw number looks superficially
convincing; and a low-similarity match is rejected regardless of
distance, since a wrong identity shouldn't be accepted just because
the system happens to trust the *measurement* at that distance.

---

## 8. Calibration: Fitting k, gamma, and sigma0 from Real Data

### 8.1 What calibration actually does

Sections 5, 6, and 7 describe formulas that depend on three unknown
constants: `k` (decay rate), `sigma0` (baseline noise), and `gamma`
(noise growth rate). These are not universal constants — they depend
on the specific camera, lens, lighting conditions, and even the
specific person's face, and must be **fit from real captured data**
for each deployment. This is the job of `calibrate.py`.

### 8.2 The calibration procedure, step by step

1. **Collect embeddings across all captured distances.** For every
   enrolled person, for every captured depth folder (both real
   captures and synthetically generated ones — see Section 9), for
   every view (front/left/right/top), extract the ArcFace embedding
   using the same detection+recognition pipeline used at live
   inference time.

2. **Establish each person's reference embedding.** The embedding
   from their *smallest real captured depth* (their closest,
   cleanest photo) for each view becomes that person's reference for
   that view — this is exactly what ends up stored in `gallery.npz`
   and used for live matching.

3. **Compute a similarity table.** For every other captured depth,
   compute the raw cosine similarity between that depth's embedding
   and the person's reference embedding (same view). This produces a
   table of (distance, raw similarity) pairs.

4. **Fit the decay curve.** Given the (distance, similarity) pairs,
   fit the exponential decay model `similarity ≈ alpha(d) = exp(-k *
   d)` using nonlinear least-squares curve fitting (this project
   uses `scipy.optimize.curve_fit` internally) to find the single
   best-fitting value of `k`.

5. **Fit the noise growth curve.** Using the *residuals* — how far
   each actual similarity value deviates from what the fitted decay
   curve predicts — as an empirical estimate of noise at each
   distance, fit the linear growth model `sigma(d) = sigma0 * (1 +
   gamma * d)` to those residuals, again via least-squares fitting,
   to find `sigma0` and `gamma`.

6. **Produce both a "global" and a "near-range" fit.** The global fit
   uses all data, including the widest-range synthetic points, which
   is useful for understanding the overall decay shape but can be
   dominated by the extreme far-range points. The near-range fit
   restricts itself to real data under 10 meters, which this project
   treats as the more trustworthy, production-relevant fit — and
   `api_server.py` is designed to prefer the near-range values
   (`near_k`, `near_gamma`, `near_sigma0`) whenever they're available,
   falling back to the global fit only if insufficient near-range
   data exists.

### 8.3 Why nonlinear least-squares curve fitting

Given a set of observed `(d_i, similarity_i)` data points, curve
fitting seeks the parameter value `k` that minimizes the sum of
squared differences between the model's prediction and the actual
observations:

```
k_best = argmin_k  sum_i [ similarity_i - exp(-k * d_i) ]^2
```

This is a nonlinear least-squares problem (nonlinear because
`exp(-k * d)` is not linear in `k`), solved numerically via iterative
optimization (specifically, the Levenberg-Marquardt algorithm as
implemented in `scipy.optimize.curve_fit`), which repeatedly adjusts
a trial value of `k` to reduce the total squared error until it
converges to a locally optimal value. The `stderr` value reported
alongside each fitted parameter (e.g., `near_k = 0.41436, stderr
0.03888`, seen in an earlier calibration run in this project) is the
standard error of that parameter estimate — a measure of how
precisely the fitting procedure was able to pin down that value given
the available data; a smaller `stderr` relative to the fitted value
indicates a more confidently-determined parameter.

### 8.4 Why more, denser data points genuinely matter

A recurring theme in this project's real development (see Section
6.4) is that a fitting procedure's ability to detect a genuine trend
depends critically on having enough *spread* of the independent
variable (distance, in this case) in the data. Fitting two free
parameters (`sigma0` and `gamma`) from data effectively concentrated
at only one or two distinct real distances is statistically
underdetermined — there simply isn't enough independent information
in the data to separate "baseline noise level" from "how fast noise
grows," and the optimizer will tend to collapse the growth term to
zero rather than confidently assign it a value it cannot actually
support from the data.

### 8.5 A real before/after comparison from this project

This project's own calibration numbers changed meaningfully as the
underlying data improved, and it's instructive to show the actual
progression:

**Before** (4 depth folders per person: 1 real distance at 0.5m, plus
3 unrealistic synthetic distances at 30m/60m/100m/150m — see Section
10 for why 150m specifically was identified as unrealistic and
removed):

```
near_k = 0.41436  (stderr 0.03888)
near_sigma0 = 0.0944, near_gamma = 0.0000
```

**After** (12 depth folders per person: 2 real distances at 0.5m and
1.5m, plus additional real captures at 1m/2m/3m, plus 7 realistic
synthetic distances spanning 5m to 35m in 5-meter steps):

```
near_k = 0.21335  (stderr 0.04925)
near_sigma0 = 0.0998, near_gamma = 0.6270
```

The `k` value dropped substantially (0.414 → 0.213). This is not a
regression — it's the fitting procedure correctly re-distributing
where the "explanation" for observed similarity loss comes from. In
the earlier fit, with `gamma` forced to zero, `k` (decay) had to
account for *all* of the observed similarity variation with distance
on its own. Once `gamma` (noise growth) was able to take on a
genuinely non-zero value and explain part of that variation itself,
`k` correspondingly shrank to reflect only the portion of the effect
that is truly attributable to pure signal decay, rather than to
growing measurement noise. This is a more physically accurate
decomposition of the same underlying phenomenon.

---

## 9. Synthetic Distance Simulation

### 9.1 The practical problem it solves

Physically capturing real photos of a person at every distance from
0.5 meters out to 30+ meters, in a small room or home office, is not
feasible — there simply isn't enough physical space to walk that far
away from a webcam indoors. `2_simulate_distance.py` (in the research
pipeline) addresses this by **synthetically generating** what a
camera *would* have captured at a longer distance, starting from a
single clean, close-range real photo.

### 9.2 The two-part degradation model

The simulation applies two physically-motivated transformations to
the source image:

**1. Resolution loss via downsample/upsample.** Per the pinhole
geometry established in Section 4, a face at distance `d_virtual`
should occupy `(d_real / d_virtual)` times fewer pixels across than
it does in the real source image at `d_real`. The simulation
approximates this by first shrinking the image by that ratio
(destroying fine detail, exactly mimicking what a lower-resolution
capture at that distance would actually contain), then enlarging it
back up to the original pixel dimensions (so the output image is
still a normal-sized JPG, but the detail that was lost during
shrinking cannot be recovered by the enlargement step — this is the
same principle used in established low-resolution face recognition
research benchmarks such as TinyFace and QMUL-SurvFace, which use
comparable downsample-based degradation to simulate long-range CCTV
conditions from higher-resolution source photos).

A minimum scale floor (`MIN_SCALE = 0.25`) prevents the simulation
from shrinking any single image below a quarter of its original
resolution, since below that point real face detectors (including
the `det_10g` model used in this project) reliably fail to find a
face at all — an unrealistically over-degraded synthetic image would
simply produce "no face detected" and contribute nothing useful to
calibration.

**2. Depth-dependent noise injection.** Gaussian (normally
distributed) pixel noise is added to the image, with standard
deviation growing with the target virtual distance, deliberately
following the *same functional form* the calibration model itself
assumes: `sigma(d) = sigma0 * (1 + gamma * d)`. This ensures the
synthetic data is internally consistent with the model it's meant to
help calibrate, rather than introducing an unrelated, arbitrary noise
pattern.

A mild additional Gaussian blur is applied at longer simulated
distances (beyond 12 meters), loosely approximating the softening
effect of atmospheric haze and subject motion blur that would be
present in a genuine long-range capture of a person walking.

### 9.3 Honest limitations of synthetic data

This project is explicit, in both code comments and this document,
that synthetic degradation is an **approximation**, not a perfect
substitute for real long-range captures. It does not reproduce every
real-world effect: true atmospheric haze, genuine motion blur from an
actually-walking subject (as opposed to a static source photo that's
merely blurred uniformly), video compression artifacts specific to a
real camera's encoder, or true lens distortion at extreme viewing
angles. Calibration results derived partly from synthetic data should
be understood as a reasonable proof-of-concept and a way to extend
the *shape* of the decay curve beyond what can be physically
captured indoors — not as a full substitute for eventually validating
against genuine long-range real-world captures.

One specific, concrete limitation worth documenting clearly (because
it was discovered during this project's actual testing, and is easy
to misinterpret if not understood): because the downsample/upsample
process resizes the degraded image **back to the original pixel
dimensions**, a synthetic "25-meter" image's face bounding box, when
run back through the live detection pipeline, still measures roughly
the same pixel width as the original close-range source photo. This
means the *pinhole distance estimate* (Section 4) applied to a
synthetic test image will report a distance close to the *original*
capture distance, not the *intended* synthetic distance — even though
the image's *quality* (blur, noise) genuinely reflects the intended
longer distance. This is not a bug in the live detection code; it's
an inherent property of how the synthetic images are constructed.
`calibrate.py` correctly sidesteps this by using each image's known
folder name (e.g. `depth_025m`) directly as its true distance during
curve fitting, rather than relying on a live pixel-width estimate —
but anyone testing `api_server.py`'s `/detect` endpoint directly
against a synthetic image should expect `distance_m` in the response
to reflect roughly the original real capture distance, not the
synthetic target distance, while `similarity` and `snr` will still
correctly reflect the intended degradation.

---

## 10. Grounding the Distance Range in Real-World Geometry

### 10.1 Why this mattered

Early in this project's development, the synthetic distance range
extended out to 150 meters, a value that had no grounding in any
real deployment scenario — it was simply an arbitrary round number
used as a stress-test value. Fitting a calibration curve to include
data at distances that could never realistically occur indoors risks
distorting the fit to accommodate a scenario that will never be
encountered in practice, at the expense of fit quality in the
distance range that actually matters.

### 10.2 Researching realistic Indian corporate space dimensions

To ground the maximum realistic distance in something concrete rather
than another arbitrary guess, this project researched typical Indian
corporate office space standards. Key findings:

- Large corporate office floor plates in India commonly range from
  roughly 2,300 to 2,600 square meters (25,000-28,000 square feet)
  per floor for an average large office building, and even major
  campus towers can have individual floor plates in the 4,100-4,600
  square meter range (for example, the World Trade Center Chennai's
  towers).
- However, raw floor plate area significantly overstates realistic
  camera line-of-sight distance, because real office floors are
  broken up by walls, cubicles, structural columns, and corridors —
  a camera essentially never has an unobstructed view across an
  entire floor plate.
- Commercial real estate design guidance for actual sightline/layout
  planning uses a much smaller and more relevant number: the
  "core-to-glass" dimension (the distance from a building's central
  structural core out to the exterior windows in a typical layout)
  is commonly cited as approximately 12.8 to 14.3 meters — this is
  the dimension that actually governs how far an unobstructed view
  across a typical office floor tends to extend in practice.

### 10.3 The worst-case room considered, and its actual geometry

To be conservative (i.e., to still support a genuinely large space
rather than only a typical office floor), this project considered a
worst-case single room with footprint 25 meters by 20 meters and
ceiling height 15 meters — dimensions large enough to represent a
grand entrance lobby or atrium of a major Indian corporate campus,
well beyond a normal working office floor (whose ceiling heights are
typically only 2.7 to 3.5 meters).

The maximum possible unobstructed line-of-sight distance within such
a room is the length of its diagonal. Using the Pythagorean theorem
in three dimensions:

```
diagonal = sqrt(L^2 + B^2 + H^2)
         = sqrt(25^2 + 20^2 + 15^2)
         = sqrt(625 + 400 + 225)
         = sqrt(1250)
         = 35.36 meters
```

If only the floor-level diagonal is considered (relevant since a
person walking is always at roughly the same height, so the
meaningful line-of-sight distance for face recognition purposes is
closer to the floor-only diagonal than the full 3D corner-to-corner
distance):

```
floor_diagonal = sqrt(L^2 + B^2)
               = sqrt(25^2 + 20^2)
               = sqrt(625 + 400)
               = sqrt(1025)
               = 32.02 meters
```

### 10.4 The resulting decision

Given this analysis, the project's synthetic distance range was
revised from the original, arbitrary `[30, 60, 100, 150]` meters to
a realistic, denser spread of `[5, 10, 15, 20, 25, 30, 35]` meters —
capped at 35 meters, comfortably covering even the computed 35.36
meter worst-case diagonal of an unusually large corporate atrium,
while providing genuinely useful, evenly-spaced data across the range
that could realistically be encountered indoors. As documented in
Section 8.5, this change directly enabled the noise-growth parameter
`gamma` to be meaningfully estimated for the first time, since the
denser real-world-grounded spread gave the fitting procedure enough
independent information to detect the effect.

---

## 11. GPU Acceleration: CUDA, DirectML, and CPU Fallback

### 11.1 Why hardware acceleration matters here

Face detection and embedding extraction (the `det_10g` and
`w600k_r50` models, primarily) are the computational bottleneck of
this entire pipeline — the actual similarity/SNR matching math
(Sections 5-7) is comparatively trivial in computational cost (well
under a millisecond even against a gallery of thousands of enrolled
people, since it's just vector dot products). Running the neural
network inference itself on a GPU, when available, can produce an
order-of-magnitude speedup over CPU-only execution.

### 11.2 The provider hierarchy implemented

`utils.py`'s `get_face_app()` function checks, at startup, which
ONNX Runtime **execution providers** are actually available on the
current machine, and selects the best one automatically:

```python
available = ort.get_available_providers()
if "CUDAExecutionProvider" in available:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
elif "DmlExecutionProvider" in available:
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
```

This preference order reflects a real, practical tradeoff discovered
during this project's development:

1. **CUDA** (NVIDIA's GPU compute platform) generally offers the
   best raw performance, but requires the *exact matching* CUDA
   Toolkit and cuDNN runtime libraries to be separately installed on
   the system — not merely having a CUDA-capable NVIDIA graphics
   driver. This project encountered this distinction directly: a
   machine's driver reported CUDA 12.9 *capability*, but the
   `onnxruntime-gpu` Python package (version 1.28.0) required CUDA
   13.0 runtime libraries specifically (confirmed via ONNX Runtime's
   own documentation, which states that GPU packages published from
   version 1.27 onward target CUDA 13.0 by default), which were not
   actually installed — leading to a `cublasLt64_13.dll` missing-file
   error and a silent fallback to CPU.

2. **DirectML** is a Microsoft-provided execution backend that works
   with essentially any DirectX12-capable GPU on Windows (NVIDIA,
   AMD, or Intel integrated graphics), without requiring a separate
   vendor-specific toolkit installation at all — it relies on the
   DirectX12 stack that's already part of Windows. This makes it
   dramatically simpler to get working reliably, at some cost in raw
   peak performance compared to a correctly-configured CUDA setup.

3. **CPU** always works as the final fallback, on any machine,
   requiring no special hardware or drivers at all — just
   meaningfully slower.

### 11.3 Real benchmark numbers from this project

To make an informed choice rather than assuming DirectML would
outperform CPU on a given machine, this project ran a controlled,
same-machine, same-image benchmark, discarding the first "warm-up"
call for each provider (since the very first inference call on any
new execution provider includes one-time initialization/shader
compilation cost that would otherwise unfairly bias the comparison),
then averaging 5 subsequent calls:

```
CPUExecutionProvider:   1508.2 ms/frame  (warmed up, average of 5)
DmlExecutionProvider:    133.4 ms/frame  (warmed up, average of 5)
```

This is an approximately **11.3x speedup** from DirectML over CPU on
the specific hardware tested (an NVIDIA GeForce GTX 1650, an
entry-level discrete GPU with 4GB of VRAM) — a substantial,
genuinely measured improvement, not an assumption.

### 11.4 The warm-up cost, and why it's handled explicitly

A critical practical detail this project encountered directly: the
*first* inference call made against a freshly-initialized DirectML
session incurs a large one-time cost — one measured instance in this
project's own testing took over 40 seconds for a single frame on
first use, due to one-time GPU shader compilation, before subsequent
calls dropped to the much faster steady-state numbers shown above.
If left unaddressed, this would mean the very first real user request
to hit a freshly-started server would experience an enormous,
confusing delay.

To address this, `api_server.py` performs an explicit **warm-up
step** at server startup, before it begins accepting real traffic:
it runs two throwaway inference calls (using a real sample image from
the dataset if one can be found, so that the recognition and landmark
models are exercised too, not just the detection model) and discards
the results. This deliberately absorbs the one-time initialization
cost during server startup — when nobody is actively waiting on a
response — rather than during the first real user-facing request.

### 11.5 The correct way to detect "is GPU actually in use"

A subtlety worth documenting: it is not sufficient to check whether a
GPU-capable package is merely *installed*, or whether the driver
*supports* a certain compute API — the system must check what
execution provider the model is **actually, currently running on**,
since provider negotiation can silently fall back to CPU for reasons
that have nothing to do with whether GPU support is nominally present
(missing runtime DLLs, version mismatches, insufficient VRAM, etc.).
This project implements this correctly via `get_active_provider()` in
`utils.py`, which queries the *live* ONNX Runtime session directly:

```python
def get_active_provider():
    app = get_face_app()
    try:
        return app.models["recognition"].session.get_providers()[0]
    except Exception:
        return "unknown"
```

This value is reported in both the `/health` endpoint and every
`/detect` response (`active_provider`, `compute_device`, `gpu_used`),
so the actual runtime state — not an assumption about what *should*
be running — is always directly observable.

---

## 12. System Architecture: The Production API

### 12.1 Overview

`api_server.py` implements the production-facing HTTP API using
Flask. Its design goal is to be a genuinely usable service, not just
a demonstration script: raw image bytes in, structured JSON out,
suitable for a real backend or mobile client to consume directly.

### 12.2 The `/detect` endpoint contract

**Request:** the raw binary bytes of an image (e.g., the literal
contents of a `.jpg` file) in the POST request body — explicitly
**not** a file path, and explicitly **not** base64-encoded text. This
mirrors how a real production image-processing pipeline typically
receives frames (e.g., directly from a camera's frame buffer, or
from an already-in-memory image), rather than requiring an
intermediate encoding/decoding step.

**Response:** a single JSON object. Every per-face attribute
(detected name, bounding box, distance, similarity, SNR, decision,
matched view) is returned as a **parallel list** — all lists the same
length, in the same order, so that index `i` in every single list
always describes the same detected face. This design choice makes it
trivial for any consuming code to iterate `for i in
range(num_faces_detected)` and pull every attribute of face `i`
consistently, without needing to look up matching fields across a
more deeply nested structure.

A genuinely important, deliberate design decision: **zero faces
detected in a valid image is treated as a normal, successful result**
(`success: true`, `num_faces_detected: 0`, all lists empty) — not an
error. Errors (`success: false`, with a human-readable `error`
string) are reserved for genuine failures: corrupt or undecodable
image bytes, an empty request body, or the server not being fully
initialized. This distinction matters for any real system consuming
this API, since conflating "nothing was found" with "something went
wrong" would force callers to write unnecessarily defensive,
ambiguous error-handling logic.

### 12.3 The mobile-friendly `/test` page

Since browsers cannot send a raw binary POST request simply by
navigating to a URL, and mobile devices are a primary intended client
for this system, `api_server.py` also serves a simple, self-contained
HTML+JavaScript page at `/test`. It presents a native file/camera
picker, reads the selected photo as raw bytes via the browser's
`fetch` API, POSTs it directly to `/detect`, and renders the parsed
JSON result as readable cards (name, ACCEPT/REJECT, distance,
similarity, SNR) — allowing a real end-to-end test directly from a
phone's browser, with no separate app or command-line tool required.

---

## 13. The Live Streaming Pipeline

### 13.1 Why streaming needs a different approach than single-frame detection

Running the full detection+recognition pipeline on every single frame
of a live video feed (typically 15-30 frames per second from a
webcam) would be computationally wasteful and, on slower hardware,
would fail to keep up with the incoming frame rate at all, since each
detection pass takes on the order of 100-400 milliseconds even with
GPU acceleration (Section 11.3). The live stream implementation
therefore runs full detection only **every Nth frame** (configurable,
defaulting to every 5th frame), while still displaying **every**
captured frame in the video feed — reusing the most recent detection
result's bounding boxes and labels to annotate the frames in between.
This keeps the displayed video feed visually smooth while keeping the
actual computational cost bounded to a fraction of the full frame
rate.

### 13.2 Architecture: a background thread plus MJPEG output

The live stream is implemented as a dedicated background thread
(`camera_worker()`) that continuously reads frames from the laptop's
webcam via OpenCV, runs detection periodically as described above,
draws bounding boxes and labels onto every frame (green for ACCEPT,
red for REJECT), and publishes the latest annotated frame, JPEG-
encoded, into a thread-safe shared variable protected by a lock.

A separate route, `/stream`, serves this as an **MJPEG** (Motion
JPEG) stream — a simple, widely-supported video streaming format
where consecutive JPEG images are sent one after another, separated
by a multipart boundary marker, at a regular interval (throttled to
roughly 10 frames per second over the network in this
implementation, independent of the underlying camera or detection
frame rate, to keep bandwidth usage reasonable). Critically, MJPEG is
natively supported by essentially every web browser simply via an
`<img>` tag pointing at the stream URL — no video codec, no special
plugin, and no custom mobile app is required to view it, which is why
this format was chosen over alternatives like WebRTC that would
require substantially more implementation complexity for a
comparable practical benefit in this context.

### 13.3 Start/stop lifecycle

The camera is not opened automatically when the server starts — it
is explicitly started via a `POST /stream/start` request and stopped
via `POST /stream/stop`, with a corresponding `/stream_page` mobile
page providing simple Start/Stop buttons and an embedded live view.
This design avoids holding the webcam open (and therefore
unavailable to any other application) for the server's entire
lifetime, only acquiring it when someone actually wants to view the
live feed.

---

## 14. Detection Logging and Reporting

### 14.1 Why every detection is logged

For any real deployment, a system that makes ACCEPT/REJECT decisions
but keeps no record of them is of limited practical use — there is no
way to audit what happened, verify the system's behavior over time,
or produce a summary of activity. `api_server.py` therefore logs
**every** detection event — whether it originated from a single
`/detect` API call or from the continuous live stream — to both an
in-memory rolling log (capped at 5,000 most recent events, to bound
memory usage) and a persistent CSV file on disk (`logs/session_log.
csv`), so the record survives a server restart.

### 14.2 What's recorded per event

Each logged event captures: a timestamp, the source (`api` for a
single `/detect` call, or `stream` for a live-stream-derived
detection), the detected employee name (or `unknown`), the decision
(`ACCEPT`/`REJECT`), the estimated distance, the corrected
similarity, the SNR value, and which enrolled view (front/left/
right/top) produced the best match.

### 14.3 The `/report` page

A mobile-friendly summary page renders: total events logged, counts
of accepted versus rejected detections, a per-person breakdown of
accepted detection counts, and a chronological list of the most
recent 30 events with their full detail — plus a link to download the
complete CSV log for further analysis in any spreadsheet tool. This
gives a genuinely useful, at-a-glance operational picture directly
from a phone, without needing to inspect server logs or write any
separate analysis tooling.

---

## 15. A Complete End-to-End Worked Example

To tie every preceding section together, here is one detection
traced completely through the full pipeline, from a raw image to a
final decision, using realistic numbers consistent with values
actually measured in this project.

**Setup:** Person "Abhishek" was enrolled with a reference photo at
`ref_depth_m = 0.5` meters, where their face measured `ref_width_px =
220` pixels wide. Calibration (using real + realistic synthetic
data, per Sections 8 and 10) produced `near_k = 0.21335`, `near_sigma0
= 0.0998`, `near_gamma = 0.6270`.

**Step 1 — a live frame arrives.** A photo is POSTed to `/detect`.
The image is decoded (Section 12.2), and the `det_10g` face detector
finds one face, with bounding box `[145, 251, 207, 351]` — a pixel
width of `207 - 145 = 62` pixels.

**Step 2 — embedding extraction.** The `w600k_r50` recognition model
produces a 512-dimensional, L2-normalized embedding for this face
crop (Section 3).

**Step 3 — distance estimation.** Using the pinhole model (Section
4):
```
depth_est = 0.5 * (220 / 62) = 1.774 meters
```

**Step 4 — relative depth and alpha correction (Section 5):**
```
relative_depth = 1.774 - 0.5 = 1.274 meters
alpha(1.274) = exp(-0.21335 * 1.274) = exp(-0.2718) = 0.7620
```
The live embedding is divided by 0.7620 before comparison, correcting
for the expected quality loss at this distance.

**Step 5 — noise estimation (Section 6):**
```
sigma(1.274) = 0.0998 * (1 + 0.6270 * 1.274)
             = 0.0998 * (1 + 0.7989)
             = 0.0998 * 1.7989
             = 0.1795
```

**Step 6 — cosine similarity.** After correction, suppose the
comparison against Abhishek's stored reference embedding yields a
corrected cosine similarity of `0.81`.

**Step 7 — SNR (Section 7):**
```
SNR = (0.7620)^2 * 1.0 / (0.1795)^2
    = 0.5806 / 0.03222
    = 18.02
```

**Step 8 — the decision.** `SNR = 18.02 >= 5.0` ✓, and `similarity =
0.81 >= 0.5` ✓ — both gates pass, so this detection is **ACCEPTed**,
and the API response reports `employee_detected: "Abhishek"`,
`decision: "ACCEPT"`, `distance_m: 1.774`, `similarity: 0.81`, `snr:
18.02`.

**Step 9 — logging.** This event is appended to the in-memory event
log and to `logs/session_log.csv` with a timestamp, ready to appear
in the next `/report` view.

This single trace demonstrates every mathematical component of the
system working together: geometric distance estimation, exponential
signal decay correction, linear noise growth modeling, and a
dual-threshold SNR-gated decision rule — arriving at one clear,
auditable, logged outcome.

---

## 16. Known Limitations

This project has been developed and tested honestly, and it's
important to document its real limitations rather than overstate its
current capabilities:

- **Single-person calibration.** As of this writing, calibration data
  comes primarily from a single enrolled real person (`person_01`).
  The fitted `k`/`gamma`/`sigma0` values, while genuinely improved
  through the process described in this document, have not yet been
  validated across a genuinely diverse set of individuals, lighting
  conditions, or camera hardware.
- **Synthetic data caveats.** As detailed in Section 9.3, synthetic
  long-range images approximate resolution loss and sensor noise, but
  do not reproduce every real physical long-range effect (true
  atmospheric haze, genuine subject motion blur, real video codec
  compression artifacts, extreme-angle lens distortion).
- **Per-frame, not per-pixel, quality correction.** The alpha
  correction (Section 5.5) applies a population-average correction
  based on estimated distance alone, not a true per-frame measured
  image-quality assessment.
- **Pinhole distance estimation assumptions.** As detailed in Section
  4.5, the distance estimate assumes a roughly constant face width
  per person and a frontal-enough pose; extreme head angles, unusual
  lighting causing detection bounding box inconsistency, or lens
  distortion near frame edges can all introduce error into the
  distance estimate specifically (though the SNR gate's conservative
  design means errors here tend to bias toward *more* caution, not
  less, since underestimating true distance would underestimate
  needed noise correction).
- **No formal fairness or bias evaluation.** This system has not been
  evaluated for differential accuracy across demographic groups, a
  well-documented concern for face recognition systems generally, and
  should not be treated as bias-free without such an evaluation.
- **Development-server deployment.** The current server runs on
  Flask's built-in development server, which is explicitly unsuitable
  for sustained production traffic (as Flask's own startup warning
  states); a production WSGI server (e.g., `waitress` on Windows) is
  recommended before any sustained real-world deployment.

---

## 17. Future Work

Several directions were identified during this project's development
as valuable next steps, deliberately scoped out of the current
implementation to keep it focused:

- **Multi-person, multi-session calibration.** Enrolling and
  calibrating against multiple real people, across multiple real
  capture sessions (different days, different lighting), would give
  the fitted decay/noise parameters substantially more statistical
  grounding than the current single-person, largely single-session
  dataset.
- **A trained distance-adaptive CNN backbone.** An earlier,
  parallel research direction in this project (not part of the
  current production pipeline) explored building a from-scratch
  neural network backbone with FiLM (Feature-wise Linear Modulation)
  conditioning, where the distance value itself is fed directly into
  the network's internal feature representations, rather than applied
  as a post-hoc correction to a fixed pretrained embedding. This
  backbone was fully implemented, including hand-derived forward and
  backward passes gradient-verified against numerical differentiation,
  but currently has untrained (random) weights, since it requires a
  substantial real-world training dataset per its own documentation
  before it could offer any real recognition capability.
- **Per-frame image quality metrics.** Incorporating an actual
  measured blur/sharpness or local noise estimate per frame, rather
  than relying solely on estimated distance as a proxy for expected
  quality, could make the alpha correction more precise for any given
  individual frame.
- **Metric-learning loss for the CNN backbone.** Should the CNN
  backbone above be trained on real data in the future, replacing its
  current classification-style loss function with a metric-learning
  objective (such as a triplet loss or an ArcFace-style margin loss,
  directly analogous to the loss function that trained the
  currently-used pretrained recognition model) would likely be a
  more appropriate objective for an identity-verification task.
- **Production-grade serving.** Migrating from Flask's development
  server to a proper production WSGI server, and potentially
  containerizing the deployment, for any sustained real-world use.

---

## 18. Formula Reference Sheet

For quick reference, every core formula used in this system,
collected in one place:

**Cosine similarity** (Section 3.3):
```
cosine_sim(a, b) = (a . b) / (||a|| * ||b||)
```

**Pinhole distance estimation** (Section 4.3):
```
depth_est = ref_depth_m * (ref_width_px / live_width_px)
```

**Relative depth**:
```
relative_depth = max(depth_est - ref_depth_m, 0)
```

**Alpha decay function** (Section 5.2):
```
alpha(d) = exp(-k * d)
```

**Embedding distance correction** (Section 5.3):
```
corrected_embedding = live_embedding / alpha(d)
```

**Sigma noise growth function** (Section 6.2):
```
sigma(d) = sigma0 * (1 + gamma * d)
```

**Signal-to-Noise Ratio** (Section 7.2):
```
SNR(d) = alpha(d)^2 * ||ref_embedding||^2 / sigma(d)^2
```

**Final decision rule** (Section 7.5):
```
ACCEPT  if  SNR(d) >= SNR_ACCEPT_THRESHOLD (5.0)
        AND corrected_similarity >= SIM_ACCEPT_THRESHOLD (0.5)
REJECT  otherwise
```

**Room diagonal (worst-case line-of-sight distance)** (Section
10.3):
```
diagonal = sqrt(L^2 + B^2 + H^2)          (full 3D)
floor_diagonal = sqrt(L^2 + B^2)           (floor-level only)
```

---

*This document was written to accompany the ADAR codebase and is
intended to be read alongside `README.md` (for setup instructions)
and the source code itself (`api_server.py`, `calibrate.py`,
`enroll_capture.py`, `utils.py`), which is the authoritative,
literal implementation of every formula described here.*

# RTSP Camera Guide — ADAR Live Stream

How to point ADAR's live detection stream at a real CCTV/IP camera
over RTSP, instead of your laptop's built-in webcam.

---

## 1. What RTSP actually is

RTSP (Real Time Streaming Protocol) is how most IP cameras and CCTV
systems expose their live video feed over a network. Instead of a
camera being plugged directly into your laptop like a USB webcam, an
RTSP camera sits on your network (WiFi or Ethernet) and streams video
to whoever connects to its RTSP URL — a link that looks like a
website address, but for video instead of a webpage:

```
rtsp://username:password@192.168.1.50:554/stream1
```

`api_server.py`'s live stream (`/stream/start`, `/stream_page`)
already supports this — you don't need any code changes, just the
correct URL for your specific camera.

---

## 2. Finding your camera's RTSP URL

Every camera brand builds its RTSP URL slightly differently. There is
no single universal format, so the fastest, most reliable way to get
the *exact* correct URL for your camera is:

1. **Check the camera's own admin/manual first.** Most IP cameras have
   a web-based admin panel (usually reached by typing the camera's IP
   address into a browser) with a "Network" or "RTSP" settings page
   that shows the exact URL to use — this is always more reliable than
   guessing.
2. **Check the physical label or box.** Some consumer CCTV cameras
   print the default RTSP path directly on a sticker or in the quick-
   start guide.
3. **If neither is available**, the table below covers the most common
   patterns by brand, to try as a starting point.

### Common RTSP URL patterns by brand

| Brand | Typical URL pattern |
|---|---|
| Hikvision | `rtsp://user:pass@<ip>:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@<ip>:554/cam/realmonitor?channel=1&subtype=0` |
| CP Plus | `rtsp://user:pass@<ip>:554/cam/realmonitor?channel=1&subtype=0` |
| Generic ONVIF camera | `rtsp://user:pass@<ip>:554/onvif1` |
| TP-Link Tapo | `rtsp://user:pass@<ip>:554/stream1` |
| Reolink | `rtsp://user:pass@<ip>:554/h264Preview_01_main` |
| Generic / unknown brand | `rtsp://user:pass@<ip>:554/stream1` or `/live` or `/live.sdp` |

**Notes on the pieces of the URL:**

- `user:pass` — your camera's login credentials (often `admin` and a
  password you set during setup, or a default printed on the camera —
  change this from any factory default before deploying on a real
  network).
- `<ip>` — the camera's local network IP address (e.g. `192.168.1.50`),
  found in your router's connected-devices list or the camera's own
  app/admin panel.
- `:554` — the standard RTSP port. Most cameras use 554 by default;
  some let you change it in settings.
- The path after the port (`/Streaming/Channels/101`, `/stream1`,
  etc.) varies the most between brands — this is the part most worth
  double-checking in your camera's own documentation if the connection
  fails.
- `subtype=0` (Dahua/CP Plus pattern) usually means the **main,
  full-resolution** stream; `subtype=1` is usually a lower-resolution
  **sub-stream**, often better for this use case (see Section 5).

---

## 3. Connecting ADAR to your camera

### Option A — from the phone/browser page (easiest)

1. Start the server as usual:
   ```
   python api_server.py --calibration_dir calibration_output --port 8000
   ```
2. On your phone or laptop browser, open:
   ```
   http://<your-laptop-ip>:8000/stream_page
   ```
3. In the text box (labeled "Leave blank for laptop webcam, or paste
   rtsp://..."), paste your camera's full RTSP URL.
4. Tap **Start**. The page will now show your CCTV camera's live feed
   with detection boxes drawn on it, exactly like it did with the
   laptop webcam.
5. Tap **Stop** when done, or leave blank and tap Start again later to
   go back to the laptop's own webcam.

### Option B — directly via curl / API call

```powershell
curl.exe -X POST -H "Content-Type: application/json" -d "{\"source\": \"rtsp://admin:yourpassword@192.168.1.50:554/stream1\"}" http://localhost:8000/stream/start
```

To stop it:
```powershell
curl.exe -X POST http://localhost:8000/stream/stop
```

---

## 4. Testing the RTSP URL independently first

Before troubleshooting inside ADAR, it's worth confirming the RTSP URL
itself actually works, completely separate from this project — that
way you know whether a connection problem is about the camera/URL, or
about ADAR specifically.

**Using VLC Media Player** (simplest way to check):
1. Open VLC → Media → Open Network Stream
2. Paste your RTSP URL exactly as you plan to use it
3. Click Play

If VLC can't connect either, the problem is the URL, credentials, or
network — not ADAR. If VLC connects successfully but ADAR doesn't,
paste the exact error from ADAR's terminal output and that'll narrow
it down further.

---

## 5. Practical tips

- **Same network required.** Your laptop (running `api_server.py`)
  and the camera must be reachable on the same network — same WiFi,
  or the camera's Ethernet segment routed to reach your laptop. A
  camera on a completely separate/isolated network (e.g. behind its
  own NVR with no route to your laptop) won't be reachable directly.
- **Prefer the sub-stream, not the main stream, if your camera offers
  one.** RTSP main streams are often full HD or higher, which can be
  more bandwidth and CPU than this pipeline needs for face detection
  at typical indoor ranges — a lower-resolution sub-stream
  (`subtype=1` in the Dahua/CP Plus pattern, or a separate "sub
  stream" URL on other brands) is usually plenty for face detection
  and reduces both network load and processing lag.
- **First connection can take a few seconds.** RTSP handshake and
  initial buffering is normal; if the feed doesn't appear within
  ~5-10 seconds of tapping Start, treat it as a real connection issue
  rather than assuming it's still loading.
- **Detection still runs every Nth frame**, exactly as it does with
  the laptop webcam (Section 13 of `ALGORITHM.md`) — an RTSP source
  doesn't change the detection logic at all, only where the video
  frames come from.

---

## 6. Troubleshooting

**"Could not open camera index..." / stream never starts**

- Double-check the RTSP URL works in VLC first (Section 4).
- Confirm the camera's IP hasn't changed (many routers assign IPs
  dynamically via DHCP — a static/reserved IP for the camera avoids
  this).
- Confirm username/password are current — some cameras lock out after
  repeated failed login attempts.

**Feed connects but is very laggy / choppy**

- Try the camera's lower-resolution sub-stream instead of the main
  stream (see Section 5).
- Check WiFi signal strength if the camera is wireless — RTSP is
  bandwidth-sensitive.

**Feed shows video but no face boxes ever appear**

- This is a detection issue, not a connection issue — check the
  camera's actual field of view and lighting; face detection still
  needs a reasonably lit, reasonably sized face in frame, regardless
  of camera source.

**Special characters in your password break the URL**

- Characters like `@`, `#`, `%`, or `/` inside a password can be
  misread as part of the URL structure. If your password contains any
  of these, either change it to something alphanumeric for this
  camera, or percent-encode the special characters (e.g. `@` becomes
  `%40`) within the credentials portion of the URL only.

---

## 7. Security note

RTSP URLs typically embed the camera's username and password in plain
text. Treat any RTSP URL you paste into `/stream_page` or send via
curl the same way you'd treat a plaintext password — avoid pasting it
into shared chat logs, screenshots, or committing it into a git
repository (this project's `.gitignore` does not currently need to
exclude anything RTSP-specific, since URLs are only ever passed at
request time and are not written to disk by this pipeline — but stay
mindful of copy/pasting them elsewhere).

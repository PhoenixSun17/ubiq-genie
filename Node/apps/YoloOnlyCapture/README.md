# YoloOnlyCapture

[English](README.md) | [简体中文](README.zh-CN.md)

Separate YOLO-only comparison for `DynamicCaptureYolo`. Existing `Yolo`, `DynamicCapture`, `DynamicCaptureYolo`, shared components, VRDC, and Unity files are unchanged. OpenCV still handles camera input, experimental lens correction, calibration, and drawing; YOLO alone determines the detected objects and their boxes. There is no foreground/background subtraction, contour detection, or foreground-to-YOLO IoU matching.

## Comparison behavior

| Property | DynamicCaptureYolo | YoloOnlyCapture |
| --- | --- | --- |
| Object count and geometry | Foreground contours | YOLO detection boxes |
| Category | YOLO matched to a contour | Category of the YOLO box |
| Missed/unrecognised objects | May remain as foreground `unknown` objects | Absent if YOLO does not detect them |
| Objects already present at startup | Suppressed by captured background | Reported if detected inside the calibrated area |
| Session IDs | Centroid matching, independent of label | Same matching policy |
| Static/Dynamic status | Pixel difference against previous frame | Box-centre displacement between consecutive processed frames |
| Camera | Index 1, DirectShow on Windows, requests 1280 x 720 | Same |
| Lens correction | Experimental medium `k1=-0.12` | Same |
| Model | Local YOLOv8n, confidence 0.5, inference size 640 | Same |
| Coordinates | Four-point floor mapping to 6 x 6 | Same mapping, using box centres and corners |
| Transport | Network ID 99, complete stdout records | Same |

Use the same four physical floor points and camera mode when comparing runs. Run one capture app at a time: both use camera 1 and send on network ID 99. A recorded video can be selected as `capture.source` for an approved repeatable comparison. Do not infer an accuracy improvement from changed counts alone; the two programs have different inclusion rules, and a YOLO box is not a physical footprint.

The original `apps/Yolo` backend uses camera 0, hardcoded full-frame calibration, per-frame IDs, and a smaller `{id,x,y,label}` payload. This new app preserves the hybrid app's information and calibration instead of modifying that older program.

## Calibration and controls

Keep the camera stationary. Click four floor points in the corrected preview, clockwise: top-left, top-right, bottom-right, bottom-left. Tracking and packet output start when the area is valid. Objects are included when their YOLO box centre is inside or on the selected polygon; the box itself is not clipped to that polygon.

Lens correction is uncalibrated. It crops and stretches peripheral content; the temporary 1920 x 1080 experiment retained about 80% of the source image area at this setting. Map construction uses the actual captured dimensions and caches the result. A frame-size change stops capture and requires restarting/calibrating again. The requested resolution stays at 1280 x 720; a different camera mode may have a different projection. Recognition improvement and metric accuracy have not been established.

- **q** exits and releases the camera.
- **b** clears the motion baseline while preserving session IDs and calibration. It does not capture or subtract a background. The next observation establishes a new motion baseline.

IDs match one-to-one within 50 image pixels and expire after eleven missed detection frames. Only currently detected objects are emitted. Label changes do not reset IDs. Crossings, large box movement, merged detections, long occlusion, and box jitter can change or swap IDs; these are session IDs, not permanent identities.

`status` is `Dynamic` when a matched object's box centre moves **more than** `motionThresholdPx` pixels since the immediately previous processed frame; otherwise it is `Static`. The default is 5 pixels. A new ID or a return after a missed frame starts as `Static` to establish a baseline; that initial status is not proof that the object is physically stationary. This simple estimate depends on processing rate and box jitter, and may miss slow movement or movement inside a stationary box. It deliberately differs from the hybrid app's pixel-difference status, as selected for this comparison.

## Configuration and environment

The app is located at `C:\ubiq-genie\Node\apps\YoloOnlyCapture`; its independent backend is `Node\services\yolo_only_capture\Capture.py`.

The supplied configuration explicitly uses `C:\ubiq-genie\Node\services\dynamic_capture_yolo\.venv\Scripts\python.exe` and the existing `apps/Yolo/yolov8n.pt`, keeping package and weight versions the same for comparison. No packages are installed or upgraded, no existing backend is imported, and no model is automatically downloaded. This app uses its own `services/yolo_only_capture/.runtime/yolo-config` settings directory. The wrapper disables Python user-site packages and bytecode output.

| Configuration key | Default | Meaning |
| --- | --- | --- |
| `pythonCommand` (top level) | Absolute path above | Existing isolated Python interpreter; no global fallback |
| `capture.source` | `1` | Camera index or local video path |
| `capture.modelPath` | `../Yolo/yolov8n.pt` | Existing local weights |
| `capture.logPath` | `yolo_only_tracking.csv` | This app's separate CSV, overwritten at the next run |
| `capture.confidence` | `0.5` | Minimum detection confidence |
| `capture.device` | `auto` | Device selection; existing CPU environment remains CPU |
| `capture.motionThresholdPx` | `5` | Nonnegative centroid-motion threshold per processed frame |

Model, CSV, and relative video paths resolve against the new app directory. To use a separate environment later, create `services/yolo_only_capture/.venv`, install this service's pinned `requirements.txt` there, then remove `pythonCommand` to select that environment or replace it with another absolute interpreter path. The lens setting is the backend constant `LENS_CORRECTION_K1`; `0` disables correction.

The room configuration matches the hybrid: joins the existing server at `localhost:8009`, room GUID `6765c52b-3ad6-4fb0-9030-2c9a05dc4731`. Start that server separately and use the same room on the receiver.

## Launch after testing approval

The new files have been reviewed as source only. Imports, compilation, model inference, offline execution checks, camera capture, and Ubiq delivery have not been run. Obtain the user's approval before performing those checks.

After approval, launch from the Node directory:

```powershell
Set-Location C:\ubiq-genie\Node
npm start YoloOnlyCapture
```

## Output

The outer network payload is `{ "data": "<complete stdout line>" }` on network ID **99**. Each Python stdout line retains the original list representation:

```text
[frame_count, object_count, '{"count": ..., "objects": [...]}']
```

Every object contains `id`, `pos: {x,y}`, `scale: {w,h}`, `status`, `label`, `class_id`, and `confidence`. Position maps the box centre to the 6 x 6 coordinate area, rounded to two decimals. Scale maps the upper-left and lower-right corners and takes their absolute coordinate differences, rounded to three decimals, matching the hybrid calculation. Confidence is the model detection score, not a calibrated probability of physical correctness.

An empty detection frame emits `count: 0` and `objects: []`; missed tracks are retained internally but are not emitted as stale objects. No `unknown` fallback object is invented when YOLO detects nothing.

CSV columns remain `frame count,object count,udpstring`. CSV and stdout are generated from the same packet. Diagnostics go to stderr; the Node app assembles split/combined UTF-8 stdout chunks into whole lines before delivery. UDP remains unused. No Unity receiver change is included.

Prediction API reference: [Ultralytics prediction interface](https://docs.ultralytics.com/modes/predict/).

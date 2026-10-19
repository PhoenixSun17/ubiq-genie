# DynamicCaptureYolo

[English](README.md) | [简体中文](README.zh-CN.md)

This separate application adds YOLO object categories to the original DynamicCapture foreground capture pipeline. It uses its own app, service, Python environment, and CSV file. The existing DynamicCapture, Yolo, shared components, and Unity code remain unchanged.

## Capture behavior

The default source is camera **1**, requested at **1280 x 720**. Keep the camera still. Click four floor points clockwise: top-left, top-right, bottom-right, bottom-left. The fourth click captures the reference background; the selected area maps to the original **6 x 6** Unity coordinates. Capture the background with the area clear of the objects you want to track.

Press **b** to capture a new background and **q** to quit. Recapturing the background does not immediately reset the session ID counter or stored tracks.

The Python backend applies the selected **medium lens correction (`k1 = -0.12`)** before calibration, background/motion detection, and YOLO. Select the four floor points using the **corrected preview**. This is an uncalibrated experimental correction: it crops and stretches peripheral content, retaining about 80% of the source image area in the 1920 x 1080 experiment; improved recognition and metric accuracy have not been established. Correction maps use the actual frame dimensions and are cached; a frame-size change requires restarting and recalibrating. The requested camera resolution remains unchanged, and a different camera mode may have a different projection. Set `LENS_CORRECTION_K1` to `0` in `services/dynamic_capture_yolo/Capture.py` to disable correction. The integrated change has not yet been tested.

Foreground detection preserves the original blurred grayscale background subtraction, threshold 25, two dilation iterations, contour area greater than 800 pixels, and centroid-inside-area requirement. Dynamic/Static status still comes from the difference against the previous frame, with the original motion-score threshold of 5. Position and scale come from the foreground contour and calibration, not the YOLO box.

YOLOv8n runs on clean frames and adds a category to a foreground contour only when their bounding boxes can be matched one-to-one with IoU at least 0.3. The default detection confidence is 0.5. An unmatched foreground object remains in the output as `label: "unknown"`, `class_id: null`, and `confidence: 0.0`. Background objects detected by YOLO do not independently create output objects. Split silhouettes, overlapping objects, and categories outside the model's vocabulary can therefore remain unknown.

IDs are session tracking IDs assigned by the existing centroid approach, independent of the label. A track can match within 50 pixels and is retained internally for ten missed frames, then removed on the eleventh. Only currently observed objects are emitted. IDs can change or swap after crossings, merged contours, fast movement, or long occlusion; they do not identify a person or permanently identify a physical object.

## Setup

These files belong under `C:\ubiq-genie\Node`. The application reuses the existing local model at `Node\apps\Yolo\yolov8n.pt`. It checks that the file exists and does not request a model download.

Create and install into the separate service environment with the following PowerShell commands. Installation is separate from testing; these commands do not start the capture application. The requirements pin the existing machine's package versions, including Ultralytics 8.4.19 and PyTorch 2.10.0. The initial Windows setup uses CPU PyTorch; `device: "auto"` consequently selects CPU. No existing Python environment is upgraded.

```powershell
Set-Location C:\ubiq-genie\Node\services\dynamic_capture_yolo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The default interpreter is `Node\services\dynamic_capture_yolo\.venv\Scripts\python.exe` on Windows, or `.venv/bin/python` on other systems. An optional top-level `pythonCommand` in this app's `config.json` can supply a different absolute interpreter path. A missing environment produces an actionable error rather than falling back to a global interpreter. Ultralytics settings are kept in the service's `.venv/yolo-config` directory, and Python user-site packages are disabled for the child process.

Do not run tests, compile/import checks, model inference, camera capture, or the application until the user has separately approved that testing. This copy was prepared without running those checks; runtime compatibility and camera behavior remain unverified.

## Configuration and approved launch

The copied room configuration joins the existing server at `localhost:8009` using room GUID `6765c52b-3ad6-4fb0-9030-2c9a05dc4731`. The server and client must already use the same room. No Unity receiver or scene is changed. Use this app in place of the original capture app when evaluating it, since both publish on network ID 99.

The new app's `config.json` contains:

| Key under `capture` | Default | Meaning |
| --- | --- | --- |
| `source` | `1` | Camera index; a video file path may also be supplied. |
| `modelPath` | `../Yolo/yolov8n.pt` | Existing local YOLO weights. |
| `logPath` | `dynamiccapture_yolo_tracking.csv` | Separate output log. |
| `confidence` | `0.5` | Minimum YOLO detection confidence. |
| `matchIou` | `0.3` | Minimum overlap for a foreground-to-YOLO match. |
| `device` | `auto` | Automatic device choice; `cpu` forces CPU inference. |

Model and log paths resolve relative to this app's directory, regardless of the terminal's initial directory. Relative video source paths also use the app directory because that is the child process's working directory. Each run writes the separate configured CSV using the original columns: `frame count`, `object count`, and `udpstring`.

**After explicit testing approval**, launch from the Node directory:

```powershell
Set-Location C:\ubiq-genie\Node
npm start DynamicCaptureYolo
```

## Output contract

Every completed capture record keeps the original Python stdout representation:

```text
[frame_count, object_count, '{"count": ..., "objects": [...]}']
```

The Node app assembles complete UTF-8 lines and forwards each record as `{ "data": "<original line>" }` on **NetworkId(99)**. The inner line is a Python list representation, not a JSON array; it is not parsed or rewritten by the new app. Diagnostics go to stderr. CSV packets retain the same JSON structure.

Each object retains `id` (integer), `pos` (`x`, `y` numbers), `scale` (`w`, `h` numbers), and `status` (`"Dynamic"` or `"Static"`). It adds `label` (string), `class_id` (integer or null), and `confidence` (number). The top-level packet remains `count` (integer) and `objects` (array). A frame without current foreground objects contains `count: 0` and an empty array.

Identity fields are delivered in the network payload and recorded in the new CSV. Displaying those fields in an existing Unity UI is not implemented by this copy.

"""YOLO-only detections with the DynamicCaptureYolo calibration and wire format.

YOLO boxes determine object count and geometry. Session tracking supplies IDs
and motion status; no background subtraction or foreground contours are used.
Stdout contains only [frame_count, count, json_string] records.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import distance as dist


SERVICE_DIR = Path(__file__).resolve().parent
NODE_DIR = SERVICE_DIR.parent.parent
DEFAULT_MODEL = NODE_DIR / "apps" / "Yolo" / "yolov8n.pt"
DEFAULT_LOG = NODE_DIR / "apps" / "YoloOnlyCapture" / "yolo_only_tracking.csv"
UNITY_AREA_WIDTH = 6
UNITY_AREA_HEIGHT = 6
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_DISAPPEARED = 10
MAX_DISTANCE = 50
MOTION_THRESHOLD_PX = 5.0
LENS_CORRECTION_K1 = -0.12


def diagnostic(message):
    print(message, file=sys.stderr, flush=True)


def create_lens_correction_maps(frame_shape):
    """Match the hybrid's experimental correction, not measured X5 intrinsics."""
    height, width = frame_shape[:2]
    focal = width / 2.0
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    if 1 + 3 * LENS_CORRECTION_K1 * (cx * cx + cy * cy) / (focal * focal) <= 0:
        raise ValueError("Lens correction is too strong for this frame's aspect ratio.")
    intrinsic = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    distortion = np.array([LENS_CORRECTION_K1, 0, 0, 0, 0], dtype=np.float64)
    return cv2.initUndistortRectifyMap(
        intrinsic, distortion, None, intrinsic, (width, height), cv2.CV_32FC1
    )


@dataclass(frozen=True)
class YoloDetection:
    x: float
    y: float
    w: float
    h: float
    class_id: int
    label: str
    confidence: float

    @property
    def centroid(self):
        return (self.x + self.w / 2, self.y + self.h / 2)


class CentroidTracker:
    """Same label-independent, one-to-one session ID policy as the hybrid."""

    def __init__(self, max_distance=MAX_DISTANCE, max_disappeared=MAX_DISAPPEARED):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_object_id = 0
        self.centroids = {}
        self.disappeared = {}

    def update(self, input_centroids):
        assignments = {}
        object_ids = sorted(self.centroids)
        used_detections = set()
        if object_ids and input_centroids:
            distances = dist.cdist(
                np.asarray([self.centroids[object_id] for object_id in object_ids]),
                np.asarray(input_centroids),
            )
            candidates = sorted(
                (float(distances[row, col]), object_id, col)
                for row, object_id in enumerate(object_ids)
                for col in range(len(input_centroids))
                if distances[row, col] <= self.max_distance
            )
            for _, object_id, detection_index in candidates:
                if object_id in assignments or detection_index in used_detections:
                    continue
                self.centroids[object_id] = input_centroids[detection_index]
                self.disappeared[object_id] = 0
                assignments[object_id] = detection_index
                used_detections.add(detection_index)
        for object_id in object_ids:
            if object_id not in assignments:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.centroids[object_id]
                    del self.disappeared[object_id]
        for detection_index, centroid in enumerate(input_centroids):
            if detection_index in used_detections:
                continue
            object_id = self.next_object_id
            self.next_object_id += 1
            self.centroids[object_id] = centroid
            self.disappeared[object_id] = 0
            assignments[object_id] = detection_index
        return assignments


def calculate_perspective_matrix(src_points, width, height):
    src = np.float32(src_points)
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    return cv2.getPerspectiveTransform(src, dst)


def map_pixel_to_unity(matrix, pixel_x, pixel_y):
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    return cv2.perspectiveTransform(point, matrix)[0][0]


def calculate_unity_scale(matrix, x, y, w, h):
    # Preserve the hybrid's two-corner box scale calculation for comparison.
    p1 = map_pixel_to_unity(matrix, x, y)
    p2 = map_pixel_to_unity(matrix, x + w, y + h)
    return round(float(abs(p2[0] - p1[0])), 3), round(float(abs(p2[1] - p1[1])), 3)


def load_model(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Local YOLO weights not found: {model_path}")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(SERVICE_DIR / ".runtime" / "yolo-config"))
    with redirect_stdout(sys.stderr):
        from ultralytics import YOLO

        return YOLO(str(model_path))


def recognize(model, clean_frame, confidence, device, calibration_points):
    with redirect_stdout(sys.stderr):
        results = model.predict(
            source=clean_frame, conf=confidence, imgsz=640,
            device=None if device == "auto" else device, verbose=False, save=False,
        )
    height, width = clean_frame.shape[:2]
    polygon = np.asarray(calibration_points, dtype=np.int32)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        for box, class_id, score in zip(boxes, classes, scores):
            if not all(math.isfinite(value) for value in (*box, class_id, score)):
                continue
            x1 = min(max(float(box[0]), 0.0), float(width))
            y1 = min(max(float(box[1]), 0.0), float(height))
            x2 = min(max(float(box[2]), 0.0), float(width))
            y2 = min(max(float(box[3]), 0.0), float(height))
            if x2 <= x1 or y2 <= y1 or score < confidence:
                continue
            detection = YoloDetection(
                x1, y1, x2 - x1, y2 - y1, int(class_id),
                str(result.names[int(class_id)]), float(score),
            )
            if cv2.pointPolygonTest(polygon, detection.centroid, False) >= 0:
                detections.append(detection)
    return detections


def build_packet(assignments, detections, matrix, previous_centroids, motion_threshold_px):
    objects = []
    for object_id, detection_index in sorted(assignments.items()):
        detection = detections[detection_index]
        cx, cy = detection.centroid
        previous = previous_centroids.get(object_id)
        # First sightings and returns after a missed frame establish a new baseline.
        is_dynamic = previous is not None and math.hypot(cx - previous[0], cy - previous[1]) > motion_threshold_px
        ux, uy = map_pixel_to_unity(matrix, cx, cy)
        uw, uh = calculate_unity_scale(matrix, detection.x, detection.y, detection.w, detection.h)
        objects.append({
            "id": int(object_id),
            "pos": {"x": round(float(ux), 2), "y": round(float(uy), 2)},
            "scale": {"w": uw, "h": uh},
            "status": "Dynamic" if is_dynamic else "Static",
            "label": detection.label,
            "class_id": detection.class_id,
            "confidence": detection.confidence,
        })
    return {"count": len(objects), "objects": objects}


def emit_packet(frame_count, packet, writer, csv_file):
    record = [frame_count, packet["count"], json.dumps(packet, allow_nan=False)]
    writer.writerow(record)
    csv_file.flush()
    print(record, flush=True)


def draw_detections(frame, assignments, detections, packet):
    for obj in packet["objects"]:
        detection = detections[assignments[obj["id"]]]
        color = (0, 0, 255) if obj["status"] == "Dynamic" else (0, 255, 0)
        cv2.rectangle(frame, (round(detection.x), round(detection.y)),
                      (round(detection.x + detection.w), round(detection.y + detection.h)), color, 2)
        label = f"ID {obj['id']}: {obj['label']} {obj['confidence']:.2f} {obj['status']}"
        cv2.putText(frame, label, (round(detection.x), max(15, round(detection.y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_tracking(video_source=1, model_path=DEFAULT_MODEL, log_path=DEFAULT_LOG,
                 confidence=0.5, device="auto", motion_threshold_px=MOTION_THRESHOLD_PX):
    model_path = Path(model_path).expanduser().resolve()
    log_path = Path(log_path).expanduser().resolve()
    model = load_model(model_path)
    tracker = CentroidTracker()
    calibration_points = []
    previous_centroids = {}
    matrix = None
    center_pixel = None
    frame_count = 0
    lens_maps = None
    frame_shape = None
    window_name = "YOLO Only Capture"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 4:
            calibration_points.append((x, y))
            diagnostic(f"Point selected: {x}, {y}")

    backend = cv2.CAP_DSHOW if isinstance(video_source, int) and os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(video_source, backend)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")
        if isinstance(video_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame count", "object count", "udpstring"])
            csv_file.flush()
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, mouse_callback)
            diagnostic(f"Uncalibrated lens correction k1={LENS_CORRECTION_K1}; calibrate the displayed view.")
            diagnostic("Click four floor points clockwise: TL, TR, BR, BL. Keep the camera stationary.")
            diagnostic("No background capture: every in-area YOLO detection is reported, including stationary objects.")
            diagnostic(f"Dynamic requires more than {motion_threshold_px:g} pixels of centroid movement per processed frame.")
            diagnostic("Press 'b' to reset the motion baseline (IDs retained), or 'q' to quit.")
            diagnostic(f"CSV: {log_path}")
            while True:
                success, clean_frame = cap.read()
                if not success:
                    diagnostic("Video source ended or stopped supplying frames.")
                    break
                if frame_shape is None:
                    frame_shape = clean_frame.shape[:2]
                    if LENS_CORRECTION_K1 != 0:
                        lens_maps = create_lens_correction_maps(frame_shape)
                elif clean_frame.shape[:2] != frame_shape:
                    raise RuntimeError("Video frame size changed. Restart and recalibrate the floor area.")
                if lens_maps is not None:
                    clean_frame = cv2.remap(clean_frame, lens_maps[0], lens_maps[1], cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
                frame_count += 1
                frame = clean_frame.copy()
                if len(calibration_points) == 4 and matrix is None:
                    polygon = np.asarray(calibration_points, dtype=np.int32)
                    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1:
                        diagnostic("Invalid calibration. Select TL, TR, BR, BL again.")
                        calibration_points.clear()
                    else:
                        matrix = calculate_perspective_matrix(calibration_points, UNITY_AREA_WIDTH, UNITY_AREA_HEIGHT)
                        center_pixel = map_pixel_to_unity(np.linalg.inv(matrix), UNITY_AREA_WIDTH / 2, UNITY_AREA_HEIGHT / 2)
                        diagnostic("Area defined. YOLO-only tracking started.")
                if matrix is None:
                    for point in calibration_points:
                        cv2.circle(frame, point, 5, (0, 0, 255), -1)
                    if len(calibration_points) > 1:
                        cv2.polylines(frame, [np.asarray(calibration_points, dtype=np.int32)], False, (0, 0, 255), 2)
                    cv2.putText(frame, f"Calibration: {len(calibration_points)}/4", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Inference sees clean pixels; only the display copy is annotated.
                    detections = recognize(model, clean_frame, confidence, device, calibration_points)
                    assignments = tracker.update([d.centroid for d in detections])
                    packet = build_packet(assignments, detections, matrix, previous_centroids, motion_threshold_px)
                    previous_centroids = {object_id: detections[index].centroid for object_id, index in assignments.items()}
                    emit_packet(frame_count, packet, writer, csv_file)
                    cv2.polylines(frame, [np.asarray(calibration_points, dtype=np.int32)], True, (255, 0, 0), 2)
                    cv2.circle(frame, (int(center_pixel[0]), int(center_pixel[1])), 8, (255, 55, 55), -1)
                    draw_detections(frame, assignments, detections, packet)
                    cv2.putText(frame, f"Active IDs: {packet['count']}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    previous_centroids.clear()
                    diagnostic("Motion baseline reset; session IDs and calibration retained. No background is used.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_source(value):
    try:
        camera_index = int(value)
    except ValueError:
        return str(Path(value).expanduser().resolve())
    if camera_index < 0:
        raise argparse.ArgumentTypeError("Camera index must be non-negative.")
    return camera_index


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Threshold must be greater than 0 and at most 1.")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Motion threshold must be finite and non-negative.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, default=1, help="Camera index or video file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Existing local YOLO weights.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Separate CSV output file.")
    parser.add_argument("--confidence", type=probability, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a supported GPU device.")
    parser.add_argument("--motion-threshold-px", type=nonnegative, default=MOTION_THRESHOLD_PX,
                        help="Dynamic when centroid movement exceeds this many pixels per processed frame.")
    args = parser.parse_args()
    try:
        run_tracking(args.source, args.model, args.log, args.confidence, args.device, args.motion_threshold_px)
    except KeyboardInterrupt:
        diagnostic("Capture interrupted.")
    except Exception as error:
        diagnostic(f"Capture failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Background-based DynamicCapture with YOLO category annotations.

Stdout is reserved for the original [frame, count, json_string] record format.
The original DynamicCapture backend and its output files are not modified.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import distance as dist


SERVICE_DIR = Path(__file__).resolve().parent
NODE_DIR = SERVICE_DIR.parent.parent
DEFAULT_MODEL = NODE_DIR / "apps" / "Yolo" / "yolov8n.pt"
DEFAULT_LOG = NODE_DIR / "apps" / "DynamicCaptureYolo" / "dynamiccapture_yolo_tracking.csv"
UNITY_AREA_WIDTH = 6
UNITY_AREA_HEIGHT = 6
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_DISAPPEARED = 10
MAX_DISTANCE = 50
MIN_CONTOUR_AREA = 800
MOTION_THRESHOLD = 5.0
LENS_CORRECTION_K1 = -0.12  # Experimental medium correction; 0 disables it.


def diagnostic(message):
    print(message, file=sys.stderr, flush=True)


def create_lens_correction_maps(frame_shape):
    """Use the temporary experiment's normalization, not measured X5 intrinsics."""
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
class ForegroundDetection:
    cx: int
    cy: int
    x: int
    y: int
    w: int
    h: int
    is_dynamic: bool

    @property
    def xyxy(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass(frozen=True)
class YoloDetection:
    xyxy: tuple[float, float, float, float]
    class_id: int
    label: str
    confidence: float


class CentroidTracker:
    """Keep missed tracks internally; return only exact current-frame assignments."""

    def __init__(self, max_distance=MAX_DISTANCE, max_disappeared=MAX_DISAPPEARED):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_object_id = 0
        self.centroids = {}
        self.disappeared = {}

    def update(self, input_centroids):
        # ID -> detection index, never ID -> a stale centroid used for rematching.
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
                # Match the original: remove on the eleventh missed frame.
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
    p1 = map_pixel_to_unity(matrix, x, y)
    p2 = map_pixel_to_unity(matrix, x + w, y + h)
    return round(float(abs(p2[0] - p1[0])), 3), round(float(abs(p2[1] - p1[1])), 3)


def extract_foreground(gray, reference_frame, previous_frame, calibration_points):
    frame_delta_base = cv2.absdiff(reference_frame, gray)
    frame_delta_prev = cv2.absdiff(previous_frame, gray)
    thresh_base = cv2.threshold(frame_delta_base, 25, 255, cv2.THRESH_BINARY)[1]
    thresh_prev = cv2.threshold(frame_delta_prev, 25, 255, cv2.THRESH_BINARY)[1]
    thresh_base = cv2.dilate(thresh_base, None, iterations=2)
    contours, _ = cv2.findContours(
        thresh_base.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygon = np.asarray(calibration_points, dtype=np.int32)
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) <= MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        if cv2.pointPolygonTest(polygon, (cx, cy), False) < 0:
            continue
        roi_motion = thresh_prev[y:y + h, x:x + w]
        motion_score = float(np.sum(roi_motion)) / (w * h)
        detections.append(
            ForegroundDetection(cx, cy, x, y, w, h, motion_score > MOTION_THRESHOLD)
        )
    return detections


def bounding_box_iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_identities(foreground, yolo_detections, minimum_iou=0.3):
    candidates = []
    for foreground_index, detection in enumerate(foreground):
        for yolo_index, identity in enumerate(yolo_detections):
            overlap = bounding_box_iou(detection.xyxy, identity.xyxy)
            if overlap >= minimum_iou:
                candidates.append(
                    (-overlap, -identity.confidence, foreground_index, yolo_index)
                )

    matches = {}
    used_yolo = set()
    for _, _, foreground_index, yolo_index in sorted(candidates):
        if foreground_index in matches or yolo_index in used_yolo:
            continue
        matches[foreground_index] = yolo_detections[yolo_index]
        used_yolo.add(yolo_index)
    return matches


def load_model(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Local YOLO weights not found: {model_path}")
    # Keep settings separate from existing YOLO installations. Never download weights.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(SERVICE_DIR / ".venv" / "yolo-config"))
    with redirect_stdout(sys.stderr):
        from ultralytics import YOLO

        return YOLO(str(model_path))


def recognize(model, raw_frame, confidence, device):
    # Libraries may print during initialization or inference; stdout is the data channel.
    with redirect_stdout(sys.stderr):
        results = model.predict(
            source=raw_frame,
            conf=confidence,
            imgsz=640,
            device=None if device == "auto" else device,
            verbose=False,
            save=False,
        )
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        for box, class_id, score in zip(boxes, classes, scores):
            detections.append(
                YoloDetection(
                    tuple(float(value) for value in box),
                    int(class_id),
                    str(result.names[int(class_id)]),
                    float(score),
                )
            )
    return detections


def build_packet(assignments, detections, identities, matrix):
    objects = []
    for object_id, detection_index in sorted(assignments.items()):
        detection = detections[detection_index]
        identity = identities.get(detection_index)
        ux, uy = map_pixel_to_unity(matrix, detection.cx, detection.cy)
        uw, uh = calculate_unity_scale(
            matrix, detection.x, detection.y, detection.w, detection.h
        )
        objects.append({
            "id": int(object_id),
            "pos": {"x": round(float(ux), 2), "y": round(float(uy), 2)},
            "scale": {"w": uw, "h": uh},
            "status": "Dynamic" if detection.is_dynamic else "Static",
            "label": identity.label if identity else "unknown",
            "class_id": identity.class_id if identity else None,
            "confidence": identity.confidence if identity else 0,
        })
    return {"count": len(objects), "objects": objects}


def emit_packet(frame_count, packet, writer, csv_file):
    packet_string = json.dumps(packet, allow_nan=False)
    record = [frame_count, packet["count"], packet_string]
    writer.writerow(record)
    csv_file.flush()
    # Deliberately Python list repr, not JSON: existing receivers expect this wrapper.
    print(record, flush=True)


def draw_detections(frame, assignments, detections, packet):
    for obj in packet["objects"]:
        detection = detections[assignments[obj["id"]]]
        x, y, w, h = detection.x, detection.y, detection.w, detection.h
        color = (0, 0, 255) if detection.is_dynamic else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"ID {obj['id']}: {obj['label']} {obj['confidence']:.2f} {obj['status']}"
        cv2.putText(frame, label, (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_tracking(video_source=1, model_path=DEFAULT_MODEL, log_path=DEFAULT_LOG,
                 confidence=0.5, match_iou=0.3, device="auto"):
    model_path = Path(model_path).expanduser().resolve()
    log_path = Path(log_path).expanduser().resolve()
    model = load_model(model_path)
    tracker = CentroidTracker()
    calibration_points = []
    reference_frame = None
    previous_frame = None
    matrix = None
    center_pixel = None
    frame_count = 0
    lens_maps = None
    lens_frame_shape = None
    window_name = "DynamicCapture YOLO"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 4:
            calibration_points.append((x, y))
            diagnostic(f"Point selected: {x}, {y}")

    # Match the original Windows camera backend; allow files for approved offline tests.
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
            diagnostic(f"Uncalibrated lens correction k1={LENS_CORRECTION_K1}; use the displayed view for calibration.")
            diagnostic("Click four floor points clockwise: TL, TR, BR, BL.")
            diagnostic("Background is captured after the fourth point. Keep the camera stationary.")
            diagnostic("Press 'b' to recapture the background, or 'q' to quit.")
            diagnostic(f"CSV: {log_path}")

            while True:
                success, raw_frame = cap.read()
                if not success:
                    diagnostic("Video source ended or stopped supplying frames.")
                    break
                # Correct clean pixels before calibration, foreground detection and YOLO.
                if LENS_CORRECTION_K1 != 0:
                    if lens_maps is None:
                        lens_frame_shape = raw_frame.shape[:2]
                        lens_maps = create_lens_correction_maps(lens_frame_shape)
                    elif raw_frame.shape[:2] != lens_frame_shape:
                        raise RuntimeError("Video frame size changed. Restart and recalibrate the floor area.")
                    raw_frame = cv2.remap(
                        raw_frame, lens_maps[0], lens_maps[1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                frame_count += 1
                frame = raw_frame.copy()

                if len(calibration_points) == 4 and matrix is None:
                    polygon = np.asarray(calibration_points, dtype=np.int32)
                    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1:
                        diagnostic("Invalid calibration. Select TL, TR, BR, BL again.")
                        calibration_points.clear()
                    else:
                        matrix = calculate_perspective_matrix(
                            calibration_points, UNITY_AREA_WIDTH, UNITY_AREA_HEIGHT
                        )
                        center_pixel = map_pixel_to_unity(
                            np.linalg.inv(matrix), UNITY_AREA_WIDTH / 2, UNITY_AREA_HEIGHT / 2
                        )
                        diagnostic("Area defined. Tracking started.")

                if matrix is None:
                    for point in calibration_points:
                        cv2.circle(frame, point, 5, (0, 0, 255), -1)
                    if len(calibration_points) > 1:
                        cv2.polylines(frame, [np.asarray(calibration_points, dtype=np.int32)],
                                      False, (0, 0, 255), 2)
                    cv2.putText(frame, f"Calibration: {len(calibration_points)}/4", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # All detection inputs are clean pixels, never annotated pixels.
                    gray = cv2.GaussianBlur(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                    cv2.polylines(frame, [np.asarray(calibration_points, dtype=np.int32)],
                                  True, (255, 0, 0), 2)
                    cv2.circle(frame, (int(center_pixel[0]), int(center_pixel[1])),
                               8, (255, 55, 55), -1)
                    if reference_frame is None:
                        reference_frame = gray.copy()
                        previous_frame = gray.copy()
                        diagnostic("Background and motion baseline captured.")
                    else:
                        detections = extract_foreground(
                            gray, reference_frame, previous_frame, calibration_points
                        )
                        assignments = tracker.update([(d.cx, d.cy) for d in detections])
                        recognized = recognize(model, raw_frame, confidence, device) if detections else []
                        identities = match_identities(detections, recognized, match_iou)
                        packet = build_packet(assignments, detections, identities, matrix)
                        previous_frame = gray
                        emit_packet(frame_count, packet, writer, csv_file)
                        draw_detections(frame, assignments, detections, packet)
                        cv2.putText(frame, f"Active IDs: {packet['count']}", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    reference_frame = None
                    previous_frame = None
                    diagnostic("Recapturing background on the next frame; track IDs are not reset.")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, default=1, help="Camera index or video file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Existing local YOLO weights.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Separate CSV output file.")
    parser.add_argument("--confidence", type=probability, default=0.5)
    parser.add_argument("--match-iou", type=probability, default=0.3)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a supported GPU device.")
    args = parser.parse_args()
    try:
        run_tracking(args.source, args.model, args.log, args.confidence, args.match_iou, args.device)
    except KeyboardInterrupt:
        diagnostic("Capture interrupted.")
    except Exception as error:
        diagnostic(f"Capture failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

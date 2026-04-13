import cv2
import numpy as np
import socket
import json
import csv
import time
from scipy.spatial import distance as dist

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5065
UNITY_AREA_WIDTH = 6
UNITY_AREA_HEIGHT = 6
LOG_FILENAME = "tracking_log.csv"
MAX_DISAPPEARED = 10  # Number of frames to wait before deleting an ID
MAX_DISTANCE = 50     # Max pixels an object can move to be considered the same ID

# Global variables for calibration
calibration_points = []
calibration_complete = False
reference_frame = None  
prev_frame = None       

# --- CENTROID TRACKER STATE ---
next_object_id = 0
tracked_centroids = {}  # {id: (cx, cy)}
disappeared_count = {}  # {id: frames_unseen}

def mouse_callback(event, x, y, flags, param):
    global calibration_points, calibration_complete
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"Point selected: {x}, {y}")
        if len(calibration_points) == 4 and not calibration_complete:
            calibration_complete = True
            print("Area defined! Tracking started.")

def calculate_perspective_matrix(src_points, width, height):
    src = np.float32(src_points)
    dst = np.float32([[0,0], [width, 0], [width, height], [0, height]])
    return cv2.getPerspectiveTransform(src, dst)

def map_pixel_to_unity(matrix, pixel_x, pixel_y):
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix)
    return transformed[0][0]

def calculate_unity_scale(matrix, x, y, w, h):
    p1 = map_pixel_to_unity(matrix, x, y)
    p2 = map_pixel_to_unity(matrix, x + w, y + h)
    u_w = abs(p2[0] - p1[0])
    u_h = abs(p2[1] - p1[1])
    return round(float(u_w), 3), round(float(u_h), 3)

def update_tracker(input_centroids):
    global next_object_id, tracked_centroids, disappeared_count
    
    # If no centroids detected, increment disappeared count for all existing
    if len(input_centroids) == 0:
        for object_id in list(disappeared_count.keys()):
            disappeared_count[object_id] += 1
            if disappeared_count[object_id] > MAX_DISAPPEARED:
                del tracked_centroids[object_id]
                del disappeared_count[object_id]
        return {}

    # If currently tracking nothing, register all inputs
    if len(tracked_centroids) == 0:
        for i in range(len(input_centroids)):
            obj_id = next_object_id
            tracked_centroids[obj_id] = input_centroids[i]
            disappeared_count[obj_id] = 0
            next_object_id += 1
    else:
        object_ids = list(tracked_centroids.keys())
        object_values = list(tracked_centroids.values())
        
        # Calculate distance between existing objects and new detections
        D = dist.cdist(np.array(object_values), input_centroids)
        
        # Match based on smallest distances
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            
            # If distance is too large, it's likely a different object
            if D[row, col] > MAX_DISTANCE:
                continue

            obj_id = object_ids[row]
            tracked_centroids[obj_id] = input_centroids[col]
            disappeared_count[obj_id] = 0
            
            used_rows.add(row)
            used_cols.add(col)

        # Handle disappeared objects
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            disappeared_count[obj_id] += 1
            if disappeared_count[obj_id] > MAX_DISAPPEARED:
                del tracked_centroids[obj_id]
                del disappeared_count[obj_id]

        # Handle new objects
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            obj_id = next_object_id
            tracked_centroids[obj_id] = input_centroids[col]
            disappeared_count[obj_id] = 0
            next_object_id += 1

    return tracked_centroids

def run_tracking(video_source=0):
    global reference_frame, prev_frame

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    csv_file = open(LOG_FILENAME, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["frame count", "object count", "udpstring"])

    window_name = "Object Detection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    M = None
    frame_count = 0

    print("\n--- INSTRUCTIONS ---")
    print("1. Click 4 points on the floor (Clockwise: TL, TR, BR, BL).")
    print("2. The system will CAPTURE the background immediately after the 4th click.")
    print("3. Press 'b' to reset/recapture the background ground truth.")
    print("4. Press 'q' to quit.")
    print("--------------------\n")

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret: break
            frame_count += 1
            frame = raw_frame 

            if not calibration_complete:
                for pt in calibration_points:
                    cv2.circle(frame, pt, 5, (0, 0, 255), -1)

                if len(calibration_points) > 1:
                    cv2.polylines(frame, [np.array(calibration_points)], False, (0, 0, 255), 2)

                cv2.putText(frame, f"Calibration: {len(calibration_points)}/4", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if M is None:
                    M = calculate_perspective_matrix(calibration_points, UNITY_AREA_WIDTH, UNITY_AREA_HEIGHT)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if reference_frame is None:
                    reference_frame, prev_frame = gray, gray
                    continue

                # Difference calculation
                frame_delta_base = cv2.absdiff(reference_frame, gray)
                frame_delta_prev = cv2.absdiff(prev_frame, gray)
                thresh_base = cv2.threshold(frame_delta_base, 25, 255, cv2.THRESH_BINARY)[1]
                thresh_prev = cv2.threshold(frame_delta_prev, 25, 255, cv2.THRESH_BINARY)[1]
                thresh_base = cv2.dilate(thresh_base, None, iterations=2)

                contours, _ = cv2.findContours(thresh_base.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                # Draw ROI
                cv2.polylines(frame, [np.array(calibration_points)], True, (255, 0, 0), 2)

                current_detections = [] # (cx, cy, x, y, w, h, is_dynamic)

                for cnt in contours:
                    if cv2.contourArea(cnt) > 800:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = x + w//2, y + h//2
                        
                        if cv2.pointPolygonTest(np.array(calibration_points), (cx, cy), False) >= 0:
                            roi_motion = thresh_prev[y:y+h, x:x+w]
                            motion_score = np.sum(roi_motion) / (w * h)
                            current_detections.append((cx, cy, x, y, w, h, motion_score > 5.0))

                # Update Tracker
                input_centroids = [d[:2] for d in current_detections]
                active_tracks = update_tracker(input_centroids)

                detected_objects = []
                for obj_id, centroid in active_tracks.items():
                    # Find the detection metadata corresponding to this centroid
                    # (Matching by proximity to find the correct bounding box/status)
                    match = None
                    min_dist = 999
                    for det in current_detections:
                        d = dist.euclidean(centroid, det[:2])
                        if d < min_dist:
                            min_dist = d
                            match = det
                    
                    if match:
                        cx, cy, x, y, w, h, is_dynamic = match
                        ux, uy = map_pixel_to_unity(M, cx, cy)
                        uw, uh = calculate_unity_scale(M, x, y, w, h)
                        status = "Dynamic" if is_dynamic else "Static"

                        obj_data = {
                            "id": int(obj_id),
                            "pos": {"x": round(float(ux), 2), "y": round(float(uy), 2)},
                            "scale": {"w": uw, "h": uh},
                            "status": status
                        }
                        detected_objects.append(obj_data)

                        color = (0, 0, 255) if is_dynamic else (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"ID {obj_id}: {status}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                prev_frame = gray
                udp_packet = {"count": len(detected_objects), "objects": detected_objects}
                udp_string = json.dumps(udp_packet)
                # sock.sendto(udp_string.encode(), (UDP_IP, UDP_PORT))

                # key to send over ubiq genie
                print([frame_count, len(detected_objects), udp_string])

                writer.writerow([frame_count, len(detected_objects), udp_string])
                
                cv2.putText(frame, f"Active IDs: {len(active_tracks)}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('b'): reference_frame = None

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

if __name__ == "__main__":
    run_tracking(1)
import cv2
import numpy as np
import socket
import json
import csv
import time

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5065
UNITY_AREA_WIDTH = 10.0
UNITY_AREA_HEIGHT = 10.0
LOG_FILENAME = "tracking_log.csv"

# Camera Params (Insta360)
K = np.array([[1.15709813e+03, 0, 9.86153891e+02],
              [0, 1.17329523e+03, 5.82845627e+02],
              ], dtype=np.float32)
D = np.array([-0.57465333, 0.64656927, 0.01184943, 0.0153547, -0.39175262], dtype=np.float32)

# Global variables for calibration
calibration_points = []
calibration_complete = False

def mouse_callback(event, x, y, flags, param):
    global calibration_points, calibration_complete
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"Point selected: {x}, {y}")
        if len(calibration_points) == 4:
            calibration_complete = True
            print("Area defined! Tracking started.")

def calculate_perspective_matrix(src_points, width, height):
    src = np.float32(src_points)
    # Unity Coordinate Mapping (Top-Left Origin)
    dst = np.float32([[0,0],
        [width, 0],
        [width, height],
        [0, height]])
    return cv2.getPerspectiveTransform(src, dst)

def map_pixel_to_unity(matrix, pixel_x, pixel_y):
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix)
    return transformed[0][0]

def run_tracking(video_source=0):
    # 1. Setup UDP
    #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 2. Setup Camera
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW) # Changed to 0 for default webcam, change back to 1 if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 3. Undistort Setup (Optimized)
    DIM = (1920, 1080)
    # Estimate new camera matrix for rectified view
    '''new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, DIM, np.eye(3), balance=0.5
    )
    # Use 16-bit fixed point maps for faster remap
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2
    )'''

    # 4. Background Subtraction Setup
    # detectShadows=True helps separate objects, thresholding later removes the shadows
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # 5. Logging Setup
    csv_file = open(LOG_FILENAME, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow("frame count, object count, udpstring")
    print(f"Logging started to {LOG_FILENAME}")

    window_name = "Undistorted Tracking"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    M = None
    frame_count = 0

    print("\n--- INSTRUCTIONS ---")
    print("1. The video is now UNDISTORTED.")
    print("2. Click 4 points on the floor (Clockwise: TL, TR, BR, BL) to calibrate Unity space.")
    print("--------------------\n")

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret: break
            
            frame_count += 1

            # --- UNDISTORTION STEP ---
            # Apply the maps calculated during setup
            frame = raw_frame# cv2.remap(raw_frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # --- CALIBRATION PHASE ---
            if not calibration_complete:
                for pt in calibration_points:
                    cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                if len(calibration_points) > 1:
                    cv2.polylines(frame, [np.array(calibration_points)], False, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Click 4 Points: {len(calibration_points)}/4", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- TRACKING PHASE ---
            else:
                if M is None:
                    M = calculate_perspective_matrix(calibration_points, UNITY_AREA_WIDTH, UNITY_AREA_HEIGHT)

                # Draw ROI
                cv2.polylines(frame, [np.array(calibration_points)], True, (255, 0, 0), 2)

                # 1. Background Subtraction
                fgMask = backSub.apply(frame)
                
                # 2. Thresholding (Remove Shadows)
                # Shadows are typically 127 in MOG2, Objects are 255. 
                # Threshold at 200 to keep only hard objects.
                _, thresh = cv2.threshold(fgMask, 220, 255, cv2.THRESH_BINARY)

                # 3. Noise Removal (Morphology)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # Remove noise
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Fill holes

                # 4. Find Contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detected_objects = []

                # 5. Process ALL Contours
                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    
                    # Filter small noise
                    if area > 800:
                        # Get Bounding Box
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # Get Centroid
                        M_moments = cv2.moments(cnt)
                        if M_moments["m00"]!= 0:
                            cx = int(M_moments["m10"] / M_moments["m00"])
                            cy = int(M_moments["m01"] / M_moments["m00"])
                            
                             # Draw on screen
                            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                            # Transform to Unity Coordinates
                            unity_x, unity_y = map_pixel_to_unity(M, cx, cy)

                            # Store Data
                            obj_data = {
                                "id": i,
                                "x": round(float(unity_x), 2),
                                "y": round(float(unity_y), 2)
                            }
                            detected_objects.append(obj_data)

                            # Visualization
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(frame, f"{unity_x:.1f},{unity_y:.1f}", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 6. Report & Log Data
                total_objects = len(detected_objects)
                
                # Format for UDP (JSON is safest for lists)
                # Example: {"count": 2, "objects": [{"id":0, "x":1.2, "y":3.4},...]}
                udp_packet = {
                    "count": total_objects,
                    "objects": detected_objects
                }
                udp_string = json.dumps(udp_packet)
                
                # Send to Unity
                # sock.sendto(udp_string.encode(), (UDP_IP, UDP_PORT))

                # Log to CSV
                writer.writerow([frame_count, total_objects, udp_string])
                print([frame_count, total_objects, udp_string])
                # On-screen Counter
                cv2.putText(frame, f"Objects: {total_objects}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print("Execution finished. Log saved.")

if __name__ == "__main__":
    run_tracking()
import time
import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2
from collections import deque

FHP_THRESHOLD = 1.05
SLOUCHING_THRESHOLD = 0.7
SHOULDER_ASYMMETRY_THRESHOLD = 0.08
SMOOTHING_WINDOW = 10

GRACE_PERIOD_SECONDS = 3.0
ABSENCE_THRESHOLD = 5.0

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, 
    smooth_landmarks=True,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Initializing Pi5 Camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()

calibrated = False
baseline_face_width = 0
baseline_neck_dist = 0
baseline_shoulder_tilt = 0

face_width_buffer = deque(maxlen=SMOOTHING_WINDOW)
neck_dist_buffer = deque(maxlen=SMOOTHING_WINDOW)
shoulder_tilt_buffer = deque(maxlen=SMOOTHING_WINDOW)

last_detection_time = time.time()
bad_posture_start_time = None
current_displayed_status = "SEARCHING"
current_color = (200, 200, 200)

print("Camera Running using HOLISTIC Model for FHP, slouching and shoulder asymmetry.")
print("1. Sit Straight.")
print("2. Press 'c' to Calibrate.")
print("3. Lean forward to test.")

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    
    results = holistic.process(frame_rgb)
    
    is_turning = False
    
    if results.face_landmarks and results.pose_landmarks:
        last_detection_time = time.time()
        is_present = True
        
        h, w, _ = frame.shape

        face_lms = results.face_landmarks.landmark
        nose = face_lms[1]
        l_ear = face_lms[234]
        r_ear = face_lms[454]

        pose_lms = results.pose_landmarks.landmark
        l_shldr = pose_lms[11]
        r_shldr = pose_lms[12]

        nx, ny = int(nose.x * w), int(nose.y * h)
        lex, ley = int(l_ear.x * w), int(l_ear.y * h)
        rex, rey = int(r_ear.x * w), int(r_ear.y * h)
        lsx, lsy = int(l_shldr.x * w), int(l_shldr.y * h)
        rsx, rsy = int(r_shldr.x * w), int(r_shldr.y * h)
        
        dist_nose_left = ((nx - lex)**2 + (ny - ley)**2)**0.5
        dist_nose_right = ((nx - rex)**2 + (ny - rey)**2)**0.5
        if dist_nose_right == 0: dist_nose_right = 0.001
        yaw_ratio = dist_nose_left / dist_nose_right
        is_turning = yaw_ratio < 0.5 or yaw_ratio > 2.0

        face_width = ((lex - rex)**2 + (ley - rey)**2)**0.5
        shoulder_mid_y = (lsy + rsy) / 2
        neck_dist = shoulder_mid_y - ny
        current_shoulder_tilt = lsy - rsy
        
        if not is_turning:
            face_width_buffer.append(face_width)
            shoulder_tilt_buffer.append(current_shoulder_tilt)
        neck_dist_buffer.append(neck_dist)
        
        avg_face_width = np.mean(face_width_buffer) if face_width_buffer else face_width
        avg_neck_dist = np.mean(neck_dist_buffer) if neck_dist_buffer else neck_dist
        avg_shoulder_tilt = np.mean(shoulder_tilt_buffer) if shoulder_tilt_buffer else current_shoulder_tilt
        
        pending_status = "UNCALIBRATED"
        pending_color = (255, 255, 0)
        face_ratio = 0.0
        slouch_ratio = 0.0
        tilt_ratio = 0.0
        
        if calibrated:
            face_ratio = avg_face_width / baseline_face_width
            slouch_ratio = avg_neck_dist / baseline_neck_dist
            tilt_change = abs(avg_shoulder_tilt - baseline_shoulder_tilt)
            tilt_ratio = tilt_change / baseline_face_width
            
            if is_turning:
                if slouch_ratio < SLOUCHING_THRESHOLD:
                    pending_status = "SLOUCHING" 
                    pending_color = (0, 0, 255)
                else:
                    pending_status = "HEAD TURNED"
                    pending_color = (255, 200, 0) 
            else:
                if tilt_ratio > SHOULDER_ASYMMETRY_THRESHOLD:
                    pending_status = "ASYMMETRIC SHOULDERS"
                    pending_color = (0, 0, 255)
                elif face_ratio > FHP_THRESHOLD:
                    pending_status = "LEANING FORWARD"
                    pending_color = (0, 0, 255)
                elif slouch_ratio < SLOUCHING_THRESHOLD:
                    pending_status = "SLOUCHING"
                    pending_color = (0, 0, 255)
                else:
                    pending_status = "Posture OK"
                    pending_color = (0, 255, 0)
                    
            if face_ratio > FHP_THRESHOLD and slouch_ratio < SLOUCHING_THRESHOLD:
                pending_status = "CRITICAL POSTURE"
                pending_color = (0, 0, 255)
                
            is_bad_posture = (pending_color == (0, 0, 255))
            
            if is_bad_posture:
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                    
                elapsed_time = time.time() - bad_posture_start_time
                
                if elapsed_time > GRACE_PERIOD_SECONDS:
                    current_displayed_status = pending_status
                    current_color = pending_color
                else:
                    countdown = int(GRACE_PERIOD_SECONDS - elapsed_time) + 1
                    current_displayed_status = f"Warning... {countdown}"
                    current_color = (0, 165, 255)
            else:
                bad_posture_start_time = None
                current_displayed_status = pending_status
                current_color = pending_color
            
            cv2.putText(frame, f"Ratio: {face_ratio:.2f}x", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(frame, f"Slouch Ratio: {slouch_ratio:.2f}x", (20,155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            cv2.putText(frame, f"Tilt (Asym): {tilt_ratio:.2f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            
        cv2.putText(frame, current_displayed_status, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 3)
        
        cv2.line(frame, (lex, ley), (rex, rey), (0, 255, 0), 2)
        cv2.line(frame, (nx, ny), (nx, int(shoulder_mid_y)), (0, 255, 255), 2)
        cv2.line(frame, (lsx, lsy), (rsx, rsy), (255, 0, 0), 2)

        cv2.putText(frame, f"Face Width: {face_width:.1f}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Neck Dist:  {neck_dist:.1f}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        if time.time() - last_detection_time > ABSENCE_THRESHOLD:
            status = "IDLE (User Away)"
            color = (100, 100, 100)
            
            face_width_buffer.clear()
            neck_dist_buffer.clear()
            shoulder_tilt_buffer.clear()
            bad_posture_start_time = None
            
            cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        else:
            cv2.putText(frame, "SEARCHING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)

    cv2.imshow("Holistic Accuracy Test", frame)
    
    key = cv2.waitKey(1) & 0xFF 

    if key == ord('q'):
        break
    elif key == ord('c'):
        if not is_turning and len(face_width_buffer) == SMOOTHING_WINDOW and len(neck_dist_buffer) == SMOOTHING_WINDOW:
            baseline_face_width = np.mean(face_width_buffer)
            baseline_neck_dist = np.mean(neck_dist_buffer)
            baseline_shoulder_tilt = np.mean(shoulder_tilt_buffer)
            calibrated = True
            print(f"✅ Calibrated! Baseline Width: {baseline_face_width:.1f} pixels! Baseline neck distance: {baseline_neck_dist:.1f}")
        else:
            print("Waiting for buffer to fill")

picam2.stop()
cv2.destroyAllWindows()

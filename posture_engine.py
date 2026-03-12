import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import config

class PostureEngine:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.calibrated = False
        self.baseline_face_width = 0
        self.baseline_neck_dist = 0
        self.baseline_shoulder_tilt = 0

        self.face_width_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.neck_dist_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.shoulder_tilt_buffer = deque(maxlen=config.SMOOTHING_WINDOW)

        self.last_detection_time = time.time()
        self.bad_posture_start_time = None
        self.current_displayed_status = "SEARCHING"
        self.current_color = (200, 200, 200)
        self.is_turning = False

    def trigger_calibration(self):
        if not self.is_turning and len(self.face_width_buffer) == config.SMOOTHING_WINDOW:
            self.baseline_face_width = np.mean(self.face_width_buffer)
            self.baseline_neck_dist = np.mean(self.neck_dist_buffer)
            self.baseline_shoulder_tilt = np.mean(self.shoulder_tilt_buffer)
            self.calibrated = True
            return True, self.baseline_face_width, self.baseline_neck_dist
        return False, 0, 0

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        
        self.is_turning = False
        h, w, _ = frame.shape
        
        landmarks_dict = None

        if results.face_landmarks and results.pose_landmarks:
            self.last_detection_time = time.time()
            
            face_lms = results.face_landmarks.landmark
            nose, l_ear, r_ear = face_lms[1], face_lms[234], face_lms[454]
            pose_lms = results.pose_landmarks.landmark
            l_shldr, r_shldr = pose_lms[11], pose_lms[12]
            
            landmarks_dict = {
                "n": [nose.x, nose.y],
                "le": [l_ear.x, l_ear.y],
                "re": [r_ear.x, r_ear.y],
                "ls": [l_shldr.x, l_shldr.y],
                "rs": [r_shldr.x, r_shldr.y]
            }

            nx, ny = int(nose.x * w), int(nose.y * h)
            lex, ley = int(l_ear.x * w), int(l_ear.y * h)
            rex, rey = int(r_ear.x * w), int(r_ear.y * h)
            lsx, lsy = int(l_shldr.x * w), int(l_shldr.y * h)
            rsx, rsy = int(r_shldr.x * w), int(r_shldr.y * h)
            
            dist_nose_left = ((nx - lex)**2 + (ny - ley)**2)**0.5
            dist_nose_right = ((nx - rex)**2 + (ny - rey)**2)**0.5
            yaw_ratio = dist_nose_left / (dist_nose_right + 1e-6)
            self.is_turning = yaw_ratio < 0.5 or yaw_ratio > 2.0

            face_width = ((lex - rex)**2 + (ley - rey)**2)**0.5
            shoulder_mid_y = (lsy + rsy) / 2
            neck_dist = shoulder_mid_y - ny
            current_shoulder_tilt = lsy - rsy
            
            if not self.is_turning:
                self.face_width_buffer.append(face_width)
                self.shoulder_tilt_buffer.append(current_shoulder_tilt)
            self.neck_dist_buffer.append(neck_dist)
            
            avg_face_width = np.mean(self.face_width_buffer) if self.face_width_buffer else face_width
            avg_neck_dist = np.mean(self.neck_dist_buffer) if self.neck_dist_buffer else neck_dist
            avg_shoulder_tilt = np.mean(self.shoulder_tilt_buffer) if self.shoulder_tilt_buffer else current_shoulder_tilt
            
            pending_status = "UNCALIBRATED"
            pending_color = (255, 255, 0)
            
            if self.calibrated:
                face_ratio = avg_face_width / (self.baseline_face_width + 1e-6)
                slouch_ratio = avg_neck_dist / (self.baseline_neck_dist + 1e-6)
                tilt_change = abs(avg_shoulder_tilt - self.baseline_shoulder_tilt)
                tilt_ratio = tilt_change / (self.baseline_face_width + 1e-6)
                
                is_slouching = slouch_ratio < config.SLOUCHING_THRESHOLD
                is_leaning = face_ratio > config.FHP_THRESHOLD
                is_asymmetric = tilt_ratio > config.SHOULDER_ASYMMETRY_THRESHOLD
                
                bad_posture_count = sum([is_slouching, is_leaning, is_asymmetric])
                
                if self.is_turning:
                    if is_slouching:
                        pending_status, pending_color = "SLOUCHING", (0, 0, 255)
                    else:
                        pending_status, pending_color = "HEAD TURNED", (255, 200, 0) 
                else:
                    if bad_posture_count >= 2:
                        pending_status, pending_color = "CRITICAL POSTURE", (0, 0, 255)
                    elif is_asymmetric:
                        pending_status, pending_color = "ASYMMETRIC SHOULDERS", (0, 0, 255)
                    elif is_leaning:
                        pending_status, pending_color = "LEANING FORWARD", (0, 0, 255)
                    elif is_slouching:
                        pending_status, pending_color = "SLOUCHING", (0, 0, 255)
                    else:
                        pending_status, pending_color = "Posture OK", (0, 255, 0)
                    
                is_bad_posture = (pending_color == (0, 0, 255))
                
                if is_bad_posture:
                    if self.bad_posture_start_time is None:
                        self.bad_posture_start_time = time.time()
                    elapsed_time = time.time() - self.bad_posture_start_time
                    if elapsed_time > config.GRACE_PERIOD_SECONDS:
                        self.current_displayed_status = pending_status
                        self.current_color = pending_color
                    else:
                        countdown = int(config.GRACE_PERIOD_SECONDS - elapsed_time) + 1
                        self.current_displayed_status = f"Warning... {countdown}"
                        self.current_color = (0, 165, 255)
                else:
                    self.bad_posture_start_time = None
                    self.current_displayed_status = pending_status
                    self.current_color = pending_color
                
                cv2.putText(frame, f"Ratio: {face_ratio:.2f}x", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(frame, f"Slouch: {slouch_ratio:.2f}x", (20,155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                
            cv2.putText(frame, self.current_displayed_status, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, self.current_color, 3)
            cv2.line(frame, (lex, ley), (rex, rey), (0, 255, 0), 2)
            cv2.line(frame, (nx, ny), (nx, int(shoulder_mid_y)), (0, 255, 255), 2)
            cv2.line(frame, (lsx, lsy), (rsx, rsy), (255, 0, 0), 2)

        else:
            if time.time() - self.last_detection_time > config.ABSENCE_THRESHOLD:
                self.face_width_buffer.clear()
                self.neck_dist_buffer.clear()
                self.shoulder_tilt_buffer.clear()
                self.bad_posture_start_time = None
                self.current_displayed_status = "IDLE (User Away)"
                cv2.putText(frame, "IDLE (User Away)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 3)
            else:
                cv2.putText(frame, "SEARCHING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)

        return frame, self.current_displayed_status, landmarks_dict

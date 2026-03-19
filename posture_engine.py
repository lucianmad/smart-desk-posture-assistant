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
        self.baseline_nose_to_mid_shoulders_dist = 0
        self.baseline_shoulder_height_diff = 0

        self.face_width_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.nose_to_mid_shoulders_dist_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.shoulder_height_diff_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.shoulder_width_buffer = deque(maxlen=5)

        self.last_detection_time = time.time()
        self.bad_posture_start_time = None
        self.current_displayed_status = "SEARCHING"
        self.current_color = (200, 200, 200)
        self.is_turning = False

    def trigger_calibration(self):
        if not self.is_turning and len(self.face_width_buffer) == config.SMOOTHING_WINDOW:
            self.baseline_face_width = np.mean(self.face_width_buffer)
            self.baseline_nose_to_mid_shoulders_dist = np.mean(self.nose_to_mid_shoulders_dist_buffer)
            self.baseline_shoulder_height_diff = np.mean(self.shoulder_height_diff_buffer)
            self.baseline_shoulder_width = np.mean(self.shoulder_width_buffer)
            self.calibrated = True
            return True, self.baseline_face_width, self.baseline_nose_to_mid_shoulders_dist
        return False, 0, 0

    def _extract_landmarks(self, results, h, w):
        face_lms = results.face_landmarks.landmark
        pose_lms = results.pose_landmarks.landmark
        
        nose, l_ear, r_ear = face_lms[1], face_lms[234], face_lms[454]
        l_shoulder, r_shoulder = pose_lms[11], pose_lms[12]
        
        landmarks_normalized = {
            "n": [nose.x, nose.y],
            "le": [l_ear.x, l_ear.y],
            "re": [r_ear.x, r_ear.y],
            "ls": [l_shoulder.x, l_shoulder.y],
            "rs": [r_shoulder.x, r_shoulder.y]
        }
        
        landmarks_pixel = {
            "nose": (int(nose.x * w), int(nose.y * h)),
            "left_ear": (int(l_ear.x * w), int(l_ear.y * h)),
            "right_ear": (int(r_ear.x * w), int(r_ear.y * h)),
            "left_shoulder": (int(l_shoulder.x * w), int(l_shoulder.y * h)),
            "right_shoulder": (int(r_shoulder.x * w), int(r_shoulder.y * h))
        }
        
        return landmarks_normalized, landmarks_pixel
        
    def _calculate_yaw(self, landmarks_pixel):
        nx, ny = landmarks_pixel["nose"]
        lex, ley = landmarks_pixel["left_ear"]
        rex, rey = landmarks_pixel["right_ear"]
        
        nose_to_left_ear_dist = ((nx - lex) ** 2 + (ny - ley) ** 2) ** 0.5
        nose_to_right_ear_dist = ((nx - rex) ** 2 + (ny - rey) ** 2) ** 0.5
        yaw_ratio = nose_to_left_ear_dist / (nose_to_right_ear_dist + 1e-6)
        
        return yaw_ratio < 0.5 or yaw_ratio > 2
        
    def _calculate_metrics(self, landmarks_pixel, is_turning):
        nx, ny = landmarks_pixel["nose"]
        lex, ley = landmarks_pixel["left_ear"]
        rex, rey = landmarks_pixel["right_ear"]
        lsx, lsy = landmarks_pixel["left_shoulder"]
        rsx, rsy = landmarks_pixel["right_shoulder"]
        
        face_width = ((lex - rex) ** 2 + (ley - rey) ** 2) ** 0.5
        shoulder_mid_y = (lsy + rsy) / 2
        nose_to_mid_shoulders_dist = ny - shoulder_mid_y
        shoulder_height_diff = lsy - rsy
        shoulder_width = ((lsx - rsx) ** 2 + (lsy - rsy) ** 2) ** 0.5
        
        if not is_turning:
            self.face_width_buffer.append(face_width)
        
        self.shoulder_height_diff_buffer.append(shoulder_height_diff)
        self.nose_to_mid_shoulders_dist_buffer.append(nose_to_mid_shoulders_dist)
        self.shoulder_width_buffer.append(shoulder_width)
        
        avg_face_width = np.mean(self.face_width_buffer) if self.face_width_buffer else face_width
        avg_nose_to_mid_shoulders_dist = np.mean(self.nose_to_mid_shoulders_dist_buffer) if self.nose_to_mid_shoulders_dist_buffer else nose_to_mid_shoulders_dist
        avg_shoulder_height_diff = np.mean(self.shoulder_height_diff_buffer) if self.shoulder_height_diff_buffer else shoulder_height_diff
        avg_shoulder_width = np.mean(self.shoulder_width_buffer) if self.shoulder_width_buffer else shoulder_width
            
        return avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_height_diff, avg_shoulder_width, shoulder_mid_y
        
    def _evaluate_posture(self, avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_height_diff, avg_shoulder_width, is_turning):
        fhp_ratio = avg_face_width / (self.baseline_face_width + 1e-6)
        nms_ratio = avg_nose_to_mid_shoulders_dist / (self.baseline_nose_to_mid_shoulders_dist + 1e-6)
        shoulder_height_diff = abs(avg_shoulder_height_diff - self.baseline_shoulder_height_diff)
        shoulder_asym_ratio = shoulder_height_diff / (avg_shoulder_width + 1e-6)
        
        is_fhp = fhp_ratio > config.FHP_THRESHOLD
        is_slouching = nms_ratio < config.SLOUCHING_THRESHOLD and fhp_ratio > config.SLOUCHING_FHP_MIN_THRESHOLD
        is_asymmetric = shoulder_asym_ratio > config.SHOULDER_ASYMMETRY_THRESHOLD
        
        bad_posture_count = sum([is_fhp, is_slouching, is_asymmetric])
        
        if is_turning:
            if is_slouching and is_asymmetric:
                pending_status, pending_color = "CRITICAL POSTURE", (0, 0, 255)
            elif is_slouching:
                pending_status, pending_color = "SLOUCHING", (0, 0, 255)
            elif is_asymmetric:
                pending_status, pending_color = "ASYMMETRIC SHOULDERS", (0, 0, 255)
            else:
                pending_status, pending_color = "HEAD TURNED", (255, 200, 0)
        else:
            if bad_posture_count >= 2:
                pending_status, pending_color = "CRITICAL POSTURE", (0, 0, 255)
            elif is_asymmetric:
                pending_status, pending_color = "ASYMMETRIC SHOULDERS", (0, 0, 255)
            elif is_fhp:
                pending_status, pending_color = "LEANING FORWARD", (0, 0, 255)
            elif is_slouching:
                pending_status, pending_color = "SLOUCHING", (0, 0, 255)
            else:
                pending_status, pending_color = "Posture OK", (0, 255, 0)
            
        return pending_status, pending_color, fhp_ratio, nms_ratio, shoulder_asym_ratio
        
    def _draw_overlay(self, frame, landmarks_pixel, shoulder_mid_y, status, color, fhp_ratio, nms_ratio, shoulder_asym_ratio):
        nx, ny = landmarks_pixel["nose"]
        lex, ley = landmarks_pixel["left_ear"]
        rex, rey = landmarks_pixel["right_ear"]
        lsx, lsy = landmarks_pixel["left_shoulder"]
        rsx, rsy = landmarks_pixel["right_shoulder"]
        
        cv2.line(frame, (lex, ley), (rex, rey), (0, 255, 0), 2)
        cv2.line(frame, (nx, ny), (nx, int(shoulder_mid_y)), (0, 255, 255), 2)
        cv2.line(frame, (lsx, lsy), (rsx, rsy), (255, 0, 0), 2)
        cv2.putText(frame, status, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
        if fhp_ratio > 0:
            cv2.putText(frame, f"FHP: {fhp_ratio:.2f}x", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(frame, f"NMS: {nms_ratio:.2f}x", (20, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"ASYM: {shoulder_asym_ratio:.2f}x", (20, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return frame
        

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        
        self.is_turning = False
        h, w, _ = frame.shape
        
        landmarks_normalized = None

        if results.face_landmarks and results.pose_landmarks:
            self.last_detection_time = time.time()
            
            landmarks_normalized, landmarks_pixel = self._extract_landmarks(results, h, w)
            
            self.is_turning = self._calculate_yaw(landmarks_pixel)
            
            avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_height_diff, avg_shoulder_width, shoulder_mid_y = self._calculate_metrics(landmarks_pixel, self.is_turning)
            
            pending_status = "UNCALIBRATED"
            pending_color = (255, 255, 0)
            
            fhp_ratio = 0
            nms_ratio = 0
            shoulder_asym_ratio = 0
            
            if self.calibrated:
                pending_status, pending_color, fhp_ratio, nms_ratio, shoulder_asym_ratio = self._evaluate_posture(avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_height_diff, avg_shoulder_width, self.is_turning)
                    
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

            frame = self._draw_overlay(frame, landmarks_pixel, shoulder_mid_y, self.current_displayed_status, self.current_color, fhp_ratio, nms_ratio, shoulder_asym_ratio)

        else:
            if time.time() - self.last_detection_time > config.ABSENCE_THRESHOLD:
                self.face_width_buffer.clear()
                self.nose_to_mid_shoulders_dist_buffer.clear()
                self.shoulder_height_diff_buffer.clear()
                self.shoulder_width_buffer.clear()
                self.bad_posture_start_time = None
                self.current_displayed_status = "IDLE (User Away)"
                cv2.putText(frame, "IDLE (User Away)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 3)
            else:
                cv2.putText(frame, "SEARCHING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)

        return frame, self.current_displayed_status, landmarks_normalized

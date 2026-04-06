import time
import cv2
import math
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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.calibrated = False
        self.baseline_face_width = 0
        self.baseline_nose_to_mid_shoulders_dist = 0
        self.baseline_shoulder_angle = 0
        self.baseline_head_angle = 0
        self.baseline_shoulder_mid_y = 0

        self.face_width_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.nose_to_mid_shoulders_dist_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.shoulder_angle_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.head_angle_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        self.shoulder_mid_y_buffer = deque(maxlen=config.SMOOTHING_WINDOW)

        self.last_detection_time = time.time()
        self.bad_posture_start_time = None
        self.prolonged_bad_posture_start_time = None
        self.current_displayed_status = config.Status.SEARCHING
        self.current_color = config.Colors.INFO
        self.is_turning = False

    def trigger_calibration(self):
        if not self.is_turning and len(self.face_width_buffer) == config.SMOOTHING_WINDOW:
            self.baseline_face_width = np.mean(self.face_width_buffer)
            self.baseline_nose_to_mid_shoulders_dist = np.mean(self.nose_to_mid_shoulders_dist_buffer)
            self.baseline_shoulder_angle = np.mean(self.shoulder_angle_buffer)
            self.baseline_head_angle = np.mean(self.head_angle_buffer)
            self.baseline_shoulder_mid_y = np.mean(self.shoulder_mid_y_buffer)
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
        nose_to_mid_shoulders_dist = shoulder_mid_y - ny
        shoulder_angle = math.degrees(math.atan2(lsy - rsy, lsx - rsx))
        head_angle = math.degrees(math.atan2(rey - ley, rex - lex))
        
        if not is_turning:
            self.face_width_buffer.append(face_width)
        
        self.nose_to_mid_shoulders_dist_buffer.append(nose_to_mid_shoulders_dist)
        self.shoulder_angle_buffer.append(shoulder_angle)
        self.head_angle_buffer.append(head_angle)
        self.shoulder_mid_y_buffer.append(shoulder_mid_y)
        
        avg_face_width = np.mean(self.face_width_buffer) if self.face_width_buffer else face_width
        avg_nose_to_mid_shoulders_dist = np.mean(self.nose_to_mid_shoulders_dist_buffer) if self.nose_to_mid_shoulders_dist_buffer else nose_to_mid_shoulders_dist
        avg_shoulder_angle = np.mean(self.shoulder_angle_buffer) if self.shoulder_angle_buffer else shoulder_angle
        avg_head_angle = np.mean(self.head_angle_buffer) if self.head_angle_buffer else head_angle
        avg_shoulder_mid_y = np.mean(self.shoulder_mid_y_buffer) if self.shoulder_mid_y_buffer else shoulder_mid_y
            
        return avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_angle, avg_head_angle, avg_shoulder_mid_y
        
    def _evaluate_posture(self, avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_angle, avg_head_angle, avg_shoulder_mid_y, is_turning):
        fhp_ratio = avg_face_width / (self.baseline_face_width + 1e-6)
        nms_ratio = avg_nose_to_mid_shoulders_dist / (self.baseline_nose_to_mid_shoulders_dist + 1e-6)
        shoulder_angle_diff = abs(avg_shoulder_angle - self.baseline_shoulder_angle)
        head_angle_diff = abs(avg_head_angle - self.baseline_head_angle)
        pixel_drop = avg_shoulder_mid_y - self.baseline_shoulder_mid_y
        normalized_drop = pixel_drop / (avg_face_width + 1e-6)
        
        is_looking_down = nms_ratio < config.LOOKING_DOWN_THRESHOLD
        is_fhp = fhp_ratio > config.FHP_THRESHOLD
        is_slouching = normalized_drop > config.NORMALIZED_DROP_THRESHOLD
        is_asymmetric_shoulders = shoulder_angle_diff > config.SHOULDER_ASYMMETRY_THRESHOLD
        is_head_tilted = head_angle_diff > config.HEAD_TILT_THRESHOLD
        
        if is_turning:
            if is_asymmetric_shoulders:
                pending_status, pending_color = config.Status.ASYMMETRIC_SHOULDERS, config.Colors.CRITICAL
            elif is_head_tilted:
                pending_status, pending_color = config.Status.HEAD_TILTED, config.Colors.CRITICAL
            else:
                pending_status, pending_color = config.Status.HEAD_TURNED, config.Colors.INFO
        else:
            if is_slouching:
                pending_status, pending_color = config.Status.SLOUCHING, config.Colors.CRITICAL
            elif is_fhp:
                pending_status, pending_color = config.Status.FHP, config.Colors.CRITICAL
            elif is_asymmetric_shoulders:
                pending_status, pending_color = config.Status.ASYMMETRIC_SHOULDERS, config.Colors.CRITICAL
            elif is_head_tilted:
                pending_status, pending_color = config.Status.HEAD_TILTED, config.Colors.CRITICAL
            elif is_looking_down:
                pending_status, pending_color = config.Status.LOOKING_DOWN, config.Colors.INFO
            else:
                pending_status, pending_color = config.Status.OK, config.Colors.OK
            
        return pending_status, pending_color, fhp_ratio, nms_ratio, shoulder_angle_diff, head_angle_diff, normalized_drop
        
    def _draw_overlay(self, frame, landmarks_pixel, avg_shoulder_mid_y, status, color, fhp_ratio, nms_ratio, shoulder_angle_diff, head_angle_diff, shoulder_drop):
        nx, ny = landmarks_pixel["nose"]
        lex, ley = landmarks_pixel["left_ear"]
        rex, rey = landmarks_pixel["right_ear"]
        lsx, lsy = landmarks_pixel["left_shoulder"]
        rsx, rsy = landmarks_pixel["right_shoulder"]
        
        cv2.line(frame, (lex, ley), (rex, rey), (0, 255, 0), 2)
        cv2.line(frame, (nx, ny), (nx, int(avg_shoulder_mid_y)), (0, 255, 255), 2)
        cv2.line(frame, (lsx, lsy), (rsx, rsy), (255, 0, 0), 2)
        cv2.putText(frame, status, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
        if fhp_ratio > 0:
            cv2.putText(frame, f"FHP: {fhp_ratio:.2f}x", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"NMS: {nms_ratio:.2f}x", (20, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"SHOULDER ASYMMETRY: {shoulder_angle_diff:.1f} degrees", (20, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"HEAD TILT: {head_angle_diff:.1f} degrees", (20, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"SHOULDER DROP: {shoulder_drop:.2f}x", (20, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
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
            
            avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_angle, avg_head_angle, avg_shoulder_mid_y = self._calculate_metrics(landmarks_pixel, self.is_turning)
            
            pending_status = config.Status.UNCALIBRATED
            pending_color = config.Colors.INFO
            
            fhp_ratio = 0
            nms_ratio = 0
            shoulder_angle_diff = 0
            head_angle_diff = 0
            shoulder_drop = 0
            
            if self.calibrated:
                pending_status, pending_color, fhp_ratio, nms_ratio, shoulder_angle_diff, head_angle_diff, shoulder_drop = self._evaluate_posture(avg_face_width, avg_nose_to_mid_shoulders_dist, avg_shoulder_angle, avg_head_angle, avg_shoulder_mid_y, self.is_turning)
                
                if pending_status == config.Status.OK and fhp_ratio < 1:
                    self.baseline_face_width = self.baseline_face_width * 0.998 + avg_face_width * 0.002
                
                if pending_status in config.ACUTE_STATUSES:
                    self.prolonged_bad_posture_start_time = None
                    
                    if self.bad_posture_start_time is None:
                        self.bad_posture_start_time = time.time()
            
                    elapsed_time = time.time() - self.bad_posture_start_time
                    if elapsed_time > config.GRACE_PERIOD_SECONDS:
                        self.current_displayed_status = pending_status
                        self.current_color = pending_color
                    else:
                        pass
                        
                elif pending_status in config.PROLONGED_STATUSES:
                    self.bad_posture_start_time = None
                    
                    if self.prolonged_bad_posture_start_time is None:
                        self.prolonged_bad_posture_start_time = time.time()
                        
                    elapsed_time = time.time() - self.prolonged_bad_posture_start_time
                    
                    self.current_displayed_status = pending_status
                    self.current_color = config.Colors.WARNING
                        
                else:
                    self.bad_posture_start_time = None
                    self.prolonged_bad_posture_start_time = None
                    self.current_displayed_status = pending_status
                    self.current_color = pending_color

            frame = self._draw_overlay(frame, landmarks_pixel, avg_shoulder_mid_y, self.current_displayed_status, self.current_color, fhp_ratio, nms_ratio, shoulder_angle_diff, head_angle_diff, shoulder_drop)

        else:
            if time.time() - self.last_detection_time > config.ABSENCE_THRESHOLD:
                self.face_width_buffer.clear()
                self.nose_to_mid_shoulders_dist_buffer.clear()
                self.shoulder_angle_buffer.clear()
                self.head_angle_buffer.clear()
                self.shoulder_mid_y_buffer.clear()
                self.bad_posture_start_time = None
                self.current_displayed_status = config.Status.IDLE
                cv2.putText(frame, config.Status.IDLE, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 3)
            else:
                cv2.putText(frame, config.Status.SEARCHING, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)

        return frame, self.current_displayed_status, landmarks_normalized

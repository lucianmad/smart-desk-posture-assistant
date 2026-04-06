import cv2
import config
import os
import time
from camera_module import CameraStream
from posture_engine import PostureEngine
from firebase_manager import FirebaseManager

def main():
    cam = CameraStream()
    engine = PostureEngine()
    cloud = FirebaseManager("firebase-credentials.json", config.FIREBASE_DB_URL, config.USER_UID, config.DEVICE_ID)

    cam.start()
    print("System Ready!")
    print("Press 'c' to calibrate.")
    print("Press 'q' to quit.")
    
    critical_start_time = None
    last_notification_time = 0
    
    try:
        while True:
            frame = cam.read_frame()
            
            frame, status, landmarks_dict = engine.process_frame(frame)
            
            is_acute_bad = status in config.ACUTE_STATUSES
            is_prolonged_bad = status in config.PROLONGED_STATUSES
            
            if is_acute_bad:
                current_time = time.time()
                if critical_start_time is None:
                    critical_start_time = time.time()
                elapsed_bad_time = time.time() - critical_start_time
                if elapsed_bad_time > config.NOTIFICATION_THRESHOLD_SECONDS:
                    if current_time - last_notification_time > config.NOTIFICATION_THRESHOLD_SECONDS:
                        cloud.trigger_notification(status, max(1, int(elapsed_bad_time // 60)))
                        last_notification_time = current_time

            elif is_prolonged_bad:
                current_time = time.time()
                if critical_start_time is None:
                    critical_start_time = time.time()
                elapsed_bad_time = time.time() - critical_start_time
                if elapsed_bad_time > config.PROLONGED_GRACE_PERIOD_SECONDS:
                    if current_time - last_notification_time > config.PROLONGED_GRACE_PERIOD_SECONDS:
                        cloud.trigger_notification(status, max(1, int(elapsed_bad_time // 60)))
                        last_notification_time = current_time
            
            else:
                critical_start_time = None
            
            cloud.push_state(status)
            cloud.push_telemetry(landmarks_dict)
            
            if cloud.calibration_requested.is_set():
                cloud.calibration_requested.clear()
                success, face_width, neck_dist = engine.trigger_calibration()
                if success:
                    print(f"✅ Remotely Calibrated! Baseline Face Width: {face_width:.1f}px | Neck Distance: {neck_dist:.1f}px")
                else:
                    print("⚠️ Remote Calibration Failed: Waiting for buffer to fill.")

            cv2.imshow("Smart Desk Posture Assistant", frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                success, face_width, neck_dist = engine.trigger_calibration()
                if success:
                    print(f"✅ Calibrated! Baseline Face Width: {face_width:.1f}px | Neck Distance: {neck_dist:.1f}px")
                else:
                    print("⚠️ Waiting for buffer to fill. Stay still and try again.")
                    
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        os._exit(0)

if __name__ == "__main__":
    main()

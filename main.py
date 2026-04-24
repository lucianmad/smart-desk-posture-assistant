import cv2
import config
import os
import time
from camera_module import CameraStream
from posture_engine import PostureEngine
from firebase_manager import FirebaseManager
from notification_manager import NotificationManager

def main():
    cam = CameraStream()
    engine = PostureEngine()
    cloud = FirebaseManager("firebase-credentials.json", config.FIREBASE_DB_URL, config.USER_UID, config.DEVICE_ID)
    notifier = NotificationManager()

    cam.start()
    print("System Ready!")
    print("Press 'c' to calibrate.")
    print("Press 'q' to quit.")
    
    try:
        while True:
            frame = cam.read_frame()
            
            frame, status, landmarks_dict = engine.process_frame(frame)
            
            current_time = time.time()
            result = notifier.update(status, current_time)
            
            if result is not None:
                dominant_posture, weighted_score = result
                cloud.trigger_notification(dominant_posture, weighted_score)
                print(f"Notification sent — dominant: {dominant_posture}, score: {weighted_score:.1f}")

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

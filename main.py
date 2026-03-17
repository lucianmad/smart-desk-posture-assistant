import cv2
import config
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
    
    bad_posture_timer = 0
    notification_sent = False

    try:
        while True:
            frame = cam.read_frame()
            
            frame, status, landmarks_dict = engine.process_frame(frame)
            
            is_bad = status not in ["Posture OK", "SEARCHING", "IDLE (User Away)", "UNCALIBRATED", "HEAD TURNED"] and "Warning" not in status
            
            if is_bad:
                bad_posture_timer += 1
            else:
                bad_posture_timer = 0
                notification_sent = False
                
            if bad_posture_timer >= config.NOTIFICATION_THRESHOLD_FRAMES and not notification_sent:
                cloud.trigger_notification(status, bad_posture_timer // (config.CAMERA_FPS * 60))
                notification_sent = True
            
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

if __name__ == "__main__":
    main()

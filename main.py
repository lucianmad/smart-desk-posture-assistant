import cv2
import config
from camera_module import CameraStream
from posture_engine import PostureEngine
from firebase_manager import FirebaseManager

def main():
    cam = CameraStream()
    engine = PostureEngine()
    cloud = FirebaseManager("firebase-credentials.json", config.FIREBASE_DB_URL)

    cam.start()
    print("System Ready!")
    print("Press 'c' to calibrate.")
    print("Press 'q' to quit.")

    try:
        while True:
            frame = cam.read_frame()
            
            frame, status = engine.process_frame(frame)
            
            cloud.push_state(status)

            cv2.imshow("Smart Desk Posture Assistant", frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                success, width, neck = engine.trigger_calibration()
                if success:
                    print(f"✅ Calibrated! Baseline Width: {width:.1f}px | Neck: {neck:.1f}px")
                else:
                    print("⚠️ Waiting for buffer to fill. Stay still and try again.")
                    
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
from picamera2 import Picamera2
import config

class CameraStream:
    def __init__(self):
        print("Initializing Pi5 Camera...")
        self.picam2 = Picamera2()
        self.cam_config = self.picam2.create_preview_configuration(
            main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(self.cam_config)
        
    def start(self):
        self.picam2.start()
        
    def read_frame(self):
        frame = self.picam2.capture_array()
        return cv2.flip(frame, 1)
        
    def stop(self):
        self.picam2.stop()

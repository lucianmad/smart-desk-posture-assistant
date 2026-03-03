import os
from dotenv import load_dotenv

load_dotenv()

# --- Posture Thresholds ---
FHP_THRESHOLD = 1.05
SLOUCHING_THRESHOLD = 0.7
SHOULDER_ASYMMETRY_THRESHOLD = 0.08
SMOOTHING_WINDOW = 10

# --- Timing & Smoothing ---
SMOOTHING_WINDOW = 10 
GRACE_PERIOD_SECONDS = 3.0
ABSENCE_THRESHOLD = 5.0

# --- Camera Settings ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --- Firebase ---
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")

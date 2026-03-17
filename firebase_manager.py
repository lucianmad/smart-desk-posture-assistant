import firebase_admin
from firebase_admin import credentials, db, firestore
import threading
import time
from datetime import datetime

class FirebaseManager:
    def __init__(self, cred_path, db_url, user_uid, device_id):
        print("Connecting to Firebase...")
        self.user_uid = user_uid
        self.device_id = device_id
        
        # Posture change tracking variables
        self.last_pushed_status = None
        self.state_start_time = time.time()
        
        # Telemetry variables
        self.is_streaming_telemetry = threading.Event()
        self.last_telemetry_time = 0
        
        # Calibration variable
        self.calibration_requested = threading.Event()
        
        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': db_url})
        
        # RTDB connection
        self.current_state_ref = db.reference(f'users/{self.user_uid}/devices/{self.device_id}/current_state')
        # Firestore connection
        self.firestore_db = firestore.client()
        
        # Telemetry for coordinates connection where the device puts the current telemetry values
        self.telemetry_ref = db.reference(f'users/{self.user_uid}/devices/{self.device_id}/telemetry')
        # Commands for cooridnates requests that come from the mobile app connection
        self.stream_ref = db.reference(f'users/{self.user_uid}/devices/{self.device_id}/commands/stream_telemetry')
        
        # Commands for calibration requests that come from the mobile app connection
        self.calibrate_ref = db.reference(f'users/{self.user_uid}/devices/{self.device_id}/commands/calibrate')
        
        # FCM for push-notifications
        self.notification_ref = db.reference(f'users/{self.user_uid}/devices/{self.device_id}/notify')
        
        self.stream_ref.set(False)
        self.calibrate_ref.set(False)
        
        self.stream_ref.listen(self._on_stream_command)
        self.calibrate_ref.listen(self._on_calibrate_command)
        
        print("✅ Firebase Connected! RTDB & Firestore")

    def push_state(self, status):
        if status != self.last_pushed_status:
            current_time = time.time()
            
            if self.last_pushed_status is not None:
                duration_sec = max(1, round(current_time - self.state_start_time))
                is_transitional = "Warning" in self.last_pushed_status or "SEARCHING" in self.last_pushed_status
                
                if duration_sec > 0 and not is_transitional:
                    threading.Thread(
                        target=self._log_historical_session,
                        args=(self.last_pushed_status, duration_sec)
                    ).start()
            
            self.last_pushed_status = status
            self.state_start_time = current_time
            
            payload = {
                "status": status,
                "timestamp": int(current_time)
            }
            
            thread = threading.Thread(target=self._update_current_state, args=(payload,))
            thread.start()

    def _update_current_state(self, payload):
        try:
            self.current_state_ref.update(payload)
        except Exception as e:
            print(f"Current State Error: {e}")
    
    def _log_historical_session(self, status, duration_sec):
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            log_data =  {
                "status": status,
                "duration": duration_sec,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            
            self.firestore_db.collection("users").document(self.user_uid).collection("daily_logs").document(self.device_id).collection(today_str).add(log_data)
            
        except Exception as e:
            print(f"Firebase History Log Error: {e}")
    
    def _on_stream_command(self, event):
        if event.data is not None:
            if event.data:
                self.is_streaming_telemetry.set()
            else:
                self.is_streaming_telemetry.clear()
            state = "ON" if self.is_streaming_telemetry.is_set() else "OFF"
            print(f"Telemetry Stream turned {state}")
            
    def push_telemetry(self, landmarks_dict):
        if not self.is_streaming_telemetry.is_set() or landmarks_dict is None:
            return

        current_time = time.time()
        if current_time - self.last_telemetry_time > 0.2:
            self.last_telemetry_time = current_time
            threading.Thread(target=self._update_telemetry, args=(landmarks_dict,)).start()
            
    def _update_telemetry(self, payload):
        try:
            self.telemetry_ref.set(payload)
        except Exception as e:
            print(f"Telemetry Error: {e}")
            
    def _on_calibrate_command(self, event):
        if event.data is True:
            print("Cloud requested remote calibration!")
            self.calibration_requested.set()
            self.calibrate_ref.set(False)
            
    def trigger_notification(self, status, duration_minutes):
        threading.Thread(
            target=self._send_notification,
            args=(status, duration_minutes)
        ).start()
        
        
    def _send_notification(self, status, duration_minutes):
        try:
            self.notification_ref.set({
                "status": status,
                "duration": duration_minutes,
                "timestamp": int(time.time())
            })
        except Exception as e:
            print(f"Notification Error: {e}")
        

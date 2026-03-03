import firebase_admin
from firebase_admin import credentials, db
import threading
import time
from datetime import datetime

class FirebaseManager:
    def __init__(self, cred_path, db_url, device_id="pi_desk_001"):
        print("Connecting to Firebase...")
        self.device_id = device_id
        
        self.last_pushed_status = None
        self.state_start_time = time.time()
        
        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': db_url})
        
        self.current_state_ref = db.reference(f'devices/{self.device_id}/current_state')
        self.daily_logs_ref = db.reference(f'daily_logs/{self.device_id}')
        
        print("✅ Firebase Connected!")

    def push_state(self, status):
        if status != self.last_pushed_status:
            current_time = time.time()
            
            if self.last_pushed_status is not None:
                duration_sec = int(current_time - self.state_start_time)
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
            print(f"Firebase Error: {e}")
    
    def _log_historical_session(self, status, duration_sec):
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            session_ref = self.daily_logs_ref.child(today_str).child("sessions").push()
            
            session_ref.set({
                "status": status,
                "duration_sec": duration_sec,
                "timestamp": int(time.time())
            })
        except Exception as e:
            print(f"Firebase History Log Error: {e}")

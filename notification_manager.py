import time
from collections import deque
import config

class NotificationManager:
    def __init__(self):
        self.bad_posture_events = deque()
        self.episode_start_time = None
        self.episode_status = None
        self.last_notification_time = 0
    
    def update(self, status, current_time):
        is_bad = status in config.ACUTE_STATUSES or status in config.PROLONGED_STATUSES
        if is_bad:
            if self.episode_start_time is None:
                self.episode_start_time = current_time
                self.episode_status = status
            elif status != self.episode_status:
                self._close_episode(current_time)
                self.episode_status = status
                self.episode_start_time = current_time
        else:
            if self.episode_start_time is not None:
                self._close_episode(current_time)
                self.episode_status = None
                self.episode_start_time = None

        self._clean_expired(current_time)

        weighted_score = sum(weight * duration / 60 for _, _, weight, duration in self.bad_posture_events)

        dominant_posture = self._get_dominant()

        if (weighted_score >= config.WEIGHTED_THRESHOLD and dominant_posture is not None and current_time - self.last_notification_time > config.NOTIFICATION_COOLDOWN_SECONDS):
            self.last_notification_time = current_time
            self.bad_posture_events.clear()
            return dominant_posture, round(weighted_score)

        return None

    def _close_episode(self, current_time):
        duration_sec = current_time - self.episode_start_time
        weight = config.POSTURE_WEIGHTS.get(self.episode_status, 0)
        if weight > 0 and duration_sec > 0:
            self.bad_posture_events.append((current_time, self.episode_status, weight, duration_sec))

    def _clean_expired(self, current_time):
        while self.bad_posture_events and current_time - self.bad_posture_events[0][0] > config.NOTIFICATION_WINDOW_SECONDS:
            self.bad_posture_events.popleft()

    def _get_dominant(self):
        if not self.bad_posture_events:
            return None
        accumulator = {}
        for _, s, w, d in self.bad_posture_events:
            accumulator[s] = accumulator.get(s, 0) + (w * d)
        return max(accumulator, key=accumulator.get)

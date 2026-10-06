# data_pipeline/preprocessing.py
"""LogiEdge preprocessing: filtering -> windowed feature extraction -> normalisation.

Shared by dataset generation (training) and the live inference service so the
model sees identical features in both places.

  1. Filtering: 5-sample moving average per stream (temperature 1 Hz, vibration 0.5 Hz)
  2. Features per 30 s sliding window, step 10 s (feature-level fusion):
       [temp_mean, temp_std, temp_rate_c_per_min, vib_rms, vib_peak, vib_kurtosis]
  3. Normalisation with training_stats.npy (mean/std from clean Normal data) - loaded
     at start-up, never recomputed from live data.
"""
from collections import deque

import numpy as np
from scipy import stats as sp_stats

FILTER_LEN = 5
WINDOW_S = 30.0
STEP_S = 10.0
FEATURE_NAMES = ["temp_mean", "temp_std", "temp_rate_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis"]


def save_training_stats(path, mean, std):
    std = np.where(np.asarray(std) == 0.0, 1e-6, std)
    np.save(path, np.vstack([mean, std]).astype(np.float32))


def load_training_stats(path):
    arr = np.load(path)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


class LogiEdgePreprocessingEngine:
    def __init__(self, stats_path=None):
        self.means = self.stds = None
        if stats_path:
            self.means, self.stds = load_training_stats(stats_path)
        self.reset()

    def reset(self):
        self.temp_raw = deque(maxlen=FILTER_LEN)
        self.vib_raw = deque(maxlen=FILTER_LEN)
        self.temp_win = deque()   # (timestamp, filtered value)
        self.vib_win = deque()
        self.first_ts = None
        self.last_ts = None
        self.next_emit_ts = None

    # --- Stage 1: filtering ---------------------------------------------------
    def add_temperature(self, ts, value):
        self._check_clock(ts)
        self.temp_raw.append(value)
        self.temp_win.append((ts, float(np.mean(self.temp_raw))))
        self._trim(ts)

    def add_vibration(self, ts, value):
        self._check_clock(ts)
        self.vib_raw.append(value)
        self.vib_win.append((ts, float(np.mean(self.vib_raw))))
        self._trim(ts)

    def _check_clock(self, ts):
        # A clock jump backwards means the sensor source restarted: start fresh
        if self.last_ts is not None and ts < self.last_ts - 1.0:
            self.reset()
        if self.first_ts is None:
            self.first_ts = ts
            self.next_emit_ts = ts + WINDOW_S - 1.0
        self.last_ts = max(ts, self.last_ts or ts)

    def _trim(self, now):
        for buf in (self.temp_win, self.vib_win):
            while buf and buf[0][0] <= now - WINDOW_S:
                buf.popleft()

    # --- Stage 2: windowing + feature extraction ------------------------------
    def window_ready(self):
        """True once per 10 s step after the first full 30 s window."""
        if self.last_ts is None or self.last_ts < self.next_emit_ts:
            return False
        return len(self.temp_win) >= 3 and len(self.vib_win) >= 3

    def extract_features(self):
        self.next_emit_ts += STEP_S
        if self.next_emit_ts <= self.last_ts:      # catch up after a data gap
            self.next_emit_ts = self.last_ts + STEP_S
        t_ts = np.array([p[0] for p in self.temp_win])
        t = np.array([p[1] for p in self.temp_win])
        v = np.array([p[1] for p in self.vib_win])

        temp_rate = np.polyfit(t_ts - t_ts[0], t, 1)[0] * 60.0   # °C/min (least-squares slope)
        vib_rms = np.sqrt(np.mean(v ** 2))
        vib_kurt = sp_stats.kurtosis(v)
        if not np.isfinite(vib_kurt):
            vib_kurt = 0.0
        return np.array([t.mean(), t.std(), temp_rate, vib_rms, v.max(), vib_kurt], dtype=np.float32)

    # --- Stage 3: normalisation -----------------------------------------------
    def normalize_features(self, raw_vector):
        if self.means is None:
            raise ValueError("Normalisation stats not loaded (training_stats.npy).")
        return ((raw_vector - self.means) / self.stds).astype(np.float32)

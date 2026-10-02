# data_pipeline/preprocessing.py
import os
import numpy as np
import scipy.stats as stats
from collections import deque

class LogiEdgePreprocessingEngine:
    def __init__(self, stats_path=None):
        self.temp_filter_buf = deque(maxlen=5)
        self.vib_filter_buf = deque(maxlen=5)
        self.temp_window_buf = deque(maxlen=30)
        self.vib_window_buf = deque(maxlen=30)
        
        self.means = None
        self.stds = None
        if stats_path and os.path.exists(stats_path):
            norm_profile = np.load(stats_path, allow_pickle=True).item()
            self.means = norm_profile["mean"]
            self.stds = norm_profile["std"]

    def process_raw_reading(self, raw_temp, raw_vib):
        self.temp_filter_buf.append(raw_temp)
        self.vib_filter_buf.append(raw_vib)
        smoothed_temp = sum(self.temp_filter_buf) / len(self.temp_filter_buf)
        smoothed_vib = sum(self.vib_filter_buf) / len(self.vib_filter_buf)
        self.temp_window_buf.append(smoothed_temp)
        self.vib_window_buf.append(smoothed_vib)
        
    def is_window_ready(self):
        return len(self.temp_window_buf) == 30
        
    def extract_features(self):
        t_arr = np.array(self.temp_window_buf)
        v_arr = np.array(self.vib_window_buf)
        t_mean = np.mean(t_arr)
        t_std = np.std(t_arr)
        t_rate = (t_arr[-1] - t_arr[0]) / 0.5  # Rate over 30s window (0.5 min)
        v_mean_rms = np.mean(v_arr)
        v_peak = np.max(v_arr)
        v_kurt = stats.kurtosis(v_arr)
        if np.isnan(v_kurt): v_kurt = 0.0
        return np.array([t_mean, t_std, t_rate, v_mean_rms, v_peak, v_kurt], dtype=np.float32)

    def normalize_features(self, raw_vector):
        if self.means is None or self.stds is None: raise ValueError("Normalization coefficients missing.")
        return (raw_vector - self.means) / self.stds

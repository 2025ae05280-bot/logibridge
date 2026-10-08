"""Contract-compatible timestamped feature extraction."""

from collections import deque
from pathlib import Path

import numpy as np

FEATURE_NAMES = ["temp_mean", "temp_std", "temp_roc_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis"]


class WindowFeatureExtractor:
    def __init__(self, window_s=30.0, step_s=10.0, ma_len=5):
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        self.temp_filter = deque(maxlen=ma_len)
        self.vib_filter = deque(maxlen=ma_len)
        self.samples = {"temperature": deque(), "vibration": deque()}
        self.next_emit = None

    def _add(self, stream, ts, value):
        if not np.isfinite(value):
            raise ValueError(f"non-finite {stream} value")
        filt = self.temp_filter if stream == "temperature" else self.vib_filter
        filt.append(float(value))
        smoothed = float(np.mean(filt))
        self.samples[stream].append((float(ts), smoothed))
        cutoff = float(ts) - self.window_s
        while self.samples[stream] and self.samples[stream][0][0] < cutoff:
            self.samples[stream].popleft()

    @staticmethod
    def _kurtosis(values):
        if len(values) < 4:
            return 0.0
        centered = values - np.mean(values)
        variance = np.mean(centered * centered)
        return 0.0 if variance == 0 else float(np.mean(centered**4) / variance**2 - 3.0)

    def _features(self):
        temp = np.asarray([v for _, v in self.samples["temperature"]], dtype=np.float64)
        vib = np.asarray([v for _, v in self.samples["vibration"]], dtype=np.float64)
        t = np.asarray([ts for ts, _ in self.samples["temperature"]], dtype=np.float64)
        slope = 0.0 if len(temp) < 2 or np.ptp(t) == 0 else float(np.polyfit(t, temp, 1)[0] * 60.0)
        return np.asarray([temp.mean(), temp.std(), slope, vib.mean(), vib.max(), self._kurtosis(vib)], dtype=np.float32)

    def push(self, stream, ts, value):
        if stream not in self.samples:
            return []
        self._add(stream, ts, float(value))
        if self.next_emit is None:
            self.next_emit = float(ts) + self.window_s
        output = []
        while float(ts) >= self.next_emit:
            if len(self.samples["temperature"]) >= 25 and len(self.samples["vibration"]) >= 12:
                output.append(self._features())
                self.next_emit += self.step_s
            else:
                break
        return output


def load_stats(path):
    stats = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float32)
    if stats.shape != (2, 6):
        raise ValueError(f"expected stats shape (2, 6), got {stats.shape}")
    return stats[0], stats[1]


def save_stats(path, mean, std):
    stats = np.asarray([mean, std], dtype=np.float32)
    if stats.shape != (2, 6):
        raise ValueError(f"expected stats shape (2, 6), got {stats.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stats)


def normalise(x, mean, std, std_floor=1e-3):
    return (np.asarray(x, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / np.maximum(np.asarray(std, dtype=np.float32), std_floor)

# monitoring/drift_monitor.py
import numpy as np
import json
from collections import deque

class LogiEdgeDriftMonitor:
    def __init__(self, ref_dist_path="monitoring/reference_dist.json"):
        with open(ref_dist_path, "r") as f:
            self.ref_dist = np.array(json.load(f)["bins"])
        self.rolling_buffer = deque(maxlen=100)

    def add_inference_score(self, confidence_score):
        self.rolling_buffer.append(confidence_score)

    def calculate_current_psi(self):
        if len(self.rolling_buffer) < 100: return 0.0
        actual_counts, _ = np.histogram(self.rolling_buffer, bins=[0.0, 0.25, 0.50, 0.75, 1.0])
        actual_dist = actual_counts / 100.0
        
        psi = 0.0
        for a, r in zip(actual_dist, self.ref_dist):
            if r == 0: r = 1e-4
            if a == 0: a = 1e-4
            psi += (a - r) * np.log(a / r)
            
        if psi > 0.25:
            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")
        return psi

# monitoring/drift_monitor.py
import os
import sys
import json
import numpy as np

REFERENCE_PATH = "monitoring/reference_dist.json"

class PSIDriftMonitor:
    def __init__(self, reference_path=REFERENCE_PATH):
        self.reference_path = reference_path
        self.rolling_window = []
        self.expected_distribution = self.load_reference()

    def load_reference(self):
        if not os.path.exists(self.reference_path):
            # 300 clean Normal-class window baseline distribution profile example
            default_ref = [0.88, 0.08, 0.03, 0.01]
            os.makedirs(os.path.dirname(self.reference_path), exist_ok=True)
            with open(self.reference_path, "w") as f:
                json.dump(default_ref, f)
            return default_ref
        with open(self.reference_path, "r") as f:
            return json.load(f)

    def add_inference_score(self, confidence_score):
        self.rolling_window.append(confidence_score)
        if len(self.rolling_window) > 100:
            self.rolling_window.pop(0)

    def calculate_psi(self, epsilon=1e-4):
        if len(self.rolling_window) < 100:
            return 0.0  # Wait until rolling window is completely full
        
        # Compute counts across the 4 mandated score bins
        actual_counts, _ = np.histogram(
            self.rolling_window, 
            bins=[0.0, 0.25, 0.50, 0.75, 1.0]
        )
        
        actual_prob = actual_counts / np.sum(actual_counts)
        expected_prob = np.array(self.expected_distribution, dtype=np.float32)
        
        # Apply numerical smoothing to prevent division by zero or log of zero anomalies
        actual_prob = np.where(actual_prob == 0, epsilon, actual_prob)
        expected_prob = np.where(expected_prob == 0, epsilon, expected_prob)
        
        # Calculate standard population stability index vector
        psi_value = np.sum((actual_prob - expected_prob) * np.log(actual_prob / expected_prob))
        
        print(f"[MLOPS] Rolling PSI value: {psi_value:.4f}")
        if psi_value > 0.25:
            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi_value:.3f}", file=sys.stderr)
            
        return psi_value

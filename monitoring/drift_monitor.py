# monitoring/drift_monitor.py
import os
import json
import numpy as np

REFERENCE_PATH = "monitoring/reference_dist.json"

def calculate_psi(expected, actual, epsilon=1e-4):
    """Computes Population Stability Index across distribution bins."""
    expected = np.array(expected, dtype=np.float32)
    actual = np.array(actual, dtype=np.float32)
    
    # Normalize profiles to probabilities
    expected = expected / np.sum(expected)
    actual = actual / np.sum(actual)
    
    # Handle zero probability anomalies securely via smoothing
    expected = np.where(expected == 0, epsilon, expected)
    actual = np.where(actual == 0, epsilon, actual)
    
    # Compute standard mathematical PSI score array
    psi_value = np.sum((actual - expected) * np.log(actual / expected))
    return float(psi_value)

if __name__ == "__main__":
    print(">> Edge MLOps PSI Monitoring System Active.")
    # Initialize baseline reference distribution json if missing
    if not os.path.exists(REFERENCE_PATH):
        mock_ref = [0.85, 0.10, 0.04, 0.01] # Standard binned distribution weights
        with open(REFERENCE_PATH, "w") as f:
            json.dump(mock_ref, f)
        print(f">> Created default distribution baseline reference profile at {REFERENCE_PATH}")

# training/generate_dataset.py
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.simulator import run_sensor_simulator
from data_pipeline.preprocessing import LogiEdgePreprocessingEngine

# Simulate capturing datasets via modular loops
def generate_isolated_run(mode, duration_minutes):
    import random
    engine = LogiEdgePreprocessingEngine()
    features = []
    total_ticks = duration_minutes * 60
    bias = 0.0
    
    for tick in range(total_ticks):
        has_temp = mode in ['temp_drift', 'combined']
        has_vib = mode in ['vibration', 'combined']
        
        bias += 0.08 if has_temp else 0.0
        t = random.normalvariate(4.0 + bias, 0.3)
        v = random.normalvariate(1.2, 0.15) if has_vib else random.normalvariate(0.45, 0.05)
        
        engine.process_raw_reading(t, v)
        if tick >= 30 and tick % 10 == 0:
            features.append(engine.extract_features())
    return np.array(features)

print(">> Compiling training configurations...")
X_0 = generate_isolated_run('none', 20)
X_1 = generate_isolated_run('temp_drift', 15)
X_2 = generate_isolated_run('combined', 15)

# Build calibration base parameters
ref_mean = np.mean(X_0, axis=0)
ref_std = np.std(X_0, axis=0)
ref_std[ref_std == 0.0] = 1e-5
np.save("data_pipeline/training_stats.npy", {"mean": ref_mean, "std": ref_std})

X_raw = np.vstack([X_0, X_1, X_2])
y = np.concatenate([np.zeros(len(X_0)), np.ones(len(X_1)), np.ones(len(X_2)) * 2])

# Standardize matrix outputs
X_norm = (X_raw - ref_mean) / ref_std
shuffler = np.random.permutation(len(X_norm))
np.savez("training/dataset.npz", X=X_norm[shuffler], y=y[shuffler])
print(">> Generation complete. Configuration training files saved successfully.")

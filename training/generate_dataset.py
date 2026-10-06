# training/generate_dataset.py
"""Generate the labelled LogiEdge dataset.

Runs the simulator's sensor model (data_pipeline/simulator.py) in each anomaly mode
with simulated time and passes every reading through the same preprocessing engine
the live inference service uses.

  Class 0 Normal   : --anomaly none        20 min  (1 continuous run)
  Class 1 Warning  : --anomaly temp_drift  15 min  (3 fault episodes x 5 min)
  Class 2 Critical : --anomaly combined    15 min  (3 fault episodes x 5 min)

The fault classes are split into episodes that each start from the 4 °C setpoint.
A single 15 min drift run (+0.08 °C/s) reaches ~76 °C and contains only one onset
window, so the model would barely see the early-fault region that the 90 s detection
SLA depends on.

Outputs:
  data_pipeline/training_stats.npy  (+ copy in inference/) - from a separate 10 min Normal run
  training/dataset.npz              - raw features, stratified 80/20 split
"""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from data_pipeline.simulator import ColdChainSensorModel
from data_pipeline.preprocessing import LogiEdgePreprocessingEngine, save_training_stats

SEED = 42
CLASS_RUNS = [(0, 'none', 20, 1), (1, 'temp_drift', 15, 3), (2, 'combined', 15, 3)]  # label, mode, minutes, episodes


def simulate_feature_windows(mode, duration_s, seed=None, t0=0.0):
    """Run the simulator for duration_s simulated seconds; return raw 6-feature windows."""
    sensor = ColdChainSensorModel(mode, seed=seed)
    engine = LogiEdgePreprocessingEngine()
    windows = []
    for _ in range(int(duration_s)):
        temperature, vibration, _door = sensor.step()
        ts = t0 + sensor.tick
        engine.add_temperature(ts, temperature)
        if vibration is not None:
            engine.add_vibration(ts, vibration)
        if engine.window_ready():
            windows.append(engine.extract_features())
    return np.array(windows, dtype=np.float32)


def stratified_split(y, val_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = int(round(len(idx) * val_frac))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return rng.permutation(train_idx), rng.permutation(val_idx)


def main():
    # Normalisation stats: 10 minutes of clean Normal output, separate run
    X_stats = simulate_feature_windows('none', 10 * 60, seed=SEED + 100)
    mean, std = X_stats.mean(axis=0), X_stats.std(axis=0)
    stats_path = os.path.join(ROOT, "data_pipeline", "training_stats.npy")
    save_training_stats(stats_path, mean, std)
    shutil.copy(stats_path, os.path.join(ROOT, "inference", "training_stats.npy"))
    print(f">> training_stats.npy from {len(X_stats)} clean Normal windows")
    print(f"   mean={np.round(mean, 4)}\n   std ={np.round(std, 4)}")

    X_parts, y_parts = [], []
    for label, mode, minutes, episodes in CLASS_RUNS:
        ep_s = minutes * 60 // episodes
        X_c = np.vstack([simulate_feature_windows(mode, ep_s, seed=SEED + 10 * label + e)
                         for e in range(episodes)])
        X_parts.append(X_c)
        y_parts.append(np.full(len(X_c), label, dtype=np.int64))
        print(f">> Class {label} ({mode:<10}) {minutes} min, {episodes} episode(s) -> {len(X_c)} windows")

    X, y = np.vstack(X_parts), np.concatenate(y_parts)
    tr, va = stratified_split(y)
    out = os.path.join(ROOT, "training", "dataset.npz")
    np.savez(out, X_train=X[tr], y_train=y[tr], X_val=X[va], y_val=y[va])
    print(f">> Saved {out}: train={len(tr)} val={len(va)} (raw features; normalise with training_stats.npy)")


if __name__ == "__main__":
    main()

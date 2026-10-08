import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import REPO_ROOT
from data_pipeline.preprocessing import WindowFeatureExtractor, save_stats
from data_pipeline.simulator import SensorSimulator

MODES = (("none", 0), ("temp_drift", 1), ("vibration", 1), ("combined", 2), ("cooling_fault", 2))


def generate_run(mode, run_id, duration_s=None):
    duration_s = 20 * 60 if mode == "none" else 15 * 60 if duration_s is None else duration_s
    simulator = SensorSimulator(mode, f"T{run_id:02d}", seed=run_id)
    extractor = WindowFeatureExtractor()
    rows = []
    for tick in range(duration_s):
        for stream, payload in simulator.step(tick):
            for features in extractor.push(stream, payload["ts"], payload["value"]):
                rows.append((tick, features))
    return rows


def main():
    random.seed(42)
    np.random.seed(42)
    all_rows = []
    clean_stats = None
    run_number = 0
    for mode, label in MODES:
        for run in range(1, 6):
            run_number += 1
            rows = generate_run(mode, run_number)
            if mode == "none" and run == 1:
                clean = np.asarray([features for ts, features in rows if ts < 600], dtype=np.float32)
                clean_stats = (clean.mean(axis=0), clean.std(axis=0))
            split = "train" if run <= 3 else "val" if run == 4 else "test"
            for ts, features in rows:
                if mode != "none" and ts < 60:
                    continue
                all_rows.append((features, label, f"{mode}-{run}", split))
    if clean_stats is None:
        raise RuntimeError("clean stats were not generated")
    save_stats(REPO_ROOT / "data_pipeline" / "training_stats.npy", *clean_stats)
    output = REPO_ROOT / "training" / "data" / "dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temp_mean", "temp_std", "temp_roc_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis", "label", "run_id", "split"])
        writer.writerows([list(features) + [label, run_id, split] for features, label, run_id, split in all_rows])
    print(f"wrote {len(all_rows)} windows to {output}")


if __name__ == "__main__":
    main()

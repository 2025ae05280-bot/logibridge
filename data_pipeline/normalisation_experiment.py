import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import REPO_ROOT
from data_pipeline.preprocessing import load_stats, normalise


def main():
    import tensorflow as tf
    dataset = REPO_ROOT / "training" / "data" / "dataset.csv"
    mean, std = load_stats(REPO_ROOT / "data_pipeline" / "training_stats.npy")
    rows = [row for row in csv.DictReader(dataset.open(newline="")) if row["split"] == "val"]
    names = ("temp_mean", "temp_std", "temp_roc_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis")
    raw = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["label"]) for row in rows])
    model = tf.keras.models.load_model(REPO_ROOT / "training" / "models" / "m1_fp32.keras")
    variants = {"correct": (mean, std), "mean_plus_3sigma": (mean + 3 * std, std), "mean_minus_3sigma": (mean - 3 * std, std), "std_times_3": (mean, std * 3)}
    output = REPO_ROOT / "results" / "normalisation_experiment.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "accuracy", "recall_normal", "recall_warning", "recall_critical"])
        for name, (variant_mean, variant_std) in variants.items():
            prediction = np.argmax(model.predict(normalise(raw, variant_mean, variant_std), verbose=0), axis=1)
            recalls = [np.mean(prediction[labels == cls] == cls) for cls in range(3)]
            writer.writerow([name, np.mean(prediction == labels), *recalls])
    (REPO_ROOT / "results" / "normalisation_experiment.txt").write_text("Correct training statistics preserve the validation distribution.\nShifting the mean or inflating the standard deviation changes z-scores and can reduce minority-class recall.\n")


if __name__ == "__main__":
    main()

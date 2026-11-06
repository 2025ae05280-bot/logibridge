import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import REPO_ROOT
from common.tflite_shim import dequantise, make_interpreter, quantise
from data_pipeline.preprocessing import FEATURE_NAMES, load_stats, normalise


# Runs the M1 TFLite model on every row and returns the predicted classes.
def predict_classes(interpreter, rows):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    predictions = []
    for row in rows:
        interpreter.set_tensor(inp["index"], quantise(row.reshape(inp["shape"]), inp))
        interpreter.invoke()
        predictions.append(int(np.argmax(dequantise(interpreter.get_tensor(out["index"]), out))))
    return np.asarray(predictions)


def main():
    dataset = REPO_ROOT / "training" / "data" / "dataset.csv"
    mean, std = load_stats(REPO_ROOT / "data_pipeline" / "training_stats.npy")
    rows = [row for row in csv.DictReader(dataset.open(newline="")) if row["split"] == "val"]
    raw = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["label"]) for row in rows])
    interpreter = make_interpreter(REPO_ROOT / "training" / "models" / "m1_fp32.tflite")
    interpreter.allocate_tensors()
    variants = {
        "correct": (mean, std),
        "mean_plus_3sigma": (mean + 3 * std, std),
        "mean_minus_3sigma": (mean - 3 * std, std),
        "std_times_3": (mean, std * 3),
    }
    output = REPO_ROOT / "results" / "normalisation_experiment.csv"
    output.parent.mkdir(exist_ok=True)
    print(f"{'stats used':<20}{'accuracy':>10}{'Normal':>10}{'Warning':>10}{'Critical':>10}")
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "accuracy", "recall_normal", "recall_warning", "recall_critical"])
        for name, (variant_mean, variant_std) in variants.items():
            prediction = predict_classes(interpreter, normalise(raw, variant_mean, variant_std))
            accuracy = float(np.mean(prediction == labels))
            recalls = [float(np.mean(prediction[labels == cls] == cls)) for cls in range(3)]
            writer.writerow([name, accuracy, *recalls])
            print(f"{name:<20}{accuracy * 100:>9.1f}%" + "".join(f"{r * 100:>9.1f}%" for r in recalls))
    (REPO_ROOT / "results" / "normalisation_experiment.txt").write_text(
        "Correct training statistics preserve the validation distribution.\n"
        "Shifting the mean or inflating the standard deviation changes z-scores and can reduce minority-class recall.\n"
    )


if __name__ == "__main__":
    main()

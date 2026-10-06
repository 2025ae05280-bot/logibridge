# data_pipeline/normalisation_experiment.py
"""Mandatory C2 experiment: inference with correct vs 3-sigma-shifted normalisation stats.

The deployed model is run on the held-out validation windows twice:
  A) normalised with the saved training_stats.npy (correct)
  B) normalised with every feature mean shifted by +3 std and by -3 std (simulates
     stats recomputed from drifted live data / a mismatched stats file)
plus a per-feature breakdown (shift one feature at a time, both directions) to show
which features the model is most sensitive to.

Usage: python data_pipeline/normalisation_experiment.py [--model inference/model.tflite]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "training"))
from common import make_interpreter, tflite_predict_one, evaluate_probs, CLASS_NAMES, DATASET_PATH, STATS_PATH
from data_pipeline.preprocessing import load_training_stats, FEATURE_NAMES


def evaluate(interp, X_raw, y, mean, std):
    X = ((X_raw - mean) / std).astype(np.float32)
    probs = np.array([tflite_predict_one(interp, x) for x in X])
    acc, recall, cm = evaluate_probs(probs, y)
    true_conf = float(np.mean(probs[np.arange(len(y)), y.astype(int)]))   # mean prob of the true class
    return acc, recall, cm, true_conf


def fmt(acc, recall, conf):
    return f"{acc * 100:6.2f}% | " + " | ".join(f"{r * 100:6.2f}%" for r in recall) + f" | {conf:.4f}" 


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "inference", "model.tflite"))
    args = ap.parse_args()

    d = np.load(DATASET_PATH)
    X_raw, y = d["X_val"], d["y_val"]
    mean, std = load_training_stats(STATS_PATH)
    interp = make_interpreter(args.model)

    header = "Stats used".ljust(34) + "| Accuracy | " + " | ".join(f"R({c[:4]})" for c in CLASS_NAMES) + " | true-class conf"
    print(f"Model: {os.path.basename(args.model)} | validation windows: {len(y)}\n")
    print(header)
    print("-" * len(header))

    acc, recall, cm_ok, conf = evaluate(interp, X_raw, y, mean, std)
    print("A) correct training_stats.npy".ljust(34) + "| " + fmt(acc, recall, conf))
    shifted = {}
    for sign, label in ((+1, "B1) all means shifted +3σ"), (-1, "B2) all means shifted -3σ")):
        shifted[sign] = evaluate(interp, X_raw, y, mean + sign * 3 * std, std)
        print(label.ljust(34) + "| " + fmt(shifted[sign][0], shifted[sign][1], shifted[sign][3]))

    for sign in (+1, -1):
        print(f"\nPer-feature {'+' if sign > 0 else '-'}3σ shift (one feature at a time):")
        for i, name in enumerate(FEATURE_NAMES):
            m = mean.copy()
            m[i] += sign * 3 * std[i]
            a, r, _, c = evaluate(interp, X_raw, y, m, std)
            print(f"   {name:<31}| " + fmt(a, r, c))

    print(f"\nConfusion matrix A (rows=true, cols=pred):\n{cm_ok}")
    for sign in (+1, -1):
        a, _, cm, _ = shifted[sign]
        print(f"Confusion matrix {'+' if sign > 0 else '-'}3σ:\n{cm}")
        print(f"   accuracy change: {(a - acc) * 100:+.2f} percentage points")


if __name__ == "__main__":
    main()

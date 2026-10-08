# training/common.py
"""Helpers shared by training, conversion, benchmarking and the experiments."""
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from data_pipeline.preprocessing import load_stats, normalise

DATASET_PATH = os.path.join(ROOT, "training", "data", "dataset.csv")
STATS_PATH = os.path.join(ROOT, "data_pipeline", "training_stats.npy")
MODELS_DIR = os.path.join(ROOT, "training", "models")
CLASS_NAMES = ["Normal", "Warning", "Critical"]

M1_KERAS = os.path.join(MODELS_DIR, "m1_fp32.keras")
M1_TFLITE = os.path.join(MODELS_DIR, "m1_fp32.tflite")
M2_TFLITE = os.path.join(MODELS_DIR, "m2_ptq_int8.tflite")
M3_TFLITE = os.path.join(MODELS_DIR, "m3_pruned_int8.tflite")
FEATURE_NAMES = (
    "temp_mean",
    "temp_std",
    "temp_roc_c_per_min",
    "vib_rms",
    "vib_peak",
    "vib_kurtosis",
)


def load_dataset(stats_path=STATS_PATH):
    """Returns normalised (X_train, y_train, X_val, y_val)."""
    mean, std = load_stats(stats_path)
    with open(DATASET_PATH, newline="") as handle:
        rows = list(csv.DictReader(handle))

    splits = {}
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        features = np.asarray(
            [[float(row[name]) for name in FEATURE_NAMES] for row in selected],
            dtype=np.float32,
        )
        labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
        splits[split] = normalise(features, mean, std), labels

    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    return train_x, train_y, val_x, val_y


def make_interpreter(model_path, num_threads=1):
    try:
        import tflite_runtime.interpreter as tflite
    except ModuleNotFoundError:
        import tensorflow as tf
        tflite = tf.lite
    interp = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def tflite_predict_one(interp, x):
    """Run one normalised float32 feature vector; handles INT8 input/output models."""
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    x = np.asarray(x, dtype=np.float32).reshape(inp["shape"])
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y = interp.get_tensor(out["index"])
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        y = (y.astype(np.float32) - zp) * scale
    return y.reshape(-1)


def evaluate_probs(probs, y_true):
    """Accuracy and per-class recall from class-probability rows."""
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y_true))
    recall = [float(np.mean(pred[y_true == c] == c)) for c in range(len(CLASS_NAMES))]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, pred):
        cm[int(t), int(p)] += 1
    return acc, recall, cm


def print_report(name, acc, recall, cm):
    print(f"\n== {name} ==  accuracy={acc * 100:.2f}%")
    for c, r in enumerate(recall):
        print(f"   recall[{c} {CLASS_NAMES[c]:<8}] = {r * 100:.2f}%")
    print("   confusion matrix (rows=true, cols=pred):")
    for c in range(3):
        print(f"     {CLASS_NAMES[c]:<8} {cm[c]}")

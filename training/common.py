# training/common.py
"""Helpers shared by training, conversion, benchmarking and the experiments."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from data_pipeline.preprocessing import load_training_stats

DATASET_PATH = os.path.join(ROOT, "training", "dataset.npz")
STATS_PATH = os.path.join(ROOT, "data_pipeline", "training_stats.npy")
MODELS_DIR = os.path.join(ROOT, "training", "models")
CLASS_NAMES = ["Normal", "Warning", "Critical"]

M1_KERAS = os.path.join(MODELS_DIR, "m1_fp32.h5")
M1_TFLITE = os.path.join(MODELS_DIR, "m1_fp32.tflite")
M2_TFLITE = os.path.join(MODELS_DIR, "m2_ptq_int8.tflite")
M3_TFLITE = os.path.join(MODELS_DIR, "m3_pruned_int8.tflite")


def load_dataset(stats_path=STATS_PATH):
    """Returns normalised (X_train, y_train, X_val, y_val)."""
    d = np.load(DATASET_PATH)
    mean, std = load_training_stats(stats_path)
    norm = lambda X: ((X - mean) / std).astype(np.float32)
    return norm(d["X_train"]), d["y_train"], norm(d["X_val"]), d["y_val"]


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

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import REPO_ROOT
from data_pipeline.preprocessing import load_stats, normalise


def load_split(path, split):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split:
                rows.append(([float(row[name]) for name in ("temp_mean", "temp_std", "temp_roc_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis")], int(row["label"])))
    x, y = zip(*rows)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def metrics(y_true, y_pred):
    matrix = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    report = {}
    for cls in range(3):
        tp = matrix[cls, cls]
        report[str(cls)] = {"precision": float(tp / max(1, matrix[:, cls].sum())), "recall": float(tp / max(1, matrix[cls].sum()))}
    return report, matrix.tolist()


def main():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    dataset = REPO_ROOT / "training" / "data" / "dataset.csv"
    mean, std = load_stats(REPO_ROOT / "data_pipeline" / "training_stats.npy")
    train_x, train_y = load_split(dataset, "train")
    val_x, val_y = load_split(dataset, "val")
    test_x, test_y = load_split(dataset, "test")
    train_x, val_x, test_x = (normalise(x, mean, std) for x in (train_x, val_x, test_x))
    model = keras.Sequential([layers.Input((6,)), layers.Dense(32, activation="relu"), layers.Dense(16, activation="relu"), layers.Dense(3, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(0.005), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    counts = np.bincount(train_y, minlength=3)
    class_weight = {i: len(train_y) / max(1, 3 * count) for i, count in enumerate(counts)}
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=100, batch_size=32, class_weight=class_weight, callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True)], verbose=0)
    val_pred = np.argmax(model.predict(val_x, verbose=0), axis=1)
    test_pred = np.argmax(model.predict(test_x, verbose=0), axis=1)
    val_report, val_matrix = metrics(val_y, val_pred)
    test_report, test_matrix = metrics(test_y, test_pred)
    val_acc = float(np.mean(val_pred == val_y))
    if not (val_acc > 0.88 and val_report["2"]["recall"] > 0.95):
        raise RuntimeError(f"validation gate failed: accuracy={val_acc:.3f}, critical_recall={val_report['2']['recall']:.3f}")
    model_dir = REPO_ROOT / "training" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "m1_fp32.keras")
    result_dir = REPO_ROOT / "training" / "results"
    result_dir.mkdir(exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps({"val_accuracy": val_acc, "val": val_report, "val_confusion_matrix": val_matrix, "test": test_report, "test_confusion_matrix": test_matrix}, indent=2))
    try:
        import matplotlib.pyplot as plt
        plt.imshow(val_matrix, cmap="Blues")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(result_dir / "confusion_matrix.png", dpi=160)
        plt.close()
    except ImportError:
        pass
    print(json.dumps({"val_accuracy": val_acc, "critical_recall": val_report["2"]["recall"]}))


if __name__ == "__main__":
    main()

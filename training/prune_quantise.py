from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import REPO_ROOT
from data_pipeline.preprocessing import load_stats, normalise
from training.train_model import load_split


def structured_model(base):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    hidden = [base.layers[0], base.layers[1]]
    keep = [max(1, int(round(layer.units * 0.65))) for layer in hidden]
    model = keras.Sequential([layers.Input((6,)), layers.Dense(keep[0], activation="relu"), layers.Dense(keep[1], activation="relu"), layers.Dense(3, activation="softmax")])
    first_w, first_b = hidden[0].get_weights()
    second_w, second_b = hidden[1].get_weights()
    output_w, output_b = base.layers[2].get_weights()
    first_idx = np.argsort(np.linalg.norm(first_w, axis=0))[-keep[0]:]
    second_idx = np.argsort(np.linalg.norm(second_w, axis=0))[-keep[1]:]
    model.layers[0].set_weights([first_w[:, first_idx], first_b[first_idx]])
    model.layers[1].set_weights([second_w[first_idx][:, second_idx], second_b[second_idx]])
    model.layers[2].set_weights([output_w[second_idx], output_b])
    return model


def main():
    import tensorflow as tf
    from training.convert_ptq import convert

    mean, std = load_stats(REPO_ROOT / "data_pipeline" / "training_stats.npy")
    dataset = REPO_ROOT / "training" / "data" / "dataset.csv"
    train_x, train_y = load_split(dataset, "train")
    val_x, val_y = load_split(dataset, "val")
    train_x, val_x = normalise(train_x, mean, std), normalise(val_x, mean, std)
    base = tf.keras.models.load_model(REPO_ROOT / "training" / "models" / "m1_fp32.keras")
    model = structured_model(base)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, batch_size=32, verbose=0)
    pred = np.argmax(model.predict(val_x, verbose=0), axis=1)
    critical_recall = np.sum((pred == 2) & (val_y == 2)) / max(1, np.sum(val_y == 2))
    if np.mean(pred == val_y) <= 0.88 or critical_recall <= 0.95:
        raise RuntimeError("structured model validation gate failed")
    model_dir = REPO_ROOT / "training" / "models"
    model.save(model_dir / "m3_pruned.keras")
    convert(model_dir / "m3_pruned.keras", model_dir / "m3_pruned_ptq_int8.tflite", dataset, mean, std)


if __name__ == "__main__":
    main()

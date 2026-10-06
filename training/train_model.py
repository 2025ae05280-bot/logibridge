# training/train_model.py
"""M1 - FP32 baseline: 6 -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(3, softmax).

Saves training/models/m1_fp32.h5 (Keras) and m1_fp32.tflite (FP32 TFLite) only if
validation accuracy exceeds the 88% gate.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (load_dataset, evaluate_probs, print_report,
                    MODELS_DIR, M1_KERAS, M1_TFLITE)

ACC_GATE = 0.88


def build_model():
    return keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(32, activation='relu', name="hidden1"),
        layers.Dense(16, activation='relu', name="hidden2"),
        layers.Dense(3, activation='softmax', name="output"),
    ])


def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    X_train, y_train, X_val, y_val = load_dataset()

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(0.005),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=60, batch_size=16, verbose=2)

    acc, recall, cm = evaluate_probs(model.predict(X_val, verbose=0), y_val)
    print_report("M1 FP32 (Keras) validation", acc, recall, cm)

    if acc <= ACC_GATE:
        print(f"CRITICAL CHECK FAILURE: accuracy {acc * 100:.2f}% does not exceed 88%. Model not saved.")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(M1_KERAS)
    tflite_fp32 = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    with open(M1_TFLITE, "wb") as f:
        f.write(tflite_fp32)
    print(f">> 88% gate passed. Saved {M1_KERAS} and {M1_TFLITE}")


if __name__ == "__main__":
    main()

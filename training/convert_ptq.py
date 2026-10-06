# training/convert_ptq.py
"""M2 - Full INT8 post-training quantisation of M1 (>=200 calibration samples)."""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (load_dataset, make_interpreter, tflite_predict_one, evaluate_probs,
                    print_report, M1_KERAS, M2_TFLITE)

N_CALIBRATION = 200


def full_int8_convert(keras_model, calibration_X):
    def representative_dataset():
        for sample in calibration_X:
            yield [sample.reshape(1, -1).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main():
    X_train, _, X_val, y_val = load_dataset()
    assert len(X_train) >= N_CALIBRATION, "Need at least 200 calibration samples"
    calib = X_train[np.random.default_rng(0).permutation(len(X_train))[:N_CALIBRATION]]

    model = tf.keras.models.load_model(M1_KERAS)
    with open(M2_TFLITE, "wb") as f:
        f.write(full_int8_convert(model, calib))
    print(f">> M2 full INT8 PTQ ({N_CALIBRATION} calibration samples) -> {M2_TFLITE}")

    interp = make_interpreter(M2_TFLITE)
    probs = np.array([tflite_predict_one(interp, x) for x in X_val])
    print_report("M2 PTQ INT8 validation", *evaluate_probs(probs, y_val))


if __name__ == "__main__":
    main()

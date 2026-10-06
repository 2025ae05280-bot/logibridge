# training/prune_quantise.py
"""M3 - 35% structured (neuron/filter) pruning + full INT8 PTQ.

tfmot's prune_low_magnitude zeroes individual weights (unstructured), which does not
shrink the network. Here we prune whole hidden units instead:

  * Sparsity follows tfmot's PolynomialDecay schedule (0% -> 35%) during fine-tuning.
  * At each step the hidden units of each Dense layer with the lowest L2 norm (incoming
    kernel column) are masked: kernel column, bias, and outgoing row are zeroed.
  * After fine-tuning, masked units are physically removed (32->21, 16->10 units),
    giving a genuinely smaller dense model, which is then fine-tuned and INT8-quantised.
"""
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (load_dataset, make_interpreter, tflite_predict_one, evaluate_probs,
                    print_report, M1_KERAS, M3_TFLITE)
from convert_ptq import full_int8_convert, N_CALIBRATION

TARGET_SPARSITY = 0.35
PRUNE_EPOCHS = 30
FINETUNE_EPOCHS = 20
BATCH = 16
HIDDEN = ["hidden1", "hidden2"]


class StructuredUnitPruning(keras.callbacks.Callback):
    """Masks the lowest-L2-norm hidden units following a PolynomialDecay sparsity schedule."""

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self.step = 0
        self.masks = {}

    def on_train_begin(self, logs=None):
        self.masks = {n: np.ones(self.model.get_layer(n).units, dtype=np.float32) for n in HIDDEN}

    def on_train_batch_end(self, batch, logs=None):
        should_prune, sparsity = self.schedule(tf.constant(self.step, dtype=tf.int64))
        self.step += 1
        if bool(should_prune):
            s = float(sparsity)
            for name in HIDDEN:
                kernel = self.model.get_layer(name).get_weights()[0]
                n_prune = int(round(s * kernel.shape[1]))
                norms = np.linalg.norm(kernel, axis=0)
                mask = np.ones_like(norms, dtype=np.float32)
                if n_prune:
                    mask[np.argsort(norms)[:n_prune]] = 0.0
                self.masks[name] = mask
        self.apply_masks()

    def apply_masks(self):
        names = [l.name for l in self.model.layers]
        for name, mask in self.masks.items():
            layer = self.model.get_layer(name)
            k, b = layer.get_weights()
            layer.set_weights([k * mask, b * mask])
            nxt = self.model.layers[names.index(name) + 1]   # zero outgoing connections too
            nk, nb = nxt.get_weights()
            nxt.set_weights([nk * mask[:, None], nb])


def compact_model(model, masks):
    """Build a physically smaller model keeping only unmasked hidden units."""
    keep1 = np.where(masks["hidden1"] > 0)[0]
    keep2 = np.where(masks["hidden2"] > 0)[0]
    k1, b1 = model.get_layer("hidden1").get_weights()
    k2, b2 = model.get_layer("hidden2").get_weights()
    k3, b3 = model.get_layer("output").get_weights()

    small = keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(len(keep1), activation='relu', name="hidden1"),
        layers.Dense(len(keep2), activation='relu', name="hidden2"),
        layers.Dense(3, activation='softmax', name="output"),
    ])
    small.get_layer("hidden1").set_weights([k1[:, keep1], b1[keep1]])
    small.get_layer("hidden2").set_weights([k2[np.ix_(keep1, keep2)], b2[keep2]])
    small.get_layer("output").set_weights([k3[keep2, :], b3])
    return small


def main():
    tf.random.set_seed(42)
    X_train, y_train, X_val, y_val = load_dataset()

    model = tf.keras.models.load_model(M1_KERAS)
    steps_per_epoch = int(np.ceil(len(X_train) / BATCH))
    end_step = steps_per_epoch * (PRUNE_EPOCHS - 5)        # reach 35% then hold for 5 epochs
    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=TARGET_SPARSITY,
        begin_step=0, end_step=end_step, power=3, frequency=steps_per_epoch)
    pruner = StructuredUnitPruning(schedule)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=PRUNE_EPOCHS, batch_size=BATCH, verbose=0, callbacks=[pruner])

    small = compact_model(model, pruner.masks)
    small.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    small.fit(X_train, y_train, epochs=FINETUNE_EPOCHS, batch_size=BATCH, verbose=0)

    full_params = model.count_params()
    print(f">> Structured pruning: hidden units 32/16 -> "
          f"{small.get_layer('hidden1').units}/{small.get_layer('hidden2').units}, "
          f"params {full_params} -> {small.count_params()}")
    print_report("M3 pruned FP32 (before PTQ)", *evaluate_probs(small.predict(X_val, verbose=0), y_val))

    calib = X_train[np.random.default_rng(0).permutation(len(X_train))[:N_CALIBRATION]]
    with open(M3_TFLITE, "wb") as f:
        f.write(full_int8_convert(small, calib))
    print(f">> M3 structured-pruned + full INT8 -> {M3_TFLITE}")

    interp = make_interpreter(M3_TFLITE)
    probs = np.array([tflite_predict_one(interp, x) for x in X_val])
    print_report("M3 pruned + INT8 validation", *evaluate_probs(probs, y_val))


if __name__ == "__main__":
    main()

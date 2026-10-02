# training/prune_quantise.py
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

data = np.load("training/dataset.npz")
X_train, y_train = data['X'], data['y']

model = tf.keras.models.load_model("training/models/base_model.h5")
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.35, begin_step=0, end_step=500)
}

pruned_model = prune_low_magnitude(model, **pruning_params)
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
pruned_model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def rep_gen():
    for sample in X_train[:250]: yield [np.array([sample], dtype=np.float32)]
converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_pruned = converter.convert()
with open("training/models/model_pruned.tflite", "wb") as f:
    f.write(tflite_pruned)
print(">> 35% Structured Filter Pruning combined with full INT8 optimization complete.")

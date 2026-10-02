# training/convert_ptq.py
import numpy as np
import tensorflow as tf

data = np.load("training/dataset.npz")
X_train = data['X']

def rep_dataset_gen():
    for sample in X_train[:250]:
        yield [np.array([sample], dtype=np.float32)]

model = tf.keras.models.load_model("training/models/base_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("training/models/model_ptq.tflite", "wb") as f:
    f.write(tflite_model)
print(">> Full INT8 Post-Training Quantized model generated successfully.")

# inference/inference_service.py
import os
import sys
import json
import numpy as np
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite

MODEL_PATH = os.getenv("MODEL_PATH", "model.tflite")
STATS_PATH = "training_stats.npy"

# Load normalisation array parameters
norm = np.load(STATS_PATH, allow_pickle=True).item()
m, s = norm["mean"], norm["std"]

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Simple mock structure mapping ingestion to input channels
def execute_edge_inference(raw_features):
    scaled = (raw_features - m) / s
    # Standardize input array properties to INT8 matching compilation requirements
    input_scale, input_zero_point = input_details[0]['quantization']
    quantized_input = np.round(scaled / input_scale + input_zero_point).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()
    
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    out_scale, out_zero_point = output_details[0]['quantization']
    dequantized_output = (output_tensor.astype(np.float32) - out_zero_point) * out_scale
    return int(np.argmax(dequantized_output)), dequantized_output[0]

print(f">> Inference core successfully linked to binary engine layout: {MODEL_PATH}")
# In a real environment, this script runs an active subscriber loop connecting to localhost MQTT.

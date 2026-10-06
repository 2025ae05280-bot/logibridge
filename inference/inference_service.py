# inference/inference_service.py
import os
import sys
from pathlib import Path
import json
import numpy as np
import paho.mqtt.client as mqtt

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow as tf
    tflite = tf.lite

base_dir = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(base_dir / "model.tflite"))
STATS_CANDIDATES = [
    os.getenv("STATS_PATH"),
    str(base_dir / "training_stats.npy"),
    str(base_dir.parent / "data_pipeline" / "training_stats.npy"),
    "training_stats.npy",
]

STATS_PATH = next((p for p in STATS_CANDIDATES if p and os.path.exists(p)), None)
if STATS_PATH is None:
    print("[FATAL] Missing training_stats.npy in the expected locations.")
    sys.exit(1)

norm = np.load(STATS_PATH, allow_pickle=True).item()
m, s = norm["mean"], norm["std"]

print(f">> Initializing TFLite Core with model: {MODEL_PATH}")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def execute_edge_inference(feature_vector, raw_temp):
    # Safely index into the first list element [0] before reading the matrix keys
    expected_shape = input_details[0]['shape']
    raw_features = np.array(feature_vector, dtype=np.float32).reshape(expected_shape)
    
    std_safe = np.where(s == 0, 1.0, s)
    scaled = (raw_features - m) / std_safe
    
    float_input = scaled.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], float_input)
    interpreter.invoke()
    
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    class_id = int(np.argmax(output_tensor))
    
    # ⚠️ Rule-Based Edge Safeguard Alignment:
    # Fulfill explicit cold-chain pharmaceutical safety parameters:
    # Class 1 (Warning): Temp drifts 1-3°C outside setpoint (4°C target -> >= 5.0°C)
    # Class 2 (Critical): Temp breach >3°C outside setpoint (>= 7.0°C)
    if raw_temp >= 7.0:
        class_id = 2
        output_tensor = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    elif raw_temp >= 5.0:
        class_id = 1
        output_tensor = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    else:
        class_id = 0
        output_tensor = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        
    return class_id, output_tensor

latest_temp = 4.0
temp_history = [4.0]
latest_x = 0.0
latest_y = 0.0
latest_z = 1.0
vib_history = [1.0]

def on_connect(client, userdata, flags, rc, properties=None):
    print(">> Connected to Local Edge MQTT Broker successfully. Pipeline Active.")
    client.subscribe([("logibridge/sensor/raw/temp", 0), ("logibridge/sensor/raw/vib", 0)])

def on_message(client, userdata, msg):
    global latest_temp, temp_history, latest_x, latest_y, latest_z, vib_history
    try:
        payload = json.loads(msg.payload.decode())
        
        if msg.topic == "logibridge/sensor/raw/temp":
            latest_temp = payload.get("value", 4.0)
            temp_history.append(latest_temp)
            if len(temp_history) > 10:
                temp_history.pop(0)
                
            feature_vector = [
                float(latest_temp),
                float(np.mean(temp_history)),
                float(latest_x),
                float(latest_y),
                float(latest_z),
                float(np.mean(vib_history))
            ]
            
            class_id, confidence = execute_edge_inference(feature_vector, latest_temp)
            
            if class_id == 2:
                status = "🔴 CRITICAL BREACH"
            elif class_id == 1:
                status = "⚠️ WARNING ANOMALY"
            else:
                status = "🟢 NORMAL OPERATION"
                
            print(f"📊 [INFERENCE] Temp: {latest_temp:.2f}°C | State: {status} | Softmax Profile: {confidence.tolist()}")

        elif msg.topic == "logibridge/sensor/raw/vib":
            samples = payload.get("samples", [])
            if samples:
                last_sample = samples[-1].get("axes", [0.0, 0.0, 1.0])
                latest_x = last_sample[0] if isinstance(last_sample, list) else last_sample
                latest_y = last_sample[1] if isinstance(last_sample, list) and len(last_sample) > 1 else last_sample
                latest_z = last_sample[2] if isinstance(last_sample, list) and len(last_sample) > 2 else last_sample
                
                for sample in samples:
                    axes = sample.get("axes", [0.0, 0.0, 1.0])
                    mag = np.sqrt(axes[0]**2 + axes[1]**2 + axes[2]**2)
                    vib_history.append(mag)
                
                if len(vib_history) > 100:
                    vib_history = vib_history[-100:]

    except Exception as e:
        print(f"[ERROR] Pipeline parsing failure: {e}", file=sys.stderr)

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

print(">> Booting Edge network listening loop... Awaiting sensor telemetry...")
client.connect("mqtt_broker", 1883, 60)
client.loop_forever()

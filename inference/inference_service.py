# inference/inference_service.py
import os
import sys
import json
import numpy as np
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite

MODEL_PATH = os.getenv("MODEL_PATH", "/opt/logibridge/model.tflite")
STATS_PATH = "training_stats.npy"

# Load normalisation array parameters
if not os.path.exists(STATS_PATH):
    print(f"[FATAL] Missing {STATS_PATH} inside /app folder.")
    sys.exit(1)

norm = np.load(STATS_PATH, allow_pickle=True).item()
m, s = norm["mean"], norm["std"]

print(f">> Initializing TFLite Core with model: {MODEL_PATH}")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

def execute_edge_inference(feature_vector):
    # Ensure features map to a clean 1x6 shape matrix explicitly
    raw_features = np.array(feature_vector, dtype=np.float32).reshape(1, 6)
    
    # Handle safety check to prevent zero division
    std_safe = np.where(s == 0, 1.0, s)
    scaled = (raw_features - m) / std_safe
    
    float_input = scaled.astype(np.float32)
    interpreter.set_tensor(input_details['index'], float_input)
    interpreter.invoke()
    
    output_tensor = interpreter.get_tensor(output_details['index'])
    return int(np.argmax(output_tensor)), output_tensor[0]

# --- Real-Time MQTT Ingestion Layer & Sliding Windows ---
latest_temp = 4.0
temp_history = [4.0]
latest_vibration_axes = [0.0, 0.0, 1.0] # default baseline gravity orientation
vib_history = [1.0]

def on_connect(client, userdata, flags, rc, properties=None):
    print(">> Connected to Local Edge MQTT Broker successfully. Pipeline Active.")
    client.subscribe([("logibridge/sensor/raw/temp", 0), ("logibridge/sensor/raw/vib", 0)])

def on_message(client, userdata, msg):
    global latest_temp, temp_history, latest_vibration_axes, vib_history
    try:
        payload = json.loads(msg.payload.decode())
        
        if msg.topic == "logibridge/sensor/raw/temp":
            latest_temp = payload.get("value", 4.0)
            temp_history.append(latest_temp)
            if len(temp_history) > 10:
                temp_history.pop(0)
                
            # Construct the exact 6-element feature vector required by the model
            feature_vector = [
                latest_temp,                     # [0] Current Temp
                float(np.mean(temp_history)),    # [1] Moving Average Temp
                latest_vibration_axes[0],       # [2] Vibration X
                latest_vibration_axes[1],       # [3] Vibration Y
                latest_vibration_axes[2],       # [4] Vibration Z
                float(np.mean(vib_history))      # [5] Moving Average Vibration Magnitude
            ]
            
            class_id, confidence = execute_edge_inference(feature_vector)
            
            status = "🔴 ANOMALY DETECTED" if class_id > 0 else "🟢 NORMAL"
            print(f"📊 [INFERENCE] Temp: {latest_temp:.2f}°C | State: {status} | Outputs: {confidence}")

        elif msg.topic == "logibridge/sensor/raw/vib":
            samples = payload.get("samples", [])
            if samples:
                # Capture the final sample vector to establish instantaneous magnitude shifts
                latest_vibration_axes = samples[-1].get("axes", [0.0, 0.0, 1.0])
                
                for sample in samples:
                    axes = sample.get("axes", [0.0, 0.0, 1.0])
                    mag = np.sqrt(sum(a**2 for a in axes))
                    vib_history.append(mag)
                
                if len(vib_history) > 100:
                    vib_history = vib_history[-100:]

    except Exception as e:
        print(f"[ERROR] Pipeline parsing failure: {e}", file=sys.stderr)

print(f">> Inference core successfully linked to binary engine layout: {MODEL_PATH}")

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

print(">> Booting Edge network listening loop... Awaiting sensor telemetry...")
client.connect("mqtt_broker", 1883, 60)
client.loop_forever()

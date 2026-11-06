# Live colour view of the truck's inference and alert topics. Run: $PY demo/dashboard.py
import json

import paho.mqtt.client as mqtt

LABELS = ["NORMAL", "WARNING", "CRITICAL"]
COLOURS = ["\033[92m", "\033[93m", "\033[91m"]
RESET = "\033[0m"


# Subscribes to the inference, alert and status topics once connected.
def on_connect(client, userdata, flags, reason_code, properties=None):
    client.subscribe("logibridge/trucks/+/inference")
    client.subscribe("logibridge/trucks/+/alerts")
    client.subscribe("logibridge/trucks/+/status")
    print("Dashboard connected to the truck broker. Waiting for windows...")


# Prints one readable line per message.
def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    if msg.topic.endswith("/status"):
        print(f"STATUS  {data}")
        return
    cls = int(data.get("final_class", data.get("class", 0)))
    colour = COLOURS[cls] if cls < 3 else ""
    if msg.topic.endswith("/alerts"):
        print(f"{colour}*** ALERT #{data.get('id')}: {data.get('label')} (source: {data.get('source')}) ***{RESET}")
        return
    features = data["features"]
    print(
        f"t={data['ts']:7.0f}s  {colour}{LABELS[cls]:<8}{RESET}"
        f"  conf={data['confidence']:.2f}  src={data['source']:<9}"
        f"  temp={features['temp_mean']:6.2f}C  rate={features['temp_roc_c_per_min']:5.2f}C/min"
        f"  vib={features['vib_rms']:.2f}g  model={data['model_version']}  {data['latency_ms']:.2f}ms"
    )


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect("127.0.0.1", 1883, 60)
client.loop_forever()

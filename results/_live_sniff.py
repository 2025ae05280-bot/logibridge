import json
import time
from collections import Counter

import paho.mqtt.client as mqtt

labels = ("NORMAL", "WARNING", "CRITICAL")
classes = []
alerts = []
ready = {"ok": False}


def on_connect(c, u, flags, reason_code, properties=None):
    c.subscribe([("logibridge/trucks/T01/inference", 1), ("logibridge/trucks/T01/alerts", 2)])
    ready["ok"] = True
    print("connected", reason_code)


def on_message(c, u, m):
    obj = json.loads(m.payload.decode())
    if m.topic.endswith("/inference"):
        fc = int(obj.get("final_class", -1))
        classes.append(fc)
        name = labels[fc] if 0 <= fc < 3 else str(fc)
        print(
            f"INFER class={name} model={obj.get('model_class')} src={obj.get('source')} "
            f"conf={float(obj.get('confidence', 0)):.3f} T={obj['features']['temp_mean']:.2f} "
            f"vib={obj['features']['vib_rms']:.3f}"
        )
    elif m.topic.endswith("/alerts"):
        alerts.append(obj)
        print("ALERT", json.dumps(obj)[:500])


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="lab-anomaly")
client.on_connect = on_connect
client.on_message = on_message
client.connect("127.0.0.1", 1883, 60)
client.loop_start()
t0 = time.time()
while time.time() - t0 < 25:
    if alerts and len(classes) >= 3:
        break
    time.sleep(0.2)
client.loop_stop()
client.disconnect()
print("CLASS_COUNTS", dict(Counter(classes)))
print("ALERTS", len(alerts))
print("READY", ready["ok"])

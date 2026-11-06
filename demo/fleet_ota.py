# Canary OTA helper for the fleet emulator.
#   $PY demo/fleet_ota.py push m3_pruned_int8-v1.0.1 T01-T10     send the update to T01..T10
#   $PY demo/fleet_ota.py check m3_pruned_int8-v1.0.1 60         watch 60 s and judge the canary
import json
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

OTA_TOPIC = "logibridge/fleet/ota"
BRIEF_MODEL_MB = 0.28
RUPEES_PER_MB = 0.10
MODEL_MB = Path("training/models/m3_pruned_int8.tflite").stat().st_size / 1e6


# Turns "T01-T10" into ["T01", ..., "T10"].
def truck_range(text):
    first, last = text.split("-")
    trucks = []
    for number in range(int(first[1:]), int(last[1:]) + 1):
        trucks.append(f"T{number:02d}")
    return trucks


# Publishes one OTA command and prints what it costs on the M2M SIM.
def push(version, text):
    trucks = truck_range(text)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect("127.0.0.1", 1883, 60)
    client.loop_start()
    info = client.publish(OTA_TOPIC, json.dumps({"version": version, "trucks": trucks}), qos=1)
    info.wait_for_publish(timeout=10)
    client.loop_stop()
    client.disconnect()
    count = len(trucks)
    print(f"OTA {version} sent to {count} trucks ({trucks[0]}-{trucks[-1]})")
    print(f"  our M3 model layer: {count} x {MODEL_MB * 1000:.1f} kB = Rs {count * MODEL_MB * RUPEES_PER_MB:.4f}")
    print(f"  brief's 280 KB model: {count} x 280 KB = Rs {count * BRIEF_MODEL_MB * RUPEES_PER_MB:.2f}")


# Listens for a while and reports windows and alerts per release; the canary passes with no alerts.
def check(version, seconds):
    windows = {}
    trucks = {}
    alerts = {}
    current = {}

    def on_message(client, userdata, msg):
        data = json.loads(msg.payload.decode())
        truck = data["truck_id"]
        if msg.topic.endswith("/inference"):
            release = data["model_version"]
            current[truck] = release
            windows[release] = windows.get(release, 0) + 1
            trucks.setdefault(release, set()).add(truck)
        elif truck in current:
            release = current[truck]
            alerts[release] = alerts.get(release, 0) + 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("127.0.0.1", 1883, 60)
    client.subscribe("logibridge/trucks/+/inference", qos=1)
    client.subscribe("logibridge/trucks/+/alerts", qos=1)
    client.loop_start()
    print(f"Watching the fleet for {seconds} s ...")
    time.sleep(seconds)
    client.loop_stop()
    client.disconnect()
    for release in sorted(windows):
        print(f"  {release}: {len(trucks[release])} trucks, {windows[release]} windows, {alerts.get(release, 0)} alerts")
    if version not in windows:
        print(f"CANARY: no windows from {version} yet")
    elif len(windows) == 1:
        print(f"ROLLOUT COMPLETE: all {len(trucks[version])} trucks on {version}, {alerts.get(version, 0)} alerts")
    elif alerts.get(version, 0) == 0:
        print(f"CANARY PASSED: {version} ran {windows[version]} windows with no alerts -> roll out to the rest of the fleet")
    else:
        print(f"CANARY FAILED: {version} raised {alerts[version]} alerts -> roll the canary trucks back")


if sys.argv[1] == "push":
    push(sys.argv[2], sys.argv[3])
else:
    check(sys.argv[2], float(sys.argv[3]))

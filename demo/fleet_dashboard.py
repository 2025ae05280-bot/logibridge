# Live fleet view: one coloured cell per truck, release counts, recent alerts and drift. Run: $PY demo/fleet_dashboard.py
import json
import threading
import time

import paho.mqtt.client as mqtt

LABELS = ["NORMAL", "WARNING", "CRITICAL"]
COLOURS = ["\033[92m", "\033[93m", "\033[91m"]
RESET = "\033[0m"

latest = {}
versions = {}
drift = {}
alerts = []
lock = threading.Lock()


# Subscribes to every truck's inference, alert and drift topics.
def on_connect(client, userdata, flags, reason_code, properties=None):
    client.subscribe("logibridge/trucks/+/inference")
    client.subscribe("logibridge/trucks/+/alerts")
    client.subscribe("logibridge/trucks/+/drift")


# Stores the newest state of each truck.
def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    truck = msg.topic.split("/")[2]
    with lock:
        if msg.topic.endswith("/inference"):
            latest[truck] = int(data["final_class"])
            versions[truck] = data["model_version"]
        elif msg.topic.endswith("/drift"):
            drift[truck] = (data["psi"], time.time())
        else:
            alerts.append(f"{truck}  {data.get('label')}  (source: {data.get('source')})")
            del alerts[:-6]


# Redraws the whole screen every 2 seconds.
def draw():
    with lock:
        trucks = sorted(latest)
        print("\033[2J\033[H", end="")
        counts = [0, 0, 0]
        for truck in trucks:
            counts[latest[truck]] += 1
        print(f"LogiEdge fleet  {time.strftime('%H:%M:%S')}   trucks reporting: {len(trucks)}")
        print(f"{COLOURS[0]}NORMAL {counts[0]}{RESET}   {COLOURS[1]}WARNING {counts[1]}{RESET}   {COLOURS[2]}CRITICAL {counts[2]}{RESET}")
        release_counts = {}
        for truck in trucks:
            release_counts[versions[truck]] = release_counts.get(versions[truck], 0) + 1
        print("Releases: " + "   ".join(f"{name}: {count}" for name, count in sorted(release_counts.items())))
        print()
        newest = max(release_counts) if release_counts else ""
        line = ""
        for index, truck in enumerate(trucks):
            mark = "*" if len(release_counts) > 1 and versions[truck] == newest else " "
            line += f"{COLOURS[latest[truck]]}{truck}{RESET}{mark} "
            if index % 10 == 9:
                print(line)
                line = ""
        if line:
            print(line)
        if len(release_counts) > 1:
            print(f"\n(* = already on the newest release, {newest})")
        print("\nRecent alerts:")
        for item in alerts:
            print("  " + item)
        recent = []
        for truck in sorted(drift):
            value, seen = drift[truck]
            if time.time() - seen < 90:
                recent.append(f"{truck}={value:.2f}")
        if recent:
            print("\nDrift in the last 90 s (PSI > 0.25): " + ", ".join(recent))


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect("127.0.0.1", 1883, 60)
client.loop_start()
try:
    while True:
        draw()
        time.sleep(2)
except KeyboardInterrupt:
    client.loop_stop()

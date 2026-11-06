"""Per-truck PSI for the whole fleet: one confidence window per truck, one summary line per check."""

import argparse
import json
import time

from common.config import MQTT_HOST, MQTT_PORT, topic
from monitoring.drift_monitor import PSIDriftMonitor

DRIFT_LIMIT = 0.25


class FleetDriftMonitor:
    def __init__(self, reference_path, window=100):
        self.reference_path = reference_path
        self.window = window
        self.monitors = {}

    # Adds one confidence value to that truck's own PSI window.
    def add(self, truck_id, confidence):
        if truck_id not in self.monitors:
            self.monitors[truck_id] = PSIDriftMonitor(self.reference_path, self.window)
        self.monitors[truck_id].add(confidence)

    # Returns {truck_id: psi} for every truck whose window is full.
    def calculate(self):
        results = {}
        for truck_id in sorted(self.monitors):
            value = self.monitors[truck_id].calculate()
            if value is not None:
                results[truck_id] = value
        return results


def main():
    import paho.mqtt.client as mqtt

    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default=MQTT_HOST)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--reference", default="monitoring/reference_dist.json")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--every", type=float, default=60.0)
    args = parser.parse_args()

    fleet = FleetDriftMonitor(args.reference, args.window)
    last_report = time.monotonic()
    flagged = set()

    def on_message(client, userdata, msg):
        nonlocal last_report
        try:
            payload = json.loads(msg.payload.decode())
            fleet.add(payload["truck_id"], payload["confidence"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        now = time.monotonic()
        if now - last_report < args.every:
            return
        last_report = now
        results = fleet.calculate()
        if not results:
            print(f"[MLOPS] fleet PSI: waiting for {args.window} windows per truck ({len(fleet.monitors)} trucks seen)", flush=True)
            return
        worst = max(results, key=results.get)
        drifting = [truck for truck in results if results[truck] > DRIFT_LIMIT]
        print(f"[MLOPS] fleet PSI: {len(results)} trucks checked, max {results[worst]:.3f} ({worst}), drifting: {len(drifting)}", flush=True)
        for truck in drifting:
            flagged.add(truck)
        for truck in sorted(flagged):
            if truck in results and truck not in drifting:
                print(f"[MLOPS] {truck} PSI={results[truck]:.3f} (was drifting)", flush=True)
        for truck in drifting:
            print(f"[LOGIBRIDGE DRIFT ALERT] {truck} PSI={results[truck]:.3f}", flush=True)
            client.publish(topic(truck, "drift"), json.dumps({"ts": time.time(), "truck_id": truck, "psi": results[truck], "score": "confidence"}), qos=1)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(args.broker, args.port, 60)
    client.subscribe(topic("+", "inference"), qos=1)
    print(f"fleet PSI monitor: window {args.window} per truck, check every {args.every:.0f} s, limit {DRIFT_LIMIT}", flush=True)
    client.loop_forever()


if __name__ == "__main__":
    main()

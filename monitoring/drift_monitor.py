"""PSI monitor for the inference stream's probability of Normal."""

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

from common.config import MODEL_VERSION, MQTT_HOST, MQTT_PORT, TRUCK_ID, topic

EDGES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)


def proportions(values):
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=EDGES)
    return counts.astype(np.float64) / max(1, counts.sum())


def psi(expected, actual, epsilon=1e-4):
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    expected = (expected + epsilon) / (expected + epsilon).sum()
    actual = (actual + epsilon) / (actual + epsilon).sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def save_reference(path, values, model_version=MODEL_VERSION):
    bins = proportions(values).round(8).tolist()
    data = {"bins": bins, "edges": EDGES.tolist(), "proportions": bins, "n": len(values), "model_version": model_version, "score": "confidence"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def load_reference(path):
    data = json.loads(Path(path).read_text())
    values = data if isinstance(data, list) else data.get("proportions", data.get("bins"))
    if values is None or len(values) != 4:
        raise ValueError("reference must contain four proportions")
    values = np.asarray(values, dtype=np.float64)
    if not np.isclose(values.sum(), 1.0, atol=1e-3):
        raise ValueError("reference proportions must sum to 1")
    return values, data.get("model_version") if isinstance(data, dict) else None


class PSIDriftMonitor:
    def __init__(self, reference_path, window=100):
        self.expected, self.model_version = load_reference(reference_path)
        self.values = deque(maxlen=window)

    def add(self, p_normal):
        self.values.append(float(np.clip(p_normal, 0, 1)))

    def calculate(self):
        if len(self.values) < self.values.maxlen:
            return 0.0
        return psi(self.expected, proportions(self.values))


def _mqtt_client(host, port, truck_id, callback):
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = callback
    client.connect(host, port, 60)
    client.subscribe(topic(truck_id, "inference"), qos=1)
    return client


def build_reference(args):
    values = []
    def on_message(client, userdata, msg):
        try:
            value = json.loads(msg.payload.decode())["confidence"]
            values.append(float(value))
        except (KeyError, ValueError, json.JSONDecodeError):
            return
    client = _mqtt_client(args.broker, args.port, args.truck_id, on_message)
    client.loop_start()
    try:
        while len(values) < args.n:
            time.sleep(0.1)
    finally:
        client.loop_stop()
        client.disconnect()
    save_reference(args.reference, values[:args.n])


def monitor(args):
    monitor = PSIDriftMonitor(args.reference, args.window)
    last_report_ts = None

    def on_message(client, userdata, msg):
        nonlocal last_report_ts
        try:
            payload = json.loads(msg.payload.decode())
            if payload.get("model_version") and monitor.model_version and payload["model_version"] != monitor.model_version:
                logging.warning("model version differs from reference")
            monitor.add(payload["confidence"])
            event_ts = float(payload.get("ts", time.time()))
            if last_report_ts is None:
                last_report_ts = event_ts
            if event_ts - last_report_ts < args.every:
                return
            last_report_ts = event_ts
            value = monitor.calculate()
            if value:
                print(f"[MLOPS] PSI={value:.3f} n={len(monitor.values)}", flush=True)
                if value > 0.25:
                    print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)
                    client.publish(topic(args.truck_id, "drift"), json.dumps({"ts": time.time(), "truck_id": args.truck_id, "psi": value, "score": "p_normal"}), qos=1)
        except (KeyError, ValueError, json.JSONDecodeError):
            return
    client = _mqtt_client(args.broker, args.port, args.truck_id, on_message)
    client.loop_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default=MQTT_HOST)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--truck-id", default=TRUCK_ID)
    parser.add_argument("--reference", default="monitoring/reference_dist.json")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-reference")
    build.add_argument("--n", type=int, default=300)
    for option in ("--broker", "--port", "--truck-id", "--reference"):
        build.add_argument(option, default=argparse.SUPPRESS, type=int if option == "--port" else str)
    monitor_cmd = sub.add_parser("monitor")
    monitor_cmd.add_argument("--window", type=int, default=100)
    monitor_cmd.add_argument("--every", type=float, default=60.0)
    for option in ("--broker", "--port", "--truck-id", "--reference"):
        monitor_cmd.add_argument(option, default=argparse.SUPPRESS, type=int if option == "--port" else str)
    args = parser.parse_args()
    (build_reference if args.command == "build-reference" else monitor)(args)


if __name__ == "__main__":
    main()

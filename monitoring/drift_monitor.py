"""PSI monitor for the inference stream's maximum class confidence."""

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
    scores = np.asarray(values, dtype=np.float64)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("reference confidence scores must be non-empty and finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("reference confidence scores must be between 0 and 1")
    bins = proportions(scores).round(8).tolist()
    data = {
        "status": "ready",
        "bins": bins,
        "edges": EDGES.tolist(),
        "proportions": bins,
        "n": int(scores.size),
        "model_version": model_version,
        "score": "confidence",
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))


def load_reference(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and data.get("status", "ready") != "ready":
        raise ValueError(
            "PSI reference is pending; capture at least 300 clean confidence scores first"
        )
    values = data if isinstance(data, list) else data.get("proportions", data.get("bins"))
    if values is None or len(values) != 4:
        raise ValueError("reference must contain four proportions")
    if isinstance(data, dict):
        if data.get("score") != "confidence":
            raise ValueError("reference score must be model confidence")
        if int(data.get("n", 0)) < 300:
            raise ValueError("reference must contain at least 300 clean inference scores")
        edges = np.asarray(data.get("edges", []), dtype=np.float64)
        if not np.array_equal(edges, EDGES):
            raise ValueError("reference must use the four required confidence bins")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("reference proportions must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0, atol=1e-3):
        raise ValueError("reference proportions must sum to 1")
    return values, data.get("model_version") if isinstance(data, dict) else None


class PSIDriftMonitor:
    def __init__(self, reference_path, window=100):
        if window <= 0:
            raise ValueError("window must be positive")
        self.expected, self.model_version = load_reference(reference_path)
        self.values = deque(maxlen=window)

    def add(self, confidence):
        value = float(confidence)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        self.values.append(value)

    def calculate(self):
        if len(self.values) < self.values.maxlen:
            return None
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
            score = float(value)
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("confidence must be finite and between 0 and 1")
            values.append(score)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.warning("ignoring invalid inference confidence: %s", exc)
            return

    client = _mqtt_client(args.broker, args.port, args.truck_id, on_message)
    client.loop_start()
    deadline = time.monotonic() + args.timeout
    try:
        while len(values) < args.n:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"received {len(values)} of {args.n} required clean confidence scores"
                )
            time.sleep(0.1)
    finally:
        client.loop_stop()
        client.disconnect()
    save_reference(args.reference, values[:args.n])


def monitor(args):
    monitor = PSIDriftMonitor(args.reference, args.window)
    last_report_wall = None

    def on_message(client, userdata, msg):
        nonlocal last_report_wall
        try:
            payload = json.loads(msg.payload.decode())
            if payload.get("model_version") and monitor.model_version and payload["model_version"] != monitor.model_version:
                logging.warning("model version differs from reference")
            monitor.add(payload["confidence"])
            now = time.monotonic()
            if last_report_wall is None:
                last_report_wall = now
            if now - last_report_wall < args.every:
                return
            last_report_wall = now
            value = monitor.calculate()
            if value is not None:
                print(f"[MLOPS] PSI={value:.3f} n={len(monitor.values)}", flush=True)
                if value > 0.25:
                    print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)
                    client.publish(
                        topic(args.truck_id, "drift"),
                        json.dumps({
                            "ts": time.time(),
                            "truck_id": args.truck_id,
                            "psi": value,
                            "score": "confidence",
                        }),
                        qos=1,
                    )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.warning("ignoring invalid inference message for PSI: %s", exc)
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
    build.add_argument("--timeout", type=float, default=300.0)
    for option in ("--broker", "--port", "--truck-id", "--reference"):
        build.add_argument(option, default=argparse.SUPPRESS, type=int if option == "--port" else str)
    monitor_cmd = sub.add_parser("monitor")
    monitor_cmd.add_argument("--window", type=int, default=100)
    monitor_cmd.add_argument("--every", type=float, default=60.0)
    for option in ("--broker", "--port", "--truck-id", "--reference"):
        monitor_cmd.add_argument(option, default=argparse.SUPPRESS, type=int if option == "--port" else str)
    args = parser.parse_args()
    if args.command == "build-reference" and (args.n < 300 or args.timeout <= 0):
        parser.error("--n must be at least 300 and --timeout must be positive")
    if args.command == "monitor" and (args.window <= 0 or args.every <= 0):
        parser.error("--window and --every must be positive")
    (build_reference if args.command == "build-reference" else monitor)(args)


if __name__ == "__main__":
    main()

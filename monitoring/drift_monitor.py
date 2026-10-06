# monitoring/drift_monitor.py
"""PSI drift monitoring on the model's output confidence distribution.

Monitored score: p_normal, the model's confidence that the cargo is in the Normal
(reference) state. The reference population is clean Normal operation, where this
score sits near 1.0; a shift in the operating population moves mass to lower bins.
(The max-softmax confidence is not used: a well-trained model is equally confident
on Critical windows, so its distribution would barely move under drift.)

Modes
  --build-reference   run the model on 300 clean Normal windows -> reference_dist.json
  (default)           subscribe to logibridge/trucks/+/inference, keep the last 100
                      scores, print PSI every 60 s, alert when PSI > 0.25
  --offline-demo      no broker: simulate clean -> combined anomaly -> clean and print
                      the PSI timeline (used to verify thresholds before the live demo)
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import deque

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REFERENCE_PATH = os.path.join(ROOT, "monitoring", "reference_dist.json")
BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
PSI_ALERT = 0.25
PSI_STABLE = 0.10
EPS = 1e-4


def histogram(scores):
    counts, _ = np.histogram(np.clip(scores, 0.0, 1.0), bins=BIN_EDGES)   # last bin includes 1.0
    return counts / max(1, counts.sum())


def psi(actual, expected):
    a = np.asarray(actual, dtype=np.float64) + EPS
    e = np.asarray(expected, dtype=np.float64) + EPS
    a, e = a / a.sum(), e / e.sum()
    return float(np.sum((a - e) * np.log(a / e)))


class PSIDriftMonitor:
    def __init__(self, reference_path=REFERENCE_PATH, window=100):
        with open(reference_path) as f:
            ref = json.load(f)
        self.expected = ref["reference"]
        self.window = deque(maxlen=window)

    def add_inference_score(self, score):
        self.window.append(float(score))

    def calculate_psi(self):
        if not self.window:
            return None
        return psi(histogram(list(self.window)), self.expected)

    def report(self, prefix=""):
        value = self.calculate_psi()
        if value is None:
            print(f"{prefix}[MLOPS] waiting for inferences...")
            return None
        dist = np.round(histogram(list(self.window)), 3).tolist()
        print(f"{prefix}[MLOPS] PSI={value:.3f} over last {len(self.window)} inferences | bins={dist}")
        if value > PSI_ALERT:
            print(f"{prefix}[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}")
        return value


# --- offline helpers (model + simulator, no broker) ---------------------------
def _load_offline_pipeline(model_path):
    sys.path.append(ROOT)
    sys.path.append(os.path.join(ROOT, "training"))
    from common import make_interpreter, tflite_predict_one, STATS_PATH
    from data_pipeline.preprocessing import load_training_stats
    from training.generate_dataset import simulate_feature_windows
    mean, std = load_training_stats(STATS_PATH)
    interp = make_interpreter(model_path)

    def p_normal(mode, n_windows, seed):
        duration = 30 + 10 * (n_windows - 1)
        X = simulate_feature_windows(mode, duration, seed=seed)[:n_windows]
        return [float(tflite_predict_one(interp, (x - mean) / std)[0]) for x in X]
    return p_normal


def build_reference(model_path, n_windows=300):
    p_normal = _load_offline_pipeline(model_path)
    scores = p_normal('none', n_windows, seed=7)
    ref = histogram(scores)
    out = {
        "score": "p_normal",
        "bins": BIN_EDGES,
        "reference": [round(float(v), 6) for v in ref],
        "n_windows": len(scores),
        "model": os.path.basename(model_path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(REFERENCE_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f">> Reference from {len(scores)} clean Normal windows ({out['model']}): {out['reference']}")
    print(f">> Saved {REFERENCE_PATH}")


def offline_demo(model_path, window):
    """Simulated timeline: 20 min clean, 5 min combined anomaly, 20 min clean.
    One inference every 10 s; PSI printed every 60 s of simulated time."""
    p_normal = _load_offline_pipeline(model_path)
    phases = [("clean", 'none', 120, 11), ("DRIFT (combined)", 'combined', 30, 12), ("recovery (clean)", 'none', 120, 13)]
    mon = PSIDriftMonitor(window=window)
    t = 0
    for name, mode, n, seed in phases:
        print(f"\n--- t={t // 60:3d} min: {name} ---")
        for s in p_normal(mode, n, seed):
            mon.add_inference_score(s)
            t += 10
            if t % 60 == 0:
                mon.report(prefix=f"t={t // 60:3d}m ")


# --- live mode -----------------------------------------------------------------
def run_live(host, port, interval, window):
    import paho.mqtt.client as mqtt
    mon = PSIDriftMonitor(window=window)

    def on_connect(client, userdata, flags, rc, properties=None):
        print(f">> Drift monitor connected to {host}:{port}; PSI every {interval}s over last {window} inferences")
        client.subscribe("logibridge/trucks/+/inference", qos=1)

    def on_message(client, userdata, msg):
        try:
            mon.add_inference_score(json.loads(msg.payload.decode())["p_normal"])
        except (KeyError, ValueError) as e:
            print(f"[ERROR] bad inference payload: {e}", file=sys.stderr)

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="logiedge_drift_monitor")
    except AttributeError:
        client = mqtt.Client(client_id="logiedge_drift_monitor")
    client.on_connect, client.on_message = on_connect, on_message
    client.connect(host, port, 60)
    client.loop_start()
    try:
        while True:
            time.sleep(interval)
            mon.report(prefix=time.strftime("%H:%M:%S "))
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LogiEdge PSI drift monitor")
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--offline-demo", action="store_true")
    ap.add_argument("--model", default=os.path.join(ROOT, "inference", "model.tflite"))
    ap.add_argument("--host", default=os.getenv("MQTT_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between PSI reports")
    ap.add_argument("--window", type=int, default=100, help="rolling window of inferences")
    args = ap.parse_args()

    if args.build_reference:
        build_reference(args.model)
    elif args.offline_demo:
        offline_demo(args.model, args.window)
    else:
        run_live(args.host, args.port, args.interval, args.window)

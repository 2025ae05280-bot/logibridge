# inference/inference_service.py
"""LogiEdge on-truck inference service.

  sensors (local MQTT) -> preprocessing (filter, 30 s window / 10 s step, 6 features,
  normalise with training_stats.npy) -> TFLite model -> publish result (local MQTT)
                                                     -> alert log (append-only JSONL)
                                                     -> store-and-forward uplink to ops centre

Environment variables:
  MODEL_PATH      TFLite model (FP32 or full-INT8)          default: ./model.tflite
  STATS_PATH      normalisation stats                         default: ./training_stats.npy
  TRUCK_ID        truck identifier                            default: FB-TRUCK-001
  MQTT_HOST/PORT  on-truck Mosquitto broker                   default: localhost:1883
  UPLINK_HOST/PORT ops-centre broker over cellular; empty = no uplink configured
  ALERT_LOG_PATH  append-only alert log                       default: ./alert_log.jsonl
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))             # container: preprocessing.py copied next to this file
sys.path.append(str(BASE_DIR.parent))      # repo checkout
try:
    from preprocessing import LogiEdgePreprocessingEngine
except ModuleNotFoundError:
    from data_pipeline.preprocessing import LogiEdgePreprocessingEngine

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow as tf
    tflite = tf.lite

MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "model.tflite"))
STATS_PATH = os.getenv("STATS_PATH", str(BASE_DIR / "training_stats.npy"))
TRUCK_ID = os.getenv("TRUCK_ID", "FB-TRUCK-001")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
UPLINK_HOST = os.getenv("UPLINK_HOST", "")
UPLINK_PORT = int(os.getenv("UPLINK_PORT", "1883"))
ALERT_LOG_PATH = os.getenv("ALERT_LOG_PATH", str(BASE_DIR / "alert_log.jsonl"))

CLASS_LABELS = ["NORMAL", "WARNING", "CRITICAL"]
SENSOR_ROOT = f"logibridge/trucks/{TRUCK_ID}/sensors"
TOPIC_INFERENCE = f"logibridge/trucks/{TRUCK_ID}/inference"
TOPIC_ALERT = f"logibridge/trucks/{TRUCK_ID}/alerts"
TOPIC_UPLINK_ALERT = f"logibridge/ops/trucks/{TRUCK_ID}/alerts"


def new_client(client_id):
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


class EdgeModel:
    """TFLite wrapper that accepts float features for both FP32 and full-INT8 models."""

    def __init__(self, path):
        self.interp = tflite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        print(f">> Model {path} | input {self.inp['dtype'].__name__} | output {self.out['dtype'].__name__}")

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(self.inp["shape"])
        if self.inp["dtype"] == np.int8:
            scale, zp = self.inp["quantization"]
            x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        if self.out["dtype"] == np.int8:
            scale, zp = self.out["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        return y.reshape(-1)


class AlertStore:
    """Append-only on-device alert log with store-and-forward sync.

    Every alert is written to disk first (chain-of-custody record that survives
    connectivity gaps and restarts). A cursor file records how many log lines the
    ops centre has acknowledged (QoS 1 PUBACK); on reconnect the backlog is replayed.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.cursor_path = self.path.with_suffix(".synced")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, record):
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _synced(self):
        try:
            return int(self.cursor_path.read_text().strip() or 0)
        except FileNotFoundError:
            return 0

    def pending(self):
        if not self.path.exists():
            return 0, []
        with self.lock, open(self.path) as f:
            lines = f.readlines()
        start = self._synced()
        return start, lines[start:]

    def mark_synced(self, n):
        self.cursor_path.write_text(str(n))


class Uplink:
    """Cellular uplink to the ops centre. Reconnects automatically; flushes the backlog."""

    def __init__(self, store, host, port):
        self.store = store
        self.connected = threading.Event()
        self.flush_lock = threading.Lock()
        self.client = new_client(f"logiedge_uplink_{TRUCK_ID}")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=2, max_delay=30)
        self.client.connect_async(host, port, keepalive=30)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[UPLINK] Cellular link UP - syncing alert backlog")
        self.connected.set()
        threading.Thread(target=self.flush, daemon=True).start()

    def _on_disconnect(self, client, userdata, *args):
        if self.connected.is_set():
            print("[UPLINK] Cellular link DOWN - alerts buffered locally")
        self.connected.clear()

    def flush(self):
        if not self.connected.is_set():
            return
        with self.flush_lock:
            start, lines = self.store.pending()
            sent = 0
            for line in lines:
                info = self.client.publish(TOPIC_UPLINK_ALERT, line.strip(), qos=1)
                try:
                    info.wait_for_publish(timeout=10)
                except (RuntimeError, ValueError):
                    break
                if not info.is_published():
                    break
                sent += 1
                self.store.mark_synced(start + sent)
            if lines:
                print(f"[UPLINK] Synced {sent}/{len(lines)} buffered alert(s) to ops centre")


class InferenceService:
    def __init__(self):
        self.engine = LogiEdgePreprocessingEngine(STATS_PATH)
        self.model = EdgeModel(MODEL_PATH)
        self.store = AlertStore(ALERT_LOG_PATH)
        self.uplink = Uplink(self.store, UPLINK_HOST, UPLINK_PORT) if UPLINK_HOST else None
        self.door_state = "CLOSED"
        self.client = new_client(f"logiedge_inference_{TRUCK_ID}")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f">> Connected to on-truck broker {MQTT_HOST}:{MQTT_PORT}. Subscribed to {SENSOR_ROOT}/#")
        client.subscribe([(f"{SENSOR_ROOT}/temperature", 0),
                          (f"{SENSOR_ROOT}/vibration", 0),
                          (f"{SENSOR_ROOT}/door", 1)])

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            ts = float(payload["timestamp"])
            topic = msg.topic.rsplit("/", 1)[-1]
            if topic == "temperature":
                self.engine.add_temperature(ts, float(payload["value"]))
            elif topic == "vibration":
                self.engine.add_vibration(ts, float(payload["value"]))
            elif topic == "door":
                self.door_state = "OPEN" if payload["event"] == "OPEN" else "CLOSED"
                self.store.append({"type": "door", "truck_id": TRUCK_ID, "timestamp": ts,
                                   "event": payload["event"]})
                print(f"[DOOR] {payload['event']}")
                return

            if self.engine.window_ready():
                self.run_inference(ts)
        except Exception as e:
            print(f"[ERROR] Pipeline failure on {msg.topic}: {e}", file=sys.stderr)

    def run_inference(self, ts):
        raw = self.engine.extract_features()
        x = self.engine.normalize_features(raw)
        t0 = time.perf_counter()
        probs = self.model.predict(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        class_id = int(np.argmax(probs))

        result = {
            "truck_id": TRUCK_ID,
            "timestamp": ts,
            "class_id": class_id,
            "label": CLASS_LABELS[class_id],
            "confidence": round(float(probs[class_id]), 4),
            "p_normal": round(float(probs[0]), 4),
            "probs": [round(float(p), 4) for p in probs],
            "features": {"temp_mean": round(float(raw[0]), 3), "temp_std": round(float(raw[1]), 3),
                         "temp_rate_c_per_min": round(float(raw[2]), 3), "vib_rms": round(float(raw[3]), 3),
                         "vib_peak": round(float(raw[4]), 3), "vib_kurtosis": round(float(raw[5]), 3)},
            "door": self.door_state,
            "inference_ms": round(latency_ms, 3),
            "model": os.path.basename(MODEL_PATH),
        }
        self.client.publish(TOPIC_INFERENCE, json.dumps(result), qos=1)

        icon = ["🟢", "⚠️ ", "🔴"][class_id]
        print(f"{icon} [INFERENCE] {result['label']:<8} conf={result['confidence']:.3f} "
              f"temp={raw[0]:.2f}°C rate={raw[2]:+.2f}°C/min vib_rms={raw[3]:.3f}g "
              f"door={self.door_state} ({latency_ms:.2f} ms)")

        if class_id > 0:
            alert = {"type": "alert", **result}
            self.store.append(alert)                       # 1. durable local record first
            self.client.publish(TOPIC_ALERT, json.dumps(alert), qos=1)   # 2. in-cab / local consumers
            if self.uplink:
                threading.Thread(target=self.uplink.flush, daemon=True).start()   # 3. ops centre

    def run(self):
        print(f">> LogiEdge inference service | truck={TRUCK_ID} | broker={MQTT_HOST}:{MQTT_PORT} | "
              f"uplink={UPLINK_HOST or 'not configured'} | alert log={ALERT_LOG_PATH}")
        while True:
            try:
                self.client.connect(MQTT_HOST, MQTT_PORT, 60)
                break
            except OSError as e:
                print(f"[WAIT] Broker {MQTT_HOST}:{MQTT_PORT} not reachable ({e}); retrying in 3 s")
                time.sleep(3)
        self.client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    InferenceService().run()

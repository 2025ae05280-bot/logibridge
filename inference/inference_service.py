"""MQTT wiring for the shared feature extractor and TFLite model."""

import json
import logging
import os
import threading
import time

import numpy as np

from common.config import ALERT_DB, MODEL_PATH, MODEL_VERSION, MQTT_HOST, MQTT_PORT, SETPOINT_C, STATS_PATH, TRUCK_ID, UPLINK_HOST, topic
from data_pipeline.preprocessing import WindowFeatureExtractor
from inference.alert_store import AlertStore
from inference.decision import Debouncer
from inference.engine import ModelRunner
from inference.sync_worker import SyncWorker

log = logging.getLogger("logibridge.inference")
LABELS = ("NORMAL", "WARNING", "CRITICAL")


class InferenceService:
    def __init__(self, runner, truck_id=TRUCK_ID, setpoint=SETPOINT_C, alert_db=ALERT_DB):
        self.runner = runner
        self.truck_id = truck_id
        self.setpoint = setpoint
        self.extractor = WindowFeatureExtractor()
        self.decision = Debouncer()
        self.store = AlertStore(alert_db)
        now = time.time()
        self.last_seen = {"temperature": now, "vibration": now}
        self.last_seq = {}
        self.bad_messages = 0
        self.client = None
        self.previous_final = 0
        self.faults = set()

    def attach(self, client):
        self.client = client

    def ingest(self, stream, payload):
        try:
            ts = float(payload["ts"])
            seq = int(payload["seq"])
            if not np.isfinite(ts):
                raise ValueError("non-finite timestamp")
            if stream == "door":
                value = payload["value"]
                if value not in ("OPEN", "CLOSE"):
                    raise ValueError("door value must be OPEN or CLOSE")
            else:
                value = float(payload["value"])
                if not np.isfinite(value):
                    raise ValueError("non-finite sensor value")
        except (KeyError, TypeError, ValueError) as exc:
            self.bad_messages += 1
            log.warning("dropping invalid %s payload: %s", stream, exc)
            return
        self._note_sequence(stream, seq)
        if stream == "door":
            self.store.door_event(ts, self.truck_id, seq, value)
            return
        if stream not in ("temperature", "vibration"):
            self.bad_messages += 1
            log.warning("dropping unsupported sensor stream %s", stream)
            return
        self.last_seen[stream] = time.time()
        self.faults.discard(stream)
        for features in self.extractor.push(stream, ts, value):
            probs, latency_ms = self.runner.predict(features)
            previous_final = self.previous_final
            model_class, final_class, source = self.decision.decide(probs, features[0], self.setpoint)
            self.previous_final = final_class
            message = {"ts": ts, "truck_id": self.truck_id, "model_version": MODEL_VERSION, "model_class": model_class, "final_class": final_class, "class": final_class, "source": source, "probs": probs.tolist(), "p_normal": float(probs[0]), "confidence": float(np.max(probs)), "latency_ms": latency_ms, "features": dict(zip(("temp_mean", "temp_std", "temp_roc_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis"), features.tolist()))}
            self.store.window(ts, self.truck_id, features, probs, final_class)
            heartbeat = os.getenv("HEARTBEAT_FILE")
            if heartbeat:
                with open(heartbeat, "a", encoding="utf-8"):
                    os.utime(heartbeat, None)
            if self.client:
                self.client.publish(topic(self.truck_id, "inference"), json.dumps(message), qos=1)
            if final_class > 0 and final_class > previous_final:
                alert_id = self.store.alert(ts, self.truck_id, final_class, LABELS[final_class], source, probs)
                if self.client:
                    self.client.publish(topic(self.truck_id, "alerts"), json.dumps({**message, "id": alert_id, "label": LABELS[final_class]}), qos=2)

    def _note_sequence(self, stream, seq):
        previous = self.last_seq.get(stream)
        if previous is not None and seq > previous + 1:
            log.warning("%s sequence gap: %s missing", stream, seq - previous - 1)
        self.last_seq[stream] = seq

    def watchdog(self):
        now = time.time()
        if self.last_seen["temperature"] and now - self.last_seen["temperature"] > 10 and "temperature" not in self.faults:
            self.faults.add("temperature")
            self._fault_alert("temperature stale")
        if self.last_seen["vibration"] and now - self.last_seen["vibration"] > 20 and "vibration" not in self.faults:
            self.faults.add("vibration")
            self._fault_alert("vibration stale")

    def _fault_alert(self, reason):
        log.error("SENSOR_FAULT: %s", reason)
        alert_id = self.store.alert(time.time(), self.truck_id, 2, "SENSOR_FAULT", "watchdog", [])
        if self.client:
            self.client.publish(topic(self.truck_id, "alerts"), json.dumps({"id": alert_id, "truck_id": self.truck_id, "class": 2, "label": "SENSOR_FAULT", "source": "watchdog"}), qos=2)


def _connect(client, host, port):
    delay = 2
    for attempt in range(5):
        try:
            client.connect(host, port, 60)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(delay)
            delay = min(16, delay * 2)


def main():
    import paho.mqtt.client as mqtt

    logging.basicConfig(level=logging.INFO)
    service = InferenceService(ModelRunner(MODEL_PATH, STATS_PATH))
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"inference-{service.truck_id}")
    service.attach(client)
    client.will_set(topic(service.truck_id, "status"), json.dumps({"state": "offline", "truck_id": service.truck_id}), qos=1, retain=True)

    def on_connect(c, userdata, flags, reason_code, properties=None):
        c.subscribe(topic(service.truck_id, "sensors/#"), qos=1)
        c.publish(topic(service.truck_id, "status"), json.dumps({"state": "online", "truck_id": service.truck_id}), qos=1, retain=True)

    def on_message(c, userdata, msg):
        stream = msg.topic.rsplit("/", 1)[-1]
        try:
            service.ingest(stream, json.loads(msg.payload.decode()))
        except json.JSONDecodeError:
            service.bad_messages += 1
            log.warning("dropping invalid JSON on %s", msg.topic)

    client.on_connect = on_connect
    client.on_message = on_message
    _connect(client, MQTT_HOST, MQTT_PORT)
    client.loop_start()
    sync = SyncWorker(service.store, UPLINK_HOST, service.truck_id, MQTT_PORT) if UPLINK_HOST else None
    if sync:
        sync.start()
    try:
        while True:
            service.watchdog()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if sync:
            sync.stop()
        client.loop_stop()
        client.disconnect()
        service.store.close()


if __name__ == "__main__":
    main()

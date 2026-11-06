"""Fleet emulator: hosts one virtual edge node per truck in a single process.

Every truck keeps its own window buffer, debouncer, watchdog and model release, exactly as the
single-truck container does; only the process (and one shared SQLite file) is shared, so 85 trucks
fit on a laptop. Model updates arrive on logibridge/fleet/ota, which is how the canary rollout is shown.
"""

import argparse
import json
import logging
import queue
import threading
import time

from common.config import ALERT_DB, MQTT_HOST, MQTT_PORT, STATS_PATH, UPLINK_HOST, UPLINK_PORT, topic
from inference.alert_store import AlertStore
from inference.engine import ModelRunner
from inference.inference_service import InferenceService
from inference.sync_worker import SyncWorker

log = logging.getLogger("logibridge.fleet")
OTA_TOPIC = "logibridge/fleet/ota"


class FleetService:
    def __init__(self, runners, start_version, store):
        if start_version not in runners:
            raise ValueError(f"unknown start release {start_version}")
        self.runners = runners
        self.start_version = start_version
        self.store = store
        self.trucks = {}
        self.client = None

    # Returns the virtual edge node for a truck, creating it on the truck's first message.
    def node(self, truck_id):
        if truck_id not in self.trucks:
            service = InferenceService(self.runners[self.start_version], truck_id, store=self.store, model_version=self.start_version)
            service.attach(self.client)
            self.trucks[truck_id] = service
        return self.trucks[truck_id]

    # Sends one sensor message to its truck's own edge node.
    def ingest(self, truck_id, stream, payload):
        self.node(truck_id).ingest(stream, payload)

    # Moves the named trucks to another model release (the OTA update).
    def apply_ota(self, version, truck_ids):
        if version not in self.runners:
            log.warning("OTA ignored: release %s is not loaded", version)
            return 0
        updated = 0
        for truck_id in truck_ids:
            service = self.node(truck_id)
            service.runner = self.runners[version]
            service.model_version = version
            updated += 1
        log.info("OTA: %d trucks now on %s", updated, version)
        return updated

    def watchdog(self):
        for service in list(self.trucks.values()):
            service.watchdog()


# Reads "label=path" release options into a dict of loaded models.
def load_releases(items, stats_path):
    runners = {}
    for item in items:
        label, path = item.split("=", 1)
        runners[label] = ModelRunner(path, stats_path)
    return runners


def main():
    import paho.mqtt.client as mqtt

    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="append", required=True, help="label=path/to/model.tflite (repeatable)")
    parser.add_argument("--start", required=True, help="release every truck starts on")
    parser.add_argument("--alert-db", default=ALERT_DB)
    parser.add_argument("--broker", default=MQTT_HOST)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    fleet = FleetService(load_releases(args.release, STATS_PATH), args.start, AlertStore(args.alert_db))
    work = queue.Queue(maxsize=50000)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fleet-inference")
    fleet.client = client

    def on_connect(c, userdata, flags, reason_code, properties=None):
        c.subscribe(topic("+", "sensors/#"), qos=1)
        c.subscribe(OTA_TOPIC, qos=1)

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            log.warning("dropping invalid JSON on %s", msg.topic)
            return
        try:
            work.put_nowait((msg.topic, payload))
        except queue.Full:
            log.warning("queue full, dropping %s", msg.topic)

    # Worker thread: handles OTA commands and sensor messages in arrival order.
    def worker():
        while True:
            name, payload = work.get()
            try:
                if name == OTA_TOPIC:
                    fleet.apply_ota(payload["version"], payload["trucks"])
                else:
                    parts = name.split("/")
                    fleet.ingest(parts[2], parts[-1], payload)
            except Exception:
                log.exception("failed to handle %s", name)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, 60)
    client.loop_start()
    threading.Thread(target=worker, daemon=True).start()

    sync = SyncWorker(fleet.store, UPLINK_HOST, "fleet", UPLINK_PORT) if UPLINK_HOST else None
    if sync:
        sync.start()
    print(f"fleet inference running: releases {list(fleet.runners)}, all trucks start on {args.start}", flush=True)
    try:
        while True:
            fleet.watchdog()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if sync:
            sync.stop()
        client.loop_stop()
        client.disconnect()
        fleet.store.close()


if __name__ == "__main__":
    main()

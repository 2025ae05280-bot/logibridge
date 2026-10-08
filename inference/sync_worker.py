import json
import logging
import threading
import time

log = logging.getLogger("logibridge.sync")


class SyncWorker:
    def __init__(self, store, host, truck_id, port=1883, retry_interval=15):
        self.store, self.host, self.truck_id, self.port = store, host, truck_id, port
        self.retry_interval = retry_interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                log.warning("alert sync worker did not stop within 10 seconds")

    def _run(self):
        import paho.mqtt.client as mqtt
        while not self.stop_event.wait(self.retry_interval):
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            loop_started = False
            connected = threading.Event()
            connection_failure = []

            def on_connect(client, userdata, flags, reason_code, properties):
                if reason_code.is_failure:
                    connection_failure.append(str(reason_code))
                else:
                    connected.set()

            client.on_connect = on_connect
            try:
                client.connect(self.host, self.port, 10)
                client.loop_start()
                loop_started = True
                if not connected.wait(timeout=5):
                    detail = connection_failure[0] if connection_failure else "connection timed out"
                    raise TimeoutError(f"ops broker connection failed: {detail}")
                for row in self.store.unsynced_alerts():
                    payload = json.dumps({
                        "id": row[0],
                        "ts": row[1],
                        "truck_id": row[2],
                        "class": row[3],
                        "label": row[4],
                        "source": row[5],
                        "probs": json.loads(row[6]),
                    })
                    info = client.publish(
                        f"ops/trucks/{self.truck_id}/alerts",
                        payload,
                        qos=1,
                    )
                    if info.rc != mqtt.MQTT_ERR_SUCCESS:
                        raise RuntimeError(f"MQTT publish failed with code {info.rc}")
                    info.wait_for_publish(timeout=10)
                    if not info.is_published():
                        raise TimeoutError(f"alert {row[0]} was not acknowledged")
                    self.store.mark_synced([row[0]])
                for row in self.store.unsynced_door_events():
                    payload = json.dumps({
                        "id": row[0],
                        "ts": row[1],
                        "truck_id": row[2],
                        "seq": row[3],
                        "event": row[4],
                    })
                    info = client.publish(
                        f"ops/trucks/{self.truck_id}/events/door",
                        payload,
                        qos=1,
                    )
                    if info.rc != mqtt.MQTT_ERR_SUCCESS:
                        raise RuntimeError(f"MQTT publish failed with code {info.rc}")
                    info.wait_for_publish(timeout=10)
                    if not info.is_published():
                        raise TimeoutError(f"door event {row[0]} was not acknowledged")
                    self.store.mark_door_events_synced([row[0]])
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                log.warning("alert sync to %s failed: %s", self.host, exc)
            finally:
                try:
                    if loop_started:
                        client.loop_stop()
                    client.disconnect()
                except (OSError, RuntimeError) as exc:
                    log.warning("failed to close alert sync MQTT client: %s", exc)

import json
import threading
import time

class SyncWorker:
    def __init__(self, store, host, truck_id, port=1883):
        self.store, self.host, self.truck_id, self.port = store, host, truck_id, port
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):
        import paho.mqtt.client as mqtt
        while not self.stop_event.wait(15):
            client = mqtt.Client()
            try:
                client.connect(self.host, self.port, 10)
                client.loop_start()
                ids = []
                for row in self.store.unsynced_alerts():
                    ids.append(row[0])
                    client.publish(f"ops/trucks/{self.truck_id}/alerts", json.dumps({"id": row[0], "ts": row[1], "truck_id": row[2], "class": row[3], "label": row[4], "source": row[5], "probs": json.loads(row[6])}), qos=1).wait_for_publish()
                self.store.mark_synced(ids)
                client.loop_stop()
                client.disconnect()
            except Exception:
                try:
                    client.loop_stop()
                    client.disconnect()
                except Exception:
                    pass

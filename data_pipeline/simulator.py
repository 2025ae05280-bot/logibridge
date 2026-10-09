"""Deterministic sensor simulator used by both the live demo and training."""

import argparse
import json
import os
import random
import time

from common.config import MQTT_HOST, MQTT_PORT, sensor_topic

TEMP_NOISE_SD = 0.3
TEMP_DRIFT_PER_READING = 0.08


class SensorSimulator:
    def __init__(self, mode="none", truck_id="T01", seed=None, setpoint=4.0, bad_json_rate=0.0, drop_temp_after=None):
        self.mode = mode
        self.truck_id = truck_id
        self.setpoint = float(setpoint)
        self.random = random.Random(seed)
        self.bad_json_rate = float(bad_json_rate)
        self.drop_temp_after = drop_temp_after
        self.tick = 0
        self.seq = {"temperature": 0, "vibration": 0, "door": 0}
        self.door_open = False
        self.next_door = self.random.randint(300, 900)
        self.door_close = None

    def _payload(self, stream, ts, value):
        self.seq[stream] += 1
        return {"ts": float(ts), "truck_id": self.truck_id, "seq": self.seq[stream], "value": value}

    def step(self, t):
        """Advance one simulated second and return ``(stream, payload)`` pairs."""
        self.tick += 1
        door_event = None
        if not self.door_open and self.tick >= self.next_door:
            self.door_open = True
            self.door_close = self.tick + self.random.randint(30, 120)
            door_event = "OPEN"
        elif self.door_open and self.tick >= self.door_close:
            self.door_open = False
            self.next_door = self.tick + self.random.randint(300, 900)
            door_event = "CLOSE"

        if self.mode == "temp_drift":
            bias = self.tick * TEMP_DRIFT_PER_READING
        elif self.mode == "combined":
            bias = self.tick * TEMP_DRIFT_PER_READING
        elif self.mode == "cooling_fault":
            bias = -min(3.5, self.tick / 60.0)
        else:
            bias = 0.0
        if self.door_open:
            bias += self.random.uniform(0.5, 1.5)
        temp = self.setpoint + bias + self.random.gauss(0.0, TEMP_NOISE_SD)
        if self.random.random() < 0.005:
            temp += self.random.choice((-1.0, 1.0))

        result = [] if self.drop_temp_after is not None and self.tick >= self.drop_temp_after else [("temperature", self._payload("temperature", t, round(temp, 4)))]
        if self.tick % 2 == 0:
            vibration = self.random.gauss(1.2, 0.15) if self.mode in ("vibration", "combined") else self.random.gauss(0.45, 0.05)
            result.append(("vibration", self._payload("vibration", t, round(max(0.001, vibration), 4))))
        if door_event is not None:
            result.append(("door", self._payload("door", t, door_event)))
        return result


def _connect(client, host, port):
    delay = 2
    for attempt in range(5):
        try:
            client.connect(host, port, keepalive=60)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(delay)
            delay = min(16, delay * 2)


def run_sensor_simulator(args):
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{args.truck_id}")
    _connect(client, args.broker, args.port)
    client.loop_start()
    sim = SensorSimulator(args.anomaly, args.truck_id, args.seed, args.setpoint, args.bad_json_rate, args.drop_temp_after)
    try:
        for tick in range(args.duration or 2**31):
            for stream, payload in sim.step(float(tick)):
                data = json.dumps(payload)
                if args.bad_json_rate and sim.random.random() < args.bad_json_rate:
                    data = "{bad-json"
                client.publish(sensor_topic(args.truck_id, stream), data, qos=0)
            time.sleep(1.0 / args.fast)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly", choices=["none", "temp_drift", "vibration", "combined", "cooling_fault"], default=os.getenv("SIM_ANOMALY", "none"))
    parser.add_argument("--truck-id", default="T01")
    parser.add_argument("--broker", default=os.getenv("MQTT_HOST", MQTT_HOST))
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--duration", type=int, default=0, help="simulated seconds; 0 runs until interrupted")
    parser.add_argument("--fast", type=float, default=1.0, help="simulated seconds per wall-clock second")
    parser.add_argument("--setpoint", type=float, default=4.0)
    parser.add_argument("--drop-temp-after", type=float, default=None)
    parser.add_argument("--bad-json-rate", type=float, default=0.0)
    args = parser.parse_args()
    if args.fast <= 0 or not 0 <= args.bad_json_rate <= 1:
        parser.error("--fast must be positive and --bad-json-rate must be between 0 and 1")
    run_sensor_simulator(args)


if __name__ == "__main__":
    main()

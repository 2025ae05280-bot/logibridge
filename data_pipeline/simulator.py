# data_pipeline/simulator.py
"""Cold-chain truck sensor simulator.

The sensor physics live in ColdChainSensorModel so that the same code drives both
the live MQTT publisher (this script) and offline dataset generation
(training/generate_dataset.py), which steps the model with simulated time.

Usage:
    python data_pipeline/simulator.py --anomaly {none|temp_drift|vibration|combined}
                                      [--host localhost] [--truck-id FB-TRUCK-001] [--speed 1.0]
"""
import sys
import time
import json
import random
import argparse

TEMP_SETPOINT_C = 4.0
TEMP_NOISE_SD = 0.3
TEMP_DRIFT_PER_READING = 0.08       # temp_drift: +0.08 °C per 1 Hz reading
VIB_NORMAL = (0.45, 0.05)           # compressor RMS, g
VIB_FAULT = (1.2, 0.15)             # bearing-wear step, g
VIB_PERIOD_S = 2                    # 0.5 Hz
ANOMALY_MODES = ['none', 'temp_drift', 'vibration', 'combined']


def topic_root(truck_id):
    return f"logibridge/trucks/{truck_id}/sensors"


class ColdChainSensorModel:
    """Generates one tick (1 s) of sensor readings for a given anomaly mode."""

    def __init__(self, anomaly_mode='none', seed=None):
        if anomaly_mode not in ANOMALY_MODES:
            raise ValueError(f"Unknown anomaly mode: {anomaly_mode}")
        self.mode = anomaly_mode
        self.rng = random.Random(seed)
        self.tick = 0
        self.temp_bias = 0.0
        self.door_open = False
        self.next_door_tick = self.rng.randint(120, 600)

    def step(self):
        """Advance 1 s. Returns (temperature, vibration_or_None, door_event_or_None)."""
        self.tick += 1
        has_temp_fault = self.mode in ('temp_drift', 'combined')
        has_vib_fault = self.mode in ('vibration', 'combined')

        if has_temp_fault:
            self.temp_bias += TEMP_DRIFT_PER_READING
        temperature = self.rng.normalvariate(TEMP_SETPOINT_C + self.temp_bias, TEMP_NOISE_SD)

        vibration = None
        if self.tick % VIB_PERIOD_S == 0:
            mu, sd = VIB_FAULT if has_vib_fault else VIB_NORMAL
            vibration = max(0.001, self.rng.normalvariate(mu, sd))

        # Door: closed for 2-10 min, open for 20-90 s (delivery stops)
        door_event = None
        if self.tick >= self.next_door_tick:
            self.door_open = not self.door_open
            door_event = 'OPEN' if self.door_open else 'CLOSE'
            gap = self.rng.randint(20, 90) if self.door_open else self.rng.randint(120, 600)
            self.next_door_tick = self.tick + gap

        return temperature, vibration, door_event


def run_sensor_simulator(anomaly_mode, host='localhost', port=1883, truck_id='FB-TRUCK-001', speed=1.0):
    import paho.mqtt.client as mqtt

    root = topic_root(truck_id)
    topic_temp, topic_vib, topic_door = f"{root}/temperature", f"{root}/vibration", f"{root}/door"

    client_id = f"logiedge_sim_{truck_id}"
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        client = mqtt.Client(client_id=client_id)

    print(f">> Connecting to local broker at {host}:{port} | mode={anomaly_mode} | speed={speed}x")
    try:
        client.connect(host, port, keepalive=60)
        client.loop_start()
    except Exception as e:
        print(f"Broker connection failed: {e}")
        sys.exit(1)

    model = ColdChainSensorModel(anomaly_mode)
    # Simulated clock: real time when speed == 1, accelerated otherwise
    sim_start, wall_start = time.time(), time.monotonic()

    try:
        while True:
            temperature, vibration, door_event = model.step()
            ts = sim_start + model.tick

            client.publish(topic_temp, json.dumps({"timestamp": ts, "value": round(temperature, 4)}), qos=0)
            if vibration is not None:
                client.publish(topic_vib, json.dumps({"timestamp": ts, "value": round(vibration, 4)}), qos=0)
            if door_event is not None:
                client.publish(topic_door, json.dumps({"timestamp": ts, "event": door_event}), qos=1)
                print(f"\n[DOOR] {door_event}")

            sys.stdout.write(f"\r[Tick {model.tick:05d}] Temp: {temperature:6.2f}°C  Mode: {anomaly_mode}   ")
            sys.stdout.flush()

            next_wall = wall_start + model.tick / speed
            time.sleep(max(0.0, next_wall - time.monotonic()))
    except KeyboardInterrupt:
        print("\nHalting simulator safely...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiEdge cold-chain sensor simulator")
    parser.add_argument('--anomaly', choices=ANOMALY_MODES, default='none')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=1883)
    parser.add_argument('--truck-id', default='FB-TRUCK-001')
    parser.add_argument('--speed', type=float, default=1.0,
                        help="Time acceleration factor (e.g. 10 = 10 simulated seconds per wall second)")
    args = parser.parse_args()
    run_sensor_simulator(args.anomaly, args.host, args.port, args.truck_id, args.speed)

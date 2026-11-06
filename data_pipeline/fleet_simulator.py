"""Simulates the whole pilot fleet (T01-T85) on one MQTT connection, each truck on its own topics."""

import argparse
import json
import time

from common.config import MQTT_HOST, MQTT_PORT, sensor_topic
from data_pipeline.simulator import SensorSimulator, _connect

MODES = ("none", "temp_drift", "vibration", "combined", "cooling_fault")


# Turns a number into a truck id, e.g. 7 -> "T07".
def truck_name(number):
    return f"T{number:02d}"


# Reads "T07=combined,T23=temp_drift" into {"T07": "combined", "T23": "temp_drift"}.
def parse_faults(text):
    faults = {}
    if not text:
        return faults
    for item in text.split(","):
        truck, mode = item.split("=")
        truck = truck.strip()
        mode = mode.strip()
        if mode not in MODES:
            raise ValueError(f"unknown anomaly mode {mode} for {truck}")
        faults[truck] = mode
    return faults


# Builds one simulator per truck; trucks not named in faults run clean.
def build_fleet(count, faults, seed=1):
    fleet = []
    for number in range(1, count + 1):
        truck = truck_name(number)
        mode = faults.get(truck, "none")
        fleet.append(SensorSimulator(mode, truck, seed + number))
    for truck in faults:
        if truck not in [sim.truck_id for sim in fleet]:
            raise ValueError(f"{truck} is not in the fleet T01-{truck_name(count)}")
    return fleet


# Publishes one simulated second for every truck, then sleeps according to --fast.
def run(args):
    import paho.mqtt.client as mqtt

    faults = parse_faults(args.faults)
    fleet = build_fleet(args.trucks, faults, args.seed)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fleet-simulator")
    _connect(client, args.broker, args.port)
    client.loop_start()
    print(f"fleet simulator: {len(fleet)} trucks, faults: {faults or 'none'}, --fast {args.fast}", flush=True)
    sent = 0
    started = time.monotonic()
    try:
        for tick in range(args.duration or 2**31):
            for sim in fleet:
                for stream, payload in sim.step(float(tick)):
                    client.publish(sensor_topic(sim.truck_id, stream), json.dumps(payload), qos=args.qos)
                    sent += 1
            if tick % 60 == 59:
                rate = sent / (time.monotonic() - started)
                print(f"t={tick + 1}s  messages sent={sent}  ({rate:.0f} msg/s)", flush=True)
            time.sleep(1.0 / args.fast)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trucks", type=int, default=85)
    parser.add_argument("--faults", default="", help='e.g. "T07=combined,T23=temp_drift"')
    parser.add_argument("--broker", default=MQTT_HOST)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", type=int, default=0, help="simulated seconds; 0 runs until interrupted")
    parser.add_argument("--fast", type=float, default=1.0, help="simulated seconds per wall-clock second")
    parser.add_argument("--qos", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    if args.trucks < 1 or args.trucks > 99 or args.fast <= 0:
        parser.error("--trucks must be 1-99 and --fast must be positive")
    run(args)


if __name__ == "__main__":
    main()

# data_pipeline/simulator.py
import sys
import time
import json
import random
import argparse
import paho.mqtt.client as mqtt

def run_sensor_simulator(anomaly_mode):
    MQTT_BROKER = "localhost"
    MQTT_PORT = 1883
    CLIENT_ID = "logiedge_truck_sensor_sim"
    
    TOPIC_TEMP = "logibridge/sensor/raw/temp"
    TOPIC_VIB  = "logibridge/sensor/raw/vib"
    TOPIC_DOOR = "logibridge/sensor/raw/door"
    
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    except AttributeError:
        client = mqtt.Client(client_id=CLIENT_ID)
        
    print(f">> Connecting to local broker at {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
    except Exception as e:
        print(f"Broker connection failed: {e}")
        sys.exit(1)

    step_counter = 0
    linear_temp_bias = 0.0
    
    try:
        while True:
            step_counter += 1
            current_time = time.time()
            has_temp_fault = anomaly_mode in ['temp_drift', 'combined']
            has_vib_fault  = anomaly_mode in ['vibration', 'combined']
            
            # 1. Temp Stream (1 Hz)
            if has_temp_fault:
                linear_temp_bias += 0.08
                temp_reading = random.normalvariate(4.0 + linear_temp_bias, 0.3)
            else:
                temp_reading = random.normalvariate(4.0, 0.3)
            client.publish(TOPIC_TEMP, json.dumps({"timestamp": current_time, "value": round(temp_reading, 4)}), qos=0)
            
            # 2. Vibration RMS Stream (0.5 Hz -> every 2 ticks)
            if step_counter % 2 == 0:
                vib_reading = random.normalvariate(1.2, 0.15) if has_vib_fault else random.normalvariate(0.45, 0.05)
                vib_reading = max(0.001, vib_reading)
                client.publish(TOPIC_VIB, json.dumps({"timestamp": current_time, "value": round(vib_reading, 4)}), qos=0)
                
            # 3. Door Event
            door_payload = None
            if step_counter == 60: door_payload = {"timestamp": current_time, "event": "OPEN"}
            elif step_counter == 75: door_payload = {"timestamp": current_time, "event": "CLOSE"}
            if door_payload: client.publish(TOPIC_DOOR, json.dumps(door_payload), qos=1)
                
            sys.stdout.write(f"\r[Tick {step_counter:04d}] Temp: {temp_reading:6.2f}°C")
            sys.stdout.flush()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nHalting simulator safely...")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly', choices=['none', 'temp_drift', 'vibration', 'combined'], default='none')
    args = parser.parse_args()
    run_sensor_simulator(args.anomaly)

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env(name, default):
    return os.getenv(name, default)


MQTT_HOST = env("MQTT_HOST", "localhost")
MQTT_PORT = int(env("MQTT_PORT", "1883"))
TRUCK_ID = env("TRUCK_ID", "T01")
SETPOINT_C = float(env("SETPOINT_C", "4.0"))
ALERT_DB = env("ALERT_DB", str(REPO_ROOT / "results" / "alerts.db"))
MODEL_PATH = env("MODEL_PATH", str(REPO_ROOT / "training" / "models" / "m1_fp32.tflite"))
STATS_PATH = env("STATS_PATH", str(REPO_ROOT / "data_pipeline" / "training_stats.npy"))
MODEL_VERSION = env("MODEL_VERSION", Path(MODEL_PATH).stem)
UPLINK_HOST = env("UPLINK_HOST", "")


def topic(truck_id, stream):
    return f"logibridge/trucks/{truck_id}/{stream.lstrip('/')}"


def sensor_topic(truck_id, stream):
    return topic(truck_id, f"sensors/{stream}")

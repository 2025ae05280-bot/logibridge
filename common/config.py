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


# Reads the release version from a MODEL_VERSION file next to the model, else uses the model file name.
def default_model_version(model_path):
    version_file = Path(model_path).with_name("MODEL_VERSION")
    if version_file.exists():
        return version_file.read_text().strip()
    return Path(model_path).stem


MODEL_VERSION = env("MODEL_VERSION", default_model_version(MODEL_PATH))
UPLINK_HOST = env("UPLINK_HOST", "")
UPLINK_PORT = int(env("UPLINK_PORT", "1883"))


def topic(truck_id, stream):
    return f"logibridge/trucks/{truck_id}/{stream.lstrip('/')}"


def sensor_topic(truck_id, stream):
    return topic(truck_id, f"sensors/{stream}")

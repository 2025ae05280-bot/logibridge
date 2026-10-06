# inference/Dockerfile
# Build from the repository root:
#   docker build -f inference/Dockerfile -t logibridge-inference:latest .
FROM python:3.11-slim
WORKDIR /app

# --- Dependency layers (rarely change; cached across OTA model updates) ---
RUN pip install --no-cache-dir "numpy<2" scipy paho-mqtt tensorflow==2.13.1

# --- Application code (changes occasionally) ---
COPY data_pipeline/simulator.py /app/data_pipeline/simulator.py
COPY data_pipeline/preprocessing.py /app/preprocessing.py
COPY inference/inference_service.py /app/inference_service.py
COPY inference/training_stats.npy /app/training_stats.npy

# --- Model layer LAST: an OTA model update only rebuilds/ships this layer ---
COPY inference/model.tflite /app/model.tflite

ENV MODEL_PATH=/app/model.tflite \
    STATS_PATH=/app/training_stats.npy \
    TRUCK_ID=FB-TRUCK-001 \
    MQTT_HOST=localhost \
    MQTT_PORT=1883 \
    ALERT_LOG_PATH=/data/alert_log.jsonl \
    PYTHONUNBUFFERED=1

CMD ["python", "-u", "inference_service.py"]

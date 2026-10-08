FROM python:3.11-slim

RUN useradd -r -u 1000 edge
WORKDIR /app
COPY inference/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY common/ /app/common/
COPY data_pipeline/preprocessing.py /app/data_pipeline/preprocessing.py
COPY inference/*.py /app/inference/
RUN mkdir -p /opt/logibridge /var/lib/logibridge /tmp \
    && chown -R edge:edge /opt/logibridge /var/lib/logibridge /tmp /app

ENV MODEL_PATH=/opt/logibridge/model.tflite \
    STATS_PATH=/opt/logibridge/training_stats.npy \
    ALERT_DB=/var/lib/logibridge/alerts.db \
    TRUCK_ID=T01 \
    MQTT_HOST=mqtt_broker \
    MQTT_PORT=1883 \
    HEARTBEAT_FILE=/tmp/logibridge-heartbeat
USER edge
HEALTHCHECK --interval=30s CMD python -m inference.healthcheck
CMD ["python", "-u", "-m", "inference.inference_service"]
COPY --chown=edge:edge inference/model.tflite /opt/logibridge/model.tflite

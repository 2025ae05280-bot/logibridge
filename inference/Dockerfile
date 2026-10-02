FROM python:3.11-slim
WORKDIR /app

# Pin numpy to legacy v1 specs to ensure full compatibility with tflite-runtime
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && \
    pip install --no-cache-dir paho-mqtt "numpy<2" scipy && \
    pip install --no-cache-dir tflite-runtime && \
    apt-get purge -y --auto-remove gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

COPY inference_service.py /app/
COPY training_stats.npy /app/
COPY model.tflite /app/model.tflite
ENV MODEL_PATH=/app/model.tflite

CMD ["python", "inference_service.py"]

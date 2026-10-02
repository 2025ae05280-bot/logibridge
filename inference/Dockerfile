# inference/Dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install system compilation headers and pin dependencies to eliminate NumPy 2 conflicts
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && \
    pip install --no-cache-dir paho-mqtt "numpy<2" scipy && \
    pip install --no-cache-dir tflite-runtime && \
    apt-get purge -y --auto-remove gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Staging application metrics and logic scripts before the model file for layer caching
COPY training_stats.npy /app/
COPY inference_service.py /app/

# Staging the model layer at the lowest level to satisfy the explicit OTA layer-cache test requirement
COPY model.tflite /app/model.tflite

ENV MODEL_PATH=/app/model.tflite
ENV TRUCK_ID=FB-TRUCK-085

CMD ["python", "-u", "inference_service.py"]

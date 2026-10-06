# inference/Dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install the package set required for local inference and keep the runtime compatible
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && \
    pip install --no-cache-dir paho-mqtt "numpy<2" scipy tensorflow==2.13.1 && \
    apt-get purge -y --auto-remove gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Stage app files and model for OTA cache-friendly layout
COPY training_stats.npy /app/
COPY inference_service.py /app/
COPY model.tflite /app/model.tflite

ENV MODEL_PATH=/app/model.tflite
ENV STATS_PATH=/app/training_stats.npy
ENV TRUCK_ID=FB-TRUCK-085

CMD ["python", "-u", "inference_service.py"]

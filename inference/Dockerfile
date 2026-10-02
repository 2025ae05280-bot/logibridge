FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir paho-mqtt numpy scipy tensorflow-lite-runtime
COPY inference_service.py /app/
COPY training_stats.npy /app/
COPY model.tflite /app/model.tflite
ENV MODEL_PATH=/app/model.tflite
CMD ["python", "inference_service.py"]

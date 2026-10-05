# LogiEdge

Local cold-chain telemetry, feature extraction, TFLite inference, offline alert custody, and PSI drift monitoring.

## Host workflow

```bash
python -m pip install -r requirements.txt
python training/generate_dataset.py
python training/train_model.py
python training/convert_ptq.py
python training/prune_quantise.py
python optimisation/benchmark.py
```

The training generator and live service share `SensorSimulator` and `WindowFeatureExtractor`. Dataset splits are by run. Stats are one plain `(2, 6)` float32 file: `data_pipeline/training_stats.npy`.

## Local simulator and service

```bash
python -m data_pipeline.simulator --anomaly none --truck-id T01 --seed 7
MODEL_PATH=training/models/m2_ptq_int8.tflite python -m inference.inference_service
python -m monitoring.drift_monitor build-reference --truck-id T01 --n 300
python -m monitoring.drift_monitor monitor --truck-id T01 --window 100 --every 60
```

The service publishes inference at `logibridge/trucks/{id}/inference`, alerts at `.../alerts`, and retained status at `.../status`. SQLite stores every emitted window and alert; the sync worker forwards unsynced alert IDs to the ops topic when configured.

## Docker

```bash
docker compose up --build mqtt_broker ops_broker
docker compose --profile dev up --build
SIM_ANOMALY=combined SIM_FAST=10 docker compose up telemetry_simulator
```

Compose binds the lab broker to loopback and enables Mosquitto persistence. The lab config is anonymous for a clean demo; production needs password authentication, ACLs, and TLS.

## Verification

```bash
python -m compileall common data_pipeline inference monitoring training optimisation
pytest
```

TensorFlow is only needed for training/conversion. The host benchmark uses the TFLite shim and falls back to `tensorflow.lite`; the inference image uses `tflite-runtime`.

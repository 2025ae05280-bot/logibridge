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

Set `MODEL_FILE` to a generated TFLite file before starting `inference_engine` to switch variants without replacing the tracked fallback model. Compose persists SQLite alert and door-event custody in the `inference_data` volume and forwards acknowledged records to the local ops broker; set `UPLINK_HOST` to the configured operations endpoint for deployment. Compose binds the lab broker to loopback and enables Mosquitto persistence. The lab config is anonymous for a clean demo; production needs password authentication, ACLs, and TLS.

### Watch readable predictions

Install `jq` on the Ubuntu host (not inside the broker container). As `root`, run:

```bash
apt-get update
apt-get install -y jq
```

As a regular user, use `sudo`:

```bash
sudo apt-get update
sudo apt-get install -y jq
```

Then, while the broker, inference service, and simulator are running, run:

```bash
docker compose exec -T mqtt_broker mosquitto_sub -h localhost \
  -t 'logibridge/trucks/T01/inference' |
jq -r '
  ["NORMAL", "WARNING", "CRITICAL"] as $labels
  | . as $m
  | ($labels[$m.final_class] // "UNKNOWN") as $class
  | "Prediction: \($class)\nConfidence: \((100 * $m.confidence | round))%\nProbabilities: NORMAL \((100 * $m.probs[0] | round))% | WARNING \((100 * $m.probs[1] | round))% | CRITICAL \((100 * $m.probs[2] | round))%\nTemperature: \($m.features.temp_mean) °C\nLatency: \($m.latency_ms) ms\n"
'
```

Each incoming MQTT message is printed in a readable format. Press `Ctrl+C` to stop watching.

## Verification

```bash
python -m compileall common data_pipeline inference monitoring training optimisation
pytest
```

TensorFlow is only needed for training/conversion. The host benchmark uses the TFLite shim and falls back to `tensorflow.lite`; the inference image uses `tflite-runtime`.

## Submission evidence

Use [docs/SUBMISSION_DATA_COLLECTION.md](docs/SUBMISSION_DATA_COLLECTION.md) to collect reproducible logs, model metrics, benchmark outputs, PSI evidence, Docker layer history, and Ansible idempotency results. Record final measured values in `results/RESULTS.md`.

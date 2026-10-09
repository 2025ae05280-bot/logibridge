# LogiEdge

Local cold-chain telemetry, multi-threaded feature extraction, TFLite inference, offline alert custody, and PSI drift monitoring.

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

## Docker & Container Deployment

To launch the full Edge AI simulation cluster framework cleanly while ensuring strict memory boundary protection under WSL 2, use the built-in profile orchestrations:

```bash
# Start the baseline message brokers
docker compose up -d mqtt_broker ops_broker

# Deploy the complete multi-threaded ML pipeline ecosystem
docker compose --profile dev up -d --build
```

### Automated Live Presentation Demo Launcher

A unified orchestration script (`run_demo.sh`) is provided to automatically clear old message caching volumes, boot containers sequentially to prevent OOM memory spikes, and split your terminal window using `tmux` into a dual-pane live dashboard interface:

```bash
# Grant execution permissions
chmod +x run_demo.sh

# Launch the unified presentation dashboard
./run_demo.sh
```
* **Left Pane**: Displays continuous multi-threaded background prediction loops.
* **Right Pane**: Displays a real-time, color-coded live streaming telemetry analysis engine tracker.

---

### Interactive Monitoring & Verification Scripts

#### 1. Color-Coded Dynamic Streaming Dashboard
To audit model inference metrics, payload latency profiles, and confidence intervals without processing raw message network clutter, run this filtered host utility natively inside an active terminal workspace (requires `pip3 install paho-mqtt --break-system-packages` on Ubuntu 24.04+ environments):

```python
python3 -c "
import paho.mqtt.client as mqtt, json
def on_message(c, u, m):
    try:
        data = json.loads(m.payload.decode())
        if 'latency_ms' in data or 'confidence' in data or 'final_class' in data:
            final_class = data.get('final_class', 0)
            labels = ['NORMAL', 'WARNING', 'CRITICAL']
            status_label = labels[final_class] if final_class < len(labels) else 'UNKNOWN'
            color = '\033[92m' if final_class == 0 else ('\033[93m' if final_class == 1 else '\033[91m')
            reset = '\033[0m'
            print(f'📈 [Model Output] Topic: {m.topic} | TS: {data.get(\"ts\")} | Status: {color}{status_label}{reset} | Confidence: {data.get(\"confidence\", 0)*100:.2f}% | Latency: {data.get(\"latency_ms\", 0):.2f}ms')
    except Exception: pass
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = lambda c,u,f,r,p: c.subscribe('#')
client.on_message = on_message
try:
    client.connect('127.0.0.1', 1883)
    print('📢 Dynamic Inference Streaming Dashboard Active. Awaiting predictions...'); client.loop_forever()
except Exception as e: print('Connection error:', e)
"
```

#### 2. Safe Runtime Pipeline Hot-Reload
To trigger structural scenario resets or test active alert escalation flows right in front of an audience without tearing down network ports or causing memory drops, execute this quick command sequence:

```bash
# 1. Stop background loop workers quietly
docker compose --profile dev stop telemetry_simulator inference_engine

# 2. Bounce the broker process to completely clear its memory cache
docker compose --profile dev restart mqtt_broker

# 3. Relaunch with a target emergency simulation profile (e.g. combined)
SIM_ANOMALY=combined docker compose --profile dev start inference_engine telemetry_simulator

# 4. Watch the real-time inference loop output streams
docker compose --profile dev logs -f inference_engine
```

---

### Watch readable predictions (Legacy Option)

Install `jq` on the Ubuntu host (not inside the broker container). As a regular user, use `sudo`:

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

  | "Prediction: \($class)\nConfidence: \((100 * $m.confidence | round))%\nProbabilities: NORMAL \((100 * $m.probs[0] | round))% | WARNING \((100 * $m.probs[1] | round))% | CRITICAL \((100 * $m.probs[2] | round))%\nTemperature: \((100 * $m.features.temp_mean | round) / 100) °C\nLatency: \($m.latency_ms) ms\n"
'
```

Values vary with each prediction. Press `Ctrl+C` to stop watching.

## Verification

```bash
python -m compileall common data_pipeline inference monitoring training optimisation
pytest
```

TensorFlow is only needed for training/conversion. The host benchmark uses the TFLite shim and falls back to `tensorflow.lite`; the inference image uses `tflite-runtime`.

## Submission evidence

Use [docs/SUBMISSION_DATA_COLLECTION.md](docs/SUBMISSION_DATA_COLLECTION.md) to collect reproducible logs, model metrics, benchmark outputs, PSI evidence, Docker layer history, and Ansible idempotency results. Record final measured values in `results/RESULTS.md`.

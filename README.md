# LogiEdge: Intelligent Edge AI Platform for Cold-Chain Logistics

LogiEdge is an on-truck Edge AI pipeline for **FreightBridge Logistics Pvt. Ltd.**'s 85 refrigerated
trucks. It classifies the cargo compartment as **Normal / Warning / Critical** from cargo temperature,
compressor vibration and door events, entirely on the truck, and syncs its alert log with the
operations centre whenever cellular coverage is available.

```
simulator.py ──MQTT (local Mosquitto)──▶ inference_service.py ──▶ logibridge/trucks/{id}/inference
 temp 1 Hz, vib 0.5 Hz, door events       filter → 30 s window / 10 s step     logibridge/trucks/{id}/alerts
                                          → 6 features → normalise → TFLite    alert_log.jsonl ──uplink──▶ ops centre
                                                                    │
                                          drift_monitor.py ◀────────┘ PSI every 60 s
```

## Repository layout

| Path | Contents |
|---|---|
| `scenario_architecture/` | Constraint analysis (A1), system architecture diagram (A2) |
| `hardware/` | Constraint Triangle + Roofline analysis (B1, B2) |
| `data_pipeline/simulator.py` | Sensor simulator, `--anomaly {none\|temp_drift\|vibration\|combined}` (C1) |
| `data_pipeline/preprocessing.py` | Filtering, windowed feature extraction, normalisation (C2, C3) |
| `data_pipeline/normalisation_experiment.py` | Correct vs ±3σ-shifted stats experiment (C2) |
| `data_pipeline/mqtt_architecture.md` | Topic tree and QoS per topic |
| `training/` | Dataset generation, M1 training, M2 PTQ, M3 structured pruning + PTQ (D1, F1) |
| `inference/` | Dockerfile, inference service, deployed `model.tflite` (D2) |
| `monitoring/` | PSI drift monitor and `reference_dist.json` (E1) |
| `deployment/logibridge_deploy.yml` | 7-task Ansible OTA playbook (E2) |
| `optimisation/` | Five-metric benchmark, `results/benchmark_results.csv`, `results/pareto_chart.png` (F2) |

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "numpy<2" scipy paho-mqtt psutil matplotlib tensorflow==2.13.1 tensorflow-model-optimization==0.7.5
```

## 2. Data, training and optimisation (run from the repo root)

```bash
python training/generate_dataset.py          # dataset.npz + training_stats.npy (10 min clean Normal)
python training/train_model.py               # M1 FP32; stops if val accuracy ≤ 88%
python training/convert_ptq.py               # M2 full INT8 PTQ (200 calibration samples)
python training/prune_quantise.py            # M3 35% structured unit pruning + INT8
python optimisation/benchmark.py --tdp 15    # 5 metrics + Class 2 recall → CSV + Pareto chart
python data_pipeline/normalisation_experiment.py

cp training/models/m3_pruned_int8.tflite inference/model.tflite   # variant chosen for deployment
python monitoring/drift_monitor.py --build-reference              # 300 clean Normal windows → reference_dist.json
```

## 3. Run the on-truck stack (Docker)

```bash
docker build -f inference/Dockerfile -t logibridge-inference:latest .
docker compose up -d                          # broker + inference + simulator (mode none)
docker logs -f inference_engine

ANOMALY=combined docker compose up -d telemetry_simulator   # inject a fault
ANOMALY=none     docker compose up -d telemetry_simulator   # restore

# Switch model variant without a rebuild
MODEL_PATH=/models/m2_ptq_int8.tflite docker compose up -d inference_engine
```

Without Docker: run `mosquitto`, then `python inference/inference_service.py` and
`python data_pipeline/simulator.py --anomaly none` in two terminals.

## 4. Drift monitoring demo (E1)

```bash
python monitoring/drift_monitor.py                       # live: PSI every 60 s over last 100 inferences
python monitoring/drift_monitor.py --offline-demo        # simulated clean → combined → clean timeline
```

One inference is produced every 10 s of sensor time, so 100 inferences span ~17 min. For a
demo-length recovery, run the simulator with `--speed 10` (`SPEED=10` with compose).

## 5. OTA layer-cache demo (D2) and Ansible (E2)

```bash
cp training/models/m2_ptq_int8.tflite inference/model.tflite
docker build -f inference/Dockerfile -t logibridge-inference:latest .   # only the model layer rebuilds
docker history logibridge-inference:latest

ansible-galaxy collection install community.docker && pip install docker
docker run -d -p 5000:5000 --name registry registry:2
docker tag logibridge-inference:latest localhost:5000/logibridge-inference:latest
docker push localhost:5000/logibridge-inference:latest
ansible-playbook deployment/logibridge_deploy.yml --ask-become-pass   # run twice: 2nd run changed=0
```

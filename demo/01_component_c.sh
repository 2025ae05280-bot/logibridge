#!/bin/bash
# Segment 1: tests and Component C (simulator, MQTT, preprocessing, normalisation experiment).
# Terminal 2 should be showing:  podman exec -it mqtt_broker mosquitto_sub -h localhost -t 'logibridge/trucks/T01/sensors/#' -v
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh

step "Unit tests (14 tests)" '$PY -m pytest -q -p no:cacheprovider'
step "Simulator help: the --anomaly flag" '$PY -m data_pipeline.simulator --help'
step "Normal mode at real speed: 1 Hz temperature, 0.5 Hz vibration (watch terminal 2)" '$PY -m data_pipeline.simulator --anomaly none --truck-id T01 --seed 7 --duration 12'
step "temp_drift: +0.08 C per reading" '$PY -m data_pipeline.simulator --anomaly temp_drift --truck-id T01 --seed 7 --duration 12'
step "vibration: step to about 1.2 g" '$PY -m data_pipeline.simulator --anomaly vibration --truck-id T01 --seed 7 --duration 12'
step "combined: both faults together" '$PY -m data_pipeline.simulator --anomaly combined --truck-id T01 --seed 7 --duration 12'
step "Normalisation stats saved once from 10 min of clean data (row 1 = mean, row 2 = std)" '$PY -c "import numpy as np; np.set_printoptions(precision=3, suppress=True); print(np.load(\"data_pipeline/training_stats.npy\"))"'
step "Mandatory experiment: correct stats vs stats shifted by 3 sigma" '$PY data_pipeline/normalisation_experiment.py 2>/dev/null'
say "Segment 1 done."

# Submission evidence collection

Run every command from the repository root. This runbook collects measured evidence for the assignment; do not type benchmark, accuracy, PSI, latency, or cost numbers into the report by hand.

## 1. Prepare evidence storage

~~~powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$evidence = Join-Path $PWD "results\collection\$stamp"
New-Item -ItemType Directory -Force $evidence | Out-Null
Start-Transcript -Path (Join-Path $evidence 'terminal.log')
git rev-parse HEAD
python --version
docker --version
docker compose version
~~~

Keep raw CSVs, logs, and screenshots under results/collection/<timestamp>/.

## 2. Generate data and statistics

~~~powershell
python -m pip install -r requirements.txt
python training/generate_dataset.py
python -c "import csv,collections; r=list(csv.DictReader(open('training/data/dataset.csv'))); print(collections.Counter(x['run_id'] for x in r)); print(collections.Counter(x['split'] for x in r))"
~~~

Collect training/data/dataset.csv, data_pipeline/training_stats.npy, and the terminal output. The Normal run must be 20 minutes; anomaly runs are 15 minutes. The statistics file must be a plain float32 array with shape (2, 6).

## 3. Train and create variants

~~~powershell
python training/train_model.py
python training/convert_ptq.py
python training/prune_quantise.py
python data_pipeline/normalisation_experiment.py
~~~

Expected outputs:

- training/models/m1_fp32.keras
- training/models/m1_fp32.tflite
- training/models/m2_ptq_int8.tflite
- training/models/m3_pruned_ptq_int8.tflite
- training/results/metrics.json
- training/results/confusion_matrix.png
- results/normalisation_experiment.csv
- results/normalisation_experiment.txt

Copy validation accuracy and Critical recall from metrics.json. The gate is validation accuracy greater than 0.88 and Critical recall greater than 0.95.

## 4. Benchmark

The benchmark uses 10 warm-up invocations, 200 timed invocations, and the held-out validation split.

~~~powershell
$env:LAPTOP_TDP_W = '45'
python optimisation/benchmark.py
~~~

Collect optimisation/results/benchmark_results.csv and optimisation/results/pareto_chart.png. Keep the runtime name and TDP used with the evidence. If a model is missing, generate it; never fill the CSV manually.

## 5. Run inference

~~~powershell
docker compose up -d mqtt_broker ops_broker
$env:MODEL_FILE = "$PWD\training\models\m2_ptq_int8.tflite"
docker compose --profile dev up -d --build inference_engine
$env:SIM_ANOMALY = 'none'
$env:SIM_FAST = '10'
docker compose up -d --force-recreate telemetry_simulator
docker compose logs -f inference_engine | Tee-Object "$evidence\inference.log"
~~~

MODEL_FILE selects the generated model without overwriting the tracked fallback model. Capture payloads containing probs, confidence, features, latency_ms, online status, and alert/window writes.

## 6. Collect PSI evidence

First collect 300 clean confidence scores:

~~~powershell
python -m monitoring.drift_monitor build-reference --truck-id T01 --n 300
~~~

Verify monitoring/reference_dist.json has n: 300, four proportions summing to 1, and score: confidence.

Start the monitor in another terminal:

~~~powershell
python -m monitoring.drift_monitor monitor --truck-id T01 --window 100 --every 60 | Tee-Object "$evidence\psi.log"
~~~

Run the simulator lifecycle none -> combined -> none, recreating it after each change:

~~~powershell
docker compose stop telemetry_simulator
$env:SIM_ANOMALY = 'combined'
docker compose up -d --force-recreate telemetry_simulator
# wait for the drift interval
docker compose stop telemetry_simulator
$env:SIM_ANOMALY = 'none'
docker compose up -d --force-recreate telemetry_simulator
~~~

Keep the PSI log and screenshots showing clean, alert, and recovery phases. The required alert line is [LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}.

## 7. Collect Docker OTA evidence

~~~powershell
Copy-Item inference/model.tflite "$evidence\model.tflite.original"
Copy-Item training/models/m2_ptq_int8.tflite inference/model.tflite
docker build -f inference/Dockerfile -t logibridge-inference:ota-before .
docker history --no-trunc logibridge-inference:ota-before | Tee-Object "$evidence\docker-history-before.log"
~~~

Change only the model file used by the final COPY layer, rebuild with a new tag, and restore the tracked fallback afterward:

~~~powershell
Copy-Item training/models/m3_pruned_ptq_int8.tflite inference/model.tflite
docker build -f inference/Dockerfile -t logibridge-inference:ota-after .
docker history --no-trunc logibridge-inference:ota-after | Tee-Object "$evidence\docker-history-after.log"
Copy-Item "$evidence\model.tflite.original" inference/model.tflite
~~~

Use measured model-layer and image sizes to calculate the 85-truck bandwidth saving. State measured values, not only the 280 KB planning assumption.

## 8. Collect Ansible idempotency evidence

Start the local registry if it does not already exist, then install the collection:

~~~powershell
docker run -d --name registry -p 127.0.0.1:5000:5000 registry:2
ansible-galaxy collection install community.docker==3.12.1
ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml | Tee-Object "$evidence\ansible-run1.log"
ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml | Tee-Object "$evidence\ansible-run2.log"
~~~

The registry must contain the tag referenced by the playbook, for example: docker tag logibridge-inference:ota-after localhost:5000/logibridge-inference:latest followed by docker push localhost:5000/logibridge-inference:latest. The shared logiedge_net network is created by Docker Compose.

The second run must report changed=0. Save both logs and record the model variant, image tag, and health-check result.

## 9. Populate the results ledger

Copy every measured number into results/RESULTS.md with the exact output file or log path that produced it. Use the ledger as the only source for report and chart numbers.

~~~powershell
Stop-Transcript
python scripts/check_numbers.py
git status --short
~~~

The final submission still needs the repository URL, demo video link, and 2,500–3,000-word final PDF.

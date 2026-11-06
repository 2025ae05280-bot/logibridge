#!/bin/bash
# Segment 2: Component D (training result, OTA release v1.0.0 -> v1.0.1 of M3, MODEL_PATH switch).
# Terminal 2 should be showing the dashboard:  $PY demo/dashboard.py
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh
source demo/stack.sh

BUILD='podman build --format docker -f inference/Dockerfile'
SHOW_BUILD='2>&1 | grep -E "STEP|Using cache|-->"'

step "Training result: validation accuracy and per-class recall" '$PY -m json.tool training/results/metrics.json | head -40'
step "The three model variants (we ship M3)" 'ls -l training/models/*.tflite'
step "Dockerfile: pip install first, COPY model.tflite + MODEL_VERSION last" 'cat inference/Dockerfile'
step "Release v1.0.0: build the image with M3" "cp training/models/m3_pruned_int8.tflite inference/model.tflite && echo m3_pruned_int8-v1.0.0 > inference/MODEL_VERSION && $BUILD -t logibridge-inference:v1.0.0 . $SHOW_BUILD"
step "Deploy v1.0.0 on truck T01 (watch the dashboard)" 'sim_start none 10; release_start v1.0.0; sleep 12; podman logs --tail 3 inference_engine'
step "OTA release v1.0.1: same M3 model, new version, rebuild" "cp training/models/m3_pruned_int8.tflite inference/model.tflite && echo m3_pruned_int8-v1.0.1 > inference/MODEL_VERSION && $BUILD -t logibridge-inference:v1.0.1 . $SHOW_BUILD"
step "Layer history of v1.0.1: only the top layer is new" 'podman history logibridge-inference:v1.0.1 | head -6'
step "Layers shared between v1.0.0 and v1.0.1, and the bandwidth saving for 85 trucks" '$PY demo/ota_saving.py'
step "Apply the OTA: truck T01 moves to v1.0.1 (dashboard version changes)" 'release_start v1.0.1; sleep 12; podman logs --tail 3 inference_engine'
step "Restore the tracked model and version files" 'cp training/models/m1_fp32.tflite inference/model.tflite && echo m1_fp32 > inference/MODEL_VERSION'
step "Run the container with MODEL_PATH = M1 (watch the dashboard)" 'inference_start m1_fp32; sleep 12; podman logs --tail 3 inference_engine'
step "Switch to M2 by changing only MODEL_PATH, no rebuild" 'inference_start m2_ptq_int8; sleep 12; podman logs --tail 3 inference_engine'
step "Back to M1 for the rest of the demo" 'inference_start m1_fp32'
say "Segment 2 done. The simulator is still running in normal mode."

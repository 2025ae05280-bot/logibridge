#!/bin/bash
# After recording: stop the simulator and remove the demo containers (images are kept).
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/stack.sh
sim_stop
fleet_stop
podman rm -f inference_engine mqtt_broker ops_broker registry >/dev/null 2>&1
cp training/models/m1_fp32.tflite inference/model.tflite
echo m1_fp32 > inference/MODEL_VERSION
rm -rf "$HOME/logibridge-opt"
echo "Demo stack removed."

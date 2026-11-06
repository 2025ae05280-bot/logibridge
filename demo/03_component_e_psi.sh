#!/bin/bash
# Segment 3: Component E1 PSI drift (clean -> inject combined -> recover) and the offline dead-zone story.
# Terminal 2: $PY demo/dashboard.py
# Terminal 3: $PY -m monitoring.drift_monitor monitor --truck-id T01 --window 100 --every 60
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh
source demo/stack.sh

step "PSI reference: 300 clean windows of M1 confidence in 4 bins" 'cat monitoring/reference_dist.json'
step "Clean data (wait for 2 PSI lines below 0.10 in terminal 3, about 2-3 min)" 'inference_start m1_fp32; sim_start none 10'
step "Inject the fault: --anomaly combined (watch for [LOGIBRIDGE DRIFT ALERT])" 'sim_start combined 10'
step "Restore clean data (wait until PSI drops below 0.10, about 3 min)" 'sim_start none 10'
step "Dead zone: cut the cellular uplink (stop the ops-centre broker)" 'podman stop ops_broker'
step "Fault while offline: the truck still detects and logs it" 'sim_start combined 10; sleep 15; alert_counts'
step "Coverage returns: the backlog arrives at the operations centre within one 15 s sync" 'podman start ops_broker >/dev/null; sleep 1; podman exec ops_broker mosquitto_sub -h localhost -t "ops/#" -v -W 25 2>/dev/null; alert_counts'
step "Back to clean data" 'sim_start none 10'
say "Segment 3 done."

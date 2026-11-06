#!/bin/bash
# Segment 3c: Component E3 - canary OTA of v1.0.1 to 10 trucks, health check, then the other 75.
# Terminal 2: $PY demo/fleet_dashboard.py
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh
source demo/stack.sh

step "Fleet of 85 trucks, all on m3_pruned_int8-v1.0.0" 'fleet_start; fleet_sim "" 10; sleep 40'
step "Canary: send v1.0.1 to the first 10 trucks only (T01-T10)" '$PY demo/fleet_ota.py push m3_pruned_int8-v1.0.1 T01-T10'
step "Canary health check: watch the fleet for 60 s" '$PY demo/fleet_ota.py check m3_pruned_int8-v1.0.1 60'
step "Canary passed: roll v1.0.1 out to the remaining 75 trucks (T11-T85)" '$PY demo/fleet_ota.py push m3_pruned_int8-v1.0.1 T11-T85'
step "Confirm the whole fleet is on v1.0.1" '$PY demo/fleet_ota.py check m3_pruned_int8-v1.0.1 20'
step "Stop the fleet" 'fleet_stop'
say "Segment 3c done."

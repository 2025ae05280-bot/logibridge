#!/bin/bash
# Segment 1b: Component C at fleet scale - 85 trucks streaming, 3 of them faulty.
# Terminal 2: $PY demo/fleet_dashboard.py
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh
source demo/stack.sh

step "Start 85 virtual edge nodes (one window buffer, debouncer and alert log per truck)" 'fleet_start'
step "Stream all 85 trucks; T07 combined fault, T23 temperature drift, T51 cooling fault" 'fleet_sim "T07=combined,T23=temp_drift,T51=cooling_fault" 10'
step "Each truck has its own topics (sample of T42 for 3 s)" 'podman exec mqtt_broker mosquitto_sub -h localhost -t "logibridge/trucks/T42/sensors/#" -v -W 3 2>/dev/null | head -n 6'
step "After about 60 s: which trucks raised alerts (only the 3 faulty ones should)" 'sleep 20; fleet_alerts'
step "Stop the fleet" 'fleet_stop'
say "Segment 1b done."

#!/bin/bash
# Segment 3b: Component E1 at fleet scale - PSI per truck, drift flagged only on the faulty trucks.
# Terminal 2: $PY demo/fleet_dashboard.py
# Terminal 3: $PY -m monitoring.fleet_drift_monitor --window 100 --every 60
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh
source demo/stack.sh

step "Fleet PSI reference: the same 300 clean M1 windows used for truck T01" 'cat monitoring/reference_dist.json'
step "Clean fleet on M1 (wait for 2 fleet PSI lines with drifting: 0 in terminal 3, about 3 min)" 'fleet_start m1_fp32; fleet_sim "" 10'
step "Inject faults on T07 and T23 only (watch for [LOGIBRIDGE DRIFT ALERT] T07 / T23)" 'fleet_sim "T07=combined,T23=temp_drift" 10'
step "Restore clean data (wait until drifting: 0 again, about 3 min)" 'fleet_sim "" 10'
step "Stop the fleet" 'fleet_stop'
say "Segment 3b done."

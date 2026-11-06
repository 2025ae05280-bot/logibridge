# Helpers shared by the segment scripts for the simulator and the inference container.

IMAGE=localhost/logibridge-inference:v1.0.1
SIM_LOG="$REPO/.deps/simulator.log"

# Starts the simulator in the background in the given mode (fast = simulated seconds per real second).
sim_start() {
  sim_stop
  (nohup $PY -u -m data_pipeline.simulator --anomaly "$1" --truck-id T01 --fast "${2:-10}" >> "$SIM_LOG" 2>&1 & echo $! > "$REPO/.deps/simulator.pid")
  echo "simulator running: --anomaly $1 --fast ${2:-10}"
}

# Stops the background simulator if one is running.
sim_stop() {
  if [ -f "$REPO/.deps/simulator.pid" ]; then
    kill "$(cat "$REPO/.deps/simulator.pid")" 2>/dev/null
    rm -f "$REPO/.deps/simulator.pid"
  fi
}

# Starts the inference container with a chosen model file from training/models (no rebuild).
inference_start() {
  podman rm -f inference_engine >/dev/null 2>&1
  podman run -d --name inference_engine --network logiedge_net \
    -e MODEL_PATH="/models/$1.tflite" -e MODEL_VERSION="$1" \
    -e MQTT_HOST=mqtt_broker -e UPLINK_HOST=ops_broker \
    -v "$REPO/training/models:/models:ro" \
    -v "$REPO/data_pipeline/training_stats.npy:/opt/logibridge/training_stats.npy:ro" \
    "$IMAGE" >/dev/null
  echo "inference_engine started with MODEL_PATH=/models/$1.tflite"
}

# Starts the inference container from a release image (e.g. v1.0.0) using the model baked into it.
release_start() {
  podman rm -f inference_engine >/dev/null 2>&1
  podman run -d --name inference_engine --network logiedge_net \
    -e MQTT_HOST=mqtt_broker -e UPLINK_HOST=ops_broker \
    -v "$REPO/data_pipeline/training_stats.npy:/opt/logibridge/training_stats.npy:ro" \
    "localhost/logibridge-inference:$1" >/dev/null
  echo "inference_engine started from release $1"
}

# Counts alerts in the container's SQLite custody log, split by synced flag.
alert_counts() {
  podman exec inference_engine python -c "import sqlite3; db = sqlite3.connect('/var/lib/logibridge/alerts.db'); print('alerts (synced, count):', db.execute('select synced, count(*) from alerts group by synced').fetchall()); print('windows logged:', db.execute('select count(*) from windows').fetchone()[0])"
}

# --- Fleet emulator (85 virtual trucks) ---
FLEET_LOG="$REPO/.deps/fleet.log"
M3="training/models/m3_pruned_int8.tflite"

# Starts the fleet inference process. $1 = release every truck starts on (default M3 v1.0.0).
# M1 is loaded too because the PSI segment needs it: M3's confidence is ~1.0 on every window.
fleet_start() {
  fleet_stop
  podman rm -f inference_engine >/dev/null 2>&1
  sim_stop
  rm -f "$REPO/results/fleet_alerts.db"*
  (UPLINK_HOST=127.0.0.1 UPLINK_PORT=1884 nohup $PY -u -m inference.fleet_service \
    --release "m1_fp32=training/models/m1_fp32.tflite" \
    --release "m3_pruned_int8-v1.0.0=$M3" --release "m3_pruned_int8-v1.0.1=$M3" \
    --start "${1:-m3_pruned_int8-v1.0.0}" --alert-db "$REPO/results/fleet_alerts.db" \
    --broker 127.0.0.1 >> "$FLEET_LOG" 2>&1 & echo $! > "$REPO/.deps/fleet_service.pid")
  sleep 2
  tail -n 1 "$FLEET_LOG"
}

# Starts (or restarts) the fleet simulator: $1 = faults like "T07=combined", $2 = speed.
fleet_sim() {
  if [ -f "$REPO/.deps/fleet_sim.pid" ]; then kill "$(cat "$REPO/.deps/fleet_sim.pid")" 2>/dev/null; fi
  (nohup $PY -u -m data_pipeline.fleet_simulator --trucks 85 --faults "$1" --fast "${2:-10}" \
    --broker 127.0.0.1 >> "$FLEET_LOG" 2>&1 & echo $! > "$REPO/.deps/fleet_sim.pid")
  echo "fleet simulator: 85 trucks, faults: ${1:-none}, --fast ${2:-10}"
}

# Stops the fleet simulator and the fleet inference process.
fleet_stop() {
  for name in fleet_sim fleet_service; do
    if [ -f "$REPO/.deps/$name.pid" ]; then
      kill "$(cat "$REPO/.deps/$name.pid")" 2>/dev/null
      rm -f "$REPO/.deps/$name.pid"
    fi
  done
}

# Counts the fleet's alerts per truck from the shared SQLite log.
fleet_alerts() {
  $PY -c "import sqlite3; db = sqlite3.connect('results/fleet_alerts.db'); rows = db.execute('select truck_id, label, count(*), sum(synced) from alerts group by truck_id, label order by truck_id').fetchall(); print('truck  label     alerts  synced'); [print(f'{r[0]}    {r[1]:<9} {r[2]:<7} {r[3]}') for r in rows]; print('windows logged:', db.execute('select count(*) from windows').fetchone()[0], 'from', db.execute('select count(distinct truck_id) from windows').fetchone()[0], 'trucks')"
}

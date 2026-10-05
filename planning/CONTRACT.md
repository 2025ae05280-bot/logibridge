# LogiEdge contract v1.1

The runnable code is the source of truth for names and payloads.

## Changelog

- DEC-1: retain v1.0 names and topics; v1.1 adds `seq`, `p_normal`, and split CSV files.
- DEC-2: `temp_drift` caps at setpoint +2.5 °C; `combined` continues to a Critical-range bias.
- DEC-3: PSI monitors the assignment-required `confidence` score (`max(probs)`). `p_normal` remains available as an additional diagnostic field.
- DEC-4: `--fast` changes wall-clock speed while simulated timestamps remain one second apart.
- DEC-5: the model decides; a ±3 °C safety interlock may escalate only.
- DEC-6: host inference uses TensorFlow Lite when `tflite-runtime` is unavailable; the container uses `tflite-runtime`.

## Sensor payload

`logibridge/trucks/{truck_id}/sensors/{temperature,vibration,door}` carries `{ts, truck_id, seq, value}`. `value` is numeric for temperature/vibration and `OPEN` or `CLOSE` for door events.

## Inference payload

Inference publishes `probs`, `confidence` (PSI input), `p_normal` (diagnostic), `model_class`, `final_class`, `source`, `features`, `latency_ms`, and `model_version`.

## Runtime defaults

`MQTT_HOST=localhost`, `MQTT_PORT=1883`, `TRUCK_ID=T01`, `SETPOINT_C=4.0`, `ALERT_DB=results/alerts.db`, and `STATS_PATH=data_pipeline/training_stats.npy` on the host. Containers mount the model and the same stats file at `/opt/logibridge`.

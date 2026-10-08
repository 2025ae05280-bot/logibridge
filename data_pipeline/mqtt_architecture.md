# MQTT design and feature-level fusion

## Topic tree

```text
logibridge/trucks/{truck_id}/
  sensors/temperature   (QoS 1)
  sensors/vibration     (QoS 1)
  sensors/door          (QoS 1)
  inference             (QoS 1)
  alerts                (QoS 2)
  status                (QoS 1, retained, LWT)
  drift                 (QoS 1)
ops/trucks/{truck_id}/alerts (QoS 1)
ops/trucks/{truck_id}/events/door (QoS 1)
```

Sensor messages use `{ts, truck_id, seq, value}`. Temperature is 1 Hz and vibration is 0.5 Hz; the extractor aligns them by timestamp inside a 30-second window. Door messages are emitted only on state transitions and recorded in SQLite alongside alerts and windows. QoS 1 preserves readings across transient reconnects, while SQLite remains the durable local record. The uplink worker forwards unsynced alerts and door events only after the ops broker acknowledges them; event IDs support receiver-side deduplication. Alerts use QoS 2 because duplicate delivery is more costly than the extra handshake. Status is retained so a newly connected observer immediately sees liveness.

Feature-level fusion is the smallest useful boundary: each modality keeps its native sampling rate, then six reproducible features are combined for one model input. Data-level fusion would require resampling raw streams and increases payload size; decision-level fusion loses cross-modal interactions such as rising temperature plus vibration. Door events stay contextual and do not become a seventh feature until a trained model uses them.

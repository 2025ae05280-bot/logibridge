# Component C — MQTT Design & Pipeline Architecture

## 1. Local and Remote Topic Trees
* `logibridge/sensor/raw/temp` (Local, QoS 0): High-frequency temperature metrics.
* `logibridge/sensor/raw/vib` (Local, QoS 0): High-frequency vibration compressor metrics.
* `logibridge/sensor/raw/door` (Local, QoS 1): Discrete cabin security open/close updates.
* `logibridge/trucks/{truck_id}/inference` (Remote Uplink, QoS 1): Outbound anomalous alerts.

## 2. Rationale & QoS Strategy
We implement Feature-Level Data Fusion because merging raw modalities at the text layer creates large synchronization issues, while Decision-Level systems miss critical cross-modal patterns. 

QoS 0 is selected for local raw telemetry to minimize processing overhead on the internal bus, since occasional missing samples are smoothed out by our 5-sample moving average filter. 

QoS 1 is strictly applied to the container's outbound uplink topic to ensure at-least-once delivery for critical thermal breaches, guaranteeing that anomalies are safely buffered in the offline SQLite log until verified by the central backend.

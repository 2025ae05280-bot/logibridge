# Component C — MQTT Design & Pipeline Architecture

## 1. Topic tree

```
On-truck broker (Mosquitto, localhost:1883) — works with no cellular coverage
logibridge/
└── trucks/
    └── {truck_id}/                        e.g. FB-TRUCK-001
        ├── sensors/
        │   ├── temperature                QoS 0   1 Hz     {"timestamp", "value"}  °C
        │   ├── vibration                  QoS 0   0.5 Hz   {"timestamp", "value"}  g RMS
        │   └── door                       QoS 1   event    {"timestamp", "event": "OPEN"|"CLOSE"}
        ├── inference                      QoS 1   every 10 s  class, probs, p_normal, features, door
        └── alerts                         QoS 1   on class 1/2  same payload + "type": "alert"

Ops-centre broker (over cellular, store-and-forward)
logibridge/
└── ops/
    └── trucks/{truck_id}/alerts          QoS 1   replayed from the on-truck alert log on reconnect
```

| Publisher | Subscriber | Topic |
|---|---|---|
| `data_pipeline/simulator.py` | inference service | `.../sensors/temperature`, `.../sensors/vibration`, `.../sensors/door` |
| `inference/inference_service.py` | drift monitor, in-cab display | `.../inference`, `.../alerts` |
| `inference/inference_service.py` (uplink client) | ops-centre backend | `logibridge/ops/trucks/{truck_id}/alerts` |
| — | `monitoring/drift_monitor.py` | `logibridge/trucks/+/inference` |

## 2. QoS per topic

| Topic | QoS | Reason |
|---|---|---|
| `sensors/temperature`, `sensors/vibration` | 0 | High rate, local loopback; one lost sample is absorbed by the 5-sample moving average and the 30 s window. |
| `sensors/door` | 1 | Rare discrete events; a lost OPEN/CLOSE would corrupt the custody record. |
| `inference`, `alerts` | 1 | Every classification feeds PSI monitoring and the custody trail; duplicates are harmless (timestamped), losses are not. |
| `ops/.../alerts` (uplink) | 1 | At-least-once delivery over an unreliable cellular link; the local cursor only advances after PUBACK. |

## 3. Offline behaviour

Every alert (and door event) is appended and fsync'd to `alert_log.jsonl` before any
network publish. `alert_log.synced` stores how many lines the ops centre has
acknowledged. When the uplink reconnects (paho auto-reconnect, 2–30 s back-off) the
backlog is replayed in order. Classification never depends on the uplink.

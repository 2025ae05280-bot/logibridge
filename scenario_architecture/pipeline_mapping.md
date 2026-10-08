# Pipeline mapping

| Stage | Implementation | Evidence |
|---|---|---|
| Sensor simulation | `data_pipeline.simulator.SensorSimulator.step` | contract payload checks |
| MQTT ingestion | `inference.inference_service.main` | sensor topic subscription |
| Validation | `InferenceService.ingest` | invalid-value counter/log |
| Timestamp buffering | `WindowFeatureExtractor.push` | timestamp window tests |
| Feature fusion | `WindowFeatureExtractor._features` | six named features |
| Normalisation | `data_pipeline.preprocessing.normalise` | `(2, 6)` stats |
| TFLite inference | `inference.engine.ModelRunner.predict` | runtime benchmark CSV |
| Decision safety | `inference.decision.Debouncer` | interlock/debounce checks |
| Offline custody | `inference.alert_store.AlertStore` | SQLite `windows` and `alerts` |
| Drift/uplink | `monitoring.drift_monitor` and `inference.sync_worker` | PSI/reference and sync rows |

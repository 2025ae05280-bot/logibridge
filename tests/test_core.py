from pathlib import Path

import numpy as np

from data_pipeline.preprocessing import WindowFeatureExtractor, load_stats, normalise, save_stats
from data_pipeline.simulator import TEMP_DRIFT_PER_READING, SensorSimulator
from inference.alert_store import AlertStore
from monitoring.drift_monitor import PSIDriftMonitor, proportions, psi, save_reference
from optimisation.benchmark import pareto_front
from training.convert_ptq import representative_data


def test_simulator_contract_and_vibration_rate():
    sim = SensorSimulator(seed=1)
    first = sim.step(0)
    second = sim.step(1)
    assert first[0][1].keys() == {"ts", "truck_id", "seq", "value"}
    assert any(stream == "vibration" for stream, _ in second)


def test_timestamped_window_extraction():
    extractor = WindowFeatureExtractor()
    output = []
    sim = SensorSimulator("temp_drift", seed=2)
    for ts in range(45):
        for stream, payload in sim.step(ts):
            output.extend(extractor.push(stream, payload["ts"], payload["value"]))
    assert output and output[0].shape == (6,)
    assert np.isfinite(output[0]).all()


def test_timestamp_reset_restarts_window_schedule():
    extractor = WindowFeatureExtractor()
    sim = SensorSimulator(seed=2)
    for ts in range(45):
        for stream, payload in sim.step(ts):
            extractor.push(stream, payload["ts"], payload["value"])
    sim = SensorSimulator(seed=2)
    output = []
    for ts in range(45):
        for stream, payload in sim.step(ts):
            output.extend(extractor.push(stream, payload["ts"], payload["value"]))
    assert output


def test_stats_round_trip(tmp_path):
    path = tmp_path / "stats.npy"
    save_stats(path, np.zeros(6), np.ones(6))
    mean, std = load_stats(path)
    np.testing.assert_allclose(normalise(np.ones(6), mean, std), np.ones(6))


def test_psi_identical_and_disjoint():
    clean = proportions([0.9] * 100)
    drift = proportions([0.1] * 100)
    assert psi(clean, clean) == 0
    assert psi(clean, drift) > 0.25


def test_simulator_uses_assignment_temperature_drift_rate():
    normal = SensorSimulator(seed=7)
    drift = SensorSimulator("temp_drift", seed=7)
    normal_temperature = normal.step(0)[0][1]["value"]
    drift_temperature = drift.step(0)[0][1]["value"]
    assert np.isclose(
        drift_temperature - normal_temperature,
        TEMP_DRIFT_PER_READING,
    )


def test_alert_store_keeps_unsynced_alerts_after_reopen(tmp_path):
    path = tmp_path / "alerts.db"
    store = AlertStore(path)
    alert_id = store.alert(1.0, "T01", 2, "CRITICAL", "model", [0.01, 0.02, 0.97])
    store.door_event(2.0, "T01", 1, "OPEN")
    store.close()

    reopened = AlertStore(path)
    assert [row[0] for row in reopened.unsynced_alerts()] == [alert_id]
    events = reopened.unsynced_door_events()
    assert [(row[2], row[3], row[4]) for row in events] == [("T01", 1, "OPEN")]
    reopened.mark_synced([alert_id])
    reopened.mark_door_events_synced([events[0][0]])
    assert reopened.unsynced_alerts() == []
    assert reopened.unsynced_door_events() == []
    reopened.close()


def test_psi_monitor_reports_zero_after_full_clean_window(tmp_path):
    path = tmp_path / "reference.json"
    scores = [0.9] * 300
    save_reference(path, scores)
    monitor = PSIDriftMonitor(path, window=100)
    assert monitor.calculate() is None
    for score in scores[:100]:
        monitor.add(score)
    assert monitor.calculate() == 0.0


def test_training_helpers_use_current_csv_artifacts():
    from training.common import DATASET_PATH, M1_KERAS, M3_TFLITE, load_dataset

    train_x, train_y, val_x, val_y = load_dataset()
    assert Path(DATASET_PATH).parts[-3:] == ("training", "data", "dataset.csv")
    assert Path(M1_KERAS).name == "m1_fp32.keras"
    assert Path(M3_TFLITE).name == "m3_pruned_int8.tflite"
    assert train_x.shape[1] == val_x.shape[1] == 6
    assert len(train_x) == len(train_y)
    assert len(val_x) == len(val_y)


def test_benchmark_marks_pareto_and_recommends_only_safe_variants():
    rows = [
        {
            "variant": "fast",
            "mean_latency_ms": 1.0,
            "size_kb": 4.0,
            "accuracy": 0.90,
            "critical_recall": 0.94,
        },
        {
            "variant": "safe",
            "mean_latency_ms": 2.0,
            "size_kb": 5.0,
            "accuracy": 0.91,
            "critical_recall": 0.96,
        },
        {
            "variant": "dominated",
            "mean_latency_ms": 3.0,
            "size_kb": 6.0,
            "accuracy": 0.89,
            "critical_recall": 0.97,
        },
    ]
    front = pareto_front(rows)
    assert {row["variant"] for row in front} == {"fast", "safe"}
    assert rows[1]["recommended"] is True
    assert rows[0]["recommended"] is False


def test_quantization_representative_data_covers_every_label(tmp_path):
    path = tmp_path / "dataset.csv"
    with path.open("w", newline="") as handle:
        handle.write(
            "temp_mean,temp_std,temp_roc_c_per_min,vib_rms,vib_peak,"
            "vib_kurtosis,label,split\n"
        )
        for label in range(3):
            for _ in range(100):
                handle.write(f"{label},0,0,0,0,0,{label},train\n")

    samples = list(representative_data(path, np.zeros(6), np.ones(6)))
    sampled_labels = {int(sample[0][0, 0]) for sample in samples}
    assert len(samples) == 250
    assert sampled_labels == {0, 1, 2}


def test_simulator_emits_door_events_only_on_transitions():
    sim = SensorSimulator(seed=7)
    events = [
        payload["value"]
        for tick in range(1200)
        for stream, payload in sim.step(tick)
        if stream == "door"
    ]
    assert events
    assert all(left != right for left, right in zip(events, events[1:]))

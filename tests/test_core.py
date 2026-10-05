import numpy as np

from data_pipeline.preprocessing import WindowFeatureExtractor, load_stats, normalise, save_stats
from data_pipeline.simulator import SensorSimulator
from monitoring.drift_monitor import proportions, psi


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

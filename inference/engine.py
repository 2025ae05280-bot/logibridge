import time
from pathlib import Path

import numpy as np

from common.tflite_shim import dequantise, make_interpreter, quantise
from data_pipeline.preprocessing import load_stats, normalise


class ModelRunner:
    def __init__(self, model_path, stats_path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        self.mean, self.std = load_stats(stats_path)
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def predict(self, features):
        values = normalise(np.asarray(features, dtype=np.float32), self.mean, self.std)
        shape = tuple(self.input["shape"])
        values = values.reshape(shape)
        started = time.perf_counter_ns()
        self.interpreter.set_tensor(self.input["index"], quantise(values, self.input))
        self.interpreter.invoke()
        raw = dequantise(self.interpreter.get_tensor(self.output["index"]), self.output).reshape(-1)
        if np.any(raw < 0) or raw.sum() <= 0 or not np.isclose(raw.sum(), 1.0, atol=1e-3):
            raw = np.exp(raw - np.max(raw))
        probs = raw / raw.sum()
        return probs.astype(np.float32), (time.perf_counter_ns() - started) / 1_000_000.0

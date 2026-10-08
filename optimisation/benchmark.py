import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common.config import REPO_ROOT
from common.tflite_shim import dequantise, make_interpreter, quantise
from data_pipeline.preprocessing import load_stats, normalise


def _runtime_name():
    try:
        import tflite_runtime
        return f"tflite-runtime {getattr(tflite_runtime, '__version__', 'unknown')}"
    except ImportError:
        import tensorflow as tf
        return f"tensorflow {tf.__version__}"


def validation_data(path, mean, std):
    with Path(path).open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "val"]
    x = np.asarray([[float(r[n]) for n in ("temp_mean", "temp_std", "temp_roc_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis")] for r in rows], dtype=np.float32)
    return normalise(x, mean, std), np.asarray([int(r["label"]) for r in rows])


def benchmark_variant(path, x, y, iterations=200, warmup=10):
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("benchmark requires non-empty validation features and matching labels")
    interpreter = make_interpreter(path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    sample = x[0].reshape(input_detail["shape"])
    for _ in range(warmup):
        interpreter.set_tensor(input_detail["index"], quantise(sample, input_detail))
        interpreter.invoke()
    cpu_before = time.process_time_ns()
    started = time.perf_counter_ns()
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        interpreter.set_tensor(input_detail["index"], quantise(sample, input_detail))
        interpreter.invoke()
        latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
    wall = (time.perf_counter_ns() - started) / 1_000_000_000.0
    cpu_seconds = (time.process_time_ns() - cpu_before) / 1_000_000_000.0
    cpu_fraction = max(
        0.0,
        cpu_seconds / max(wall * max(1, os.cpu_count() or 1), 1e-9),
    )
    predictions = []
    for sample in x:
        interpreter.set_tensor(input_detail["index"], quantise(sample.reshape(input_detail["shape"]), input_detail))
        interpreter.invoke()
        predictions.append(int(np.argmax(dequantise(interpreter.get_tensor(output_detail["index"]), output_detail))))
    predictions = np.asarray(predictions)
    mean_latency = float(np.mean(latencies))
    laptop_tdp_w = float(os.getenv("LAPTOP_TDP_W", "45"))
    energy = laptop_tdp_w * cpu_fraction * mean_latency / 1000.0 * 1000.0
    critical_count = int(np.sum(y == 2))
    if critical_count == 0:
        raise ValueError("held-out validation set has no Critical examples")
    return {"mean_latency_ms": mean_latency, "p50_latency_ms": float(np.percentile(latencies, 50)), "p95_latency_ms": float(np.percentile(latencies, 95)), "p99_latency_ms": float(np.percentile(latencies, 99)), "size_kb": os.path.getsize(path) / 1024.0, "accuracy": float(np.mean(predictions == y)), "critical_recall": float(np.sum((predictions == 2) & (y == 2)) / critical_count), "energy_mj": energy, "runtime": _runtime_name()}


def pareto_front(rows):
    """Mark variants not dominated on latency, size, and validation accuracy."""
    for candidate in rows:
        candidate["pareto_optimal"] = not any(
            other is not candidate
            and other["mean_latency_ms"] <= candidate["mean_latency_ms"]
            and other["size_kb"] <= candidate["size_kb"]
            and other["accuracy"] >= candidate["accuracy"]
            and (
                other["mean_latency_ms"] < candidate["mean_latency_ms"]
                or other["size_kb"] < candidate["size_kb"]
                or other["accuracy"] > candidate["accuracy"]
            )
            for other in rows
        )
    eligible = [
        row for row in rows
        if row["accuracy"] > 0.88 and row["critical_recall"] > 0.95
    ]
    recommended = min(
        eligible,
        key=lambda row: (row["mean_latency_ms"], row["size_kb"], -row["accuracy"]),
        default=None,
    )
    for row in rows:
        row["recommended"] = row is recommended
    return [row for row in rows if row["pareto_optimal"]]


def write_chart(csv_path, chart_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(csv.DictReader(Path(csv_path).open(newline="")))
    if not rows:
        return
    accuracy = np.asarray([float(r["accuracy"]) for r in rows])
    latency = np.asarray([float(r["mean_latency_ms"]) for r in rows])
    size = np.asarray([float(r["size_kb"]) for r in rows])
    pareto = np.asarray([r["pareto_optimal"].lower() == "true" for r in rows])
    recommended = np.asarray([r["recommended"].lower() == "true" for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, (x, label) in enumerate(((latency, "Mean latency (ms)"), (size, "Model size (KB)"))):
        axes[index].scatter(x[~pareto], accuracy[~pareto], label="dominated")
        axes[index].scatter(x[pareto], accuracy[pareto], label="Pareto optimal")
        for i, row in enumerate(rows):
            axes[index].annotate(row["variant"], (x[i], accuracy[i]))
        if pareto.sum() > 1:
            order = np.argsort(x[pareto])
            axes[index].plot(x[pareto][order], accuracy[pareto][order], linestyle="--", alpha=0.5)
        if recommended.any():
            axes[index].scatter(
                x[recommended],
                accuracy[recommended],
                marker="*",
                s=180,
                label="recommended (passes safety gates)",
            )
        axes[index].set_xlabel(label)
        axes[index].set_ylabel("Held-out validation accuracy")
        axes[index].legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    mean, std = load_stats(REPO_ROOT / "data_pipeline" / "training_stats.npy")
    x, y = validation_data(REPO_ROOT / "training" / "data" / "dataset.csv", mean, std)
    model_dir = REPO_ROOT / "training" / "models"
    rows = []
    for variant, filename in (("m1_fp32", "m1_fp32.tflite"), ("m2_ptq_int8", "m2_ptq_int8.tflite"), ("m3_pruned_int8", "m3_pruned_int8.tflite")):
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"required benchmark variant {variant} is missing: {path}; "
                "generate all model variants before benchmarking"
            )
        rows.append({"variant": variant, **benchmark_variant(path, x, y, args.iterations)})
    pareto_front(rows)
    output = REPO_ROOT / "optimisation" / "results" / "benchmark_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "size_kb", "accuracy", "critical_recall", "energy_mj", "runtime", "pareto_optimal", "recommended"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_chart(output, output.parent / "pareto_chart.png")


if __name__ == "__main__":
    main()

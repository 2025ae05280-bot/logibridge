# optimisation/benchmark.py
"""Five-metric benchmark of M1 / M2 / M3 (Lab 2 methodology).

For each TFLite variant:
  1. mean latency (ms)  - 200 timed single-sample invocations after 10 warm-up runs
  2. p95 latency (ms)
  3. model file size (KB)
  4. accuracy on the held-out validation set (%)  (+ Class 2 Critical recall)
  5. energy per inference (mJ) - E = P x t, P = TDP x process CPU share (psutil)

Inputs are real validation windows (cycled), not random tensors. INT8 models get
their inputs quantised with the model's own scale / zero-point.

Usage: python optimisation/benchmark.py [--tdp 15] [--runs 200] [--warmup 10]
Writes optimisation/results/benchmark_results.csv and pareto_chart.png
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import psutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT, "training"))
from common import (load_dataset, make_interpreter, tflite_predict_one, evaluate_probs,
                    M1_TFLITE, M2_TFLITE, M3_TFLITE)

RESULTS_DIR = os.path.join(ROOT, "optimisation", "results")
VARIANTS = [("M1_FP32_Baseline", M1_TFLITE),
            ("M2_PTQ_INT8", M2_TFLITE),
            ("M3_Pruned_PTQ_INT8", M3_TFLITE)]


def benchmark_variant(model_path, X_val, y_val, tdp_w, runs, warmup):
    interp = make_interpreter(model_path, num_threads=1)

    for i in range(warmup):
        tflite_predict_one(interp, X_val[i % len(X_val)])

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)                 # prime the counter
    latencies = []
    t_start = time.perf_counter()
    for i in range(runs):
        x = X_val[i % len(X_val)]
        t0 = time.perf_counter()
        tflite_predict_one(interp, x)               # includes (de)quantisation, as in production
        latencies.append((time.perf_counter() - t0) * 1000.0)
    wall_s = time.perf_counter() - t_start
    cpu_share = proc.cpu_percent(interval=None) / 100.0 / psutil.cpu_count()

    lat = np.array(latencies)
    mean_ms = float(lat.mean())
    power_w = tdp_w * cpu_share
    energy_mj = power_w * (mean_ms / 1000.0) * 1000.0

    probs = np.array([tflite_predict_one(interp, x) for x in X_val])
    acc, recall, _ = evaluate_probs(probs, y_val)

    return {
        "Mean_Latency_ms": round(mean_ms, 4),
        "p95_Latency_ms": round(float(np.percentile(lat, 95)), 4),
        "Size_KB": round(os.path.getsize(model_path) / 1024.0, 2),
        "Accuracy_Pct": round(acc * 100, 2),
        "Energy_mJ": round(energy_mj, 6),
        "Class2_Recall_Pct": round(recall[2] * 100, 2),
        "CPU_Share_Pct": round(cpu_share * 100, 2),
        "Bench_Wall_s": round(wall_s, 3),
    }


def pareto_front(rows):
    """Variants not dominated on (latency lower, size lower, accuracy higher)."""
    front = []
    for a in rows:
        dominated = any(
            b is not a
            and b["Mean_Latency_ms"] <= a["Mean_Latency_ms"]
            and b["Size_KB"] <= a["Size_KB"]
            and b["Accuracy_Pct"] >= a["Accuracy_Pct"]
            and (b["Mean_Latency_ms"] < a["Mean_Latency_ms"] or b["Size_KB"] < a["Size_KB"]
                 or b["Accuracy_Pct"] > a["Accuracy_Pct"])
            for b in rows)
        a["Pareto_Optimal"] = not dominated
        if not dominated:
            front.append(a)
    return front


def plot_pareto(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    front = sorted([r for r in rows if r["Pareto_Optimal"]], key=lambda r: r["Mean_Latency_ms"])
    if len(front) > 1:
        ax.plot([r["Mean_Latency_ms"] for r in front], [r["Size_KB"] for r in front],
                color=ink2, lw=1.5, ls="--", zorder=1, label="Pareto frontier")

    # Label above each point, except the smallest model whose label goes below it
    smallest = min(range(len(rows)), key=lambda i: rows[i]["Size_KB"])
    offsets = {i: (10, -40) if i == smallest else (10, 8) for i in range(len(rows))}
    for i, (r, c) in enumerate(zip(rows, colors)):
        ax.scatter(r["Mean_Latency_ms"], r["Size_KB"], s=110, color=c, edgecolor="#fcfcfb",
                   linewidth=2, zorder=3, label=r["Variant"])
        ax.annotate(f'{r["Variant"]}\nacc {r["Accuracy_Pct"]:.1f}% · C2 recall {r["Class2_Recall_Pct"]:.0f}%'
                    f'\n{r["Energy_mJ"]:.4f} mJ/inf',
                    (r["Mean_Latency_ms"], r["Size_KB"]), textcoords="offset points",
                    xytext=offsets[i], fontsize=8, color=ink)

    ax.set_xlabel("Mean inference latency (ms) — lower is better", color=ink2)
    ax.set_ylabel("Model file size (KB) — lower is better", color=ink2)
    ax.set_title("LogiEdge model variants: latency vs size (Pareto frontier)", color=ink, loc="left")
    ax.grid(True, color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2)
    xs = [r["Mean_Latency_ms"] for r in rows]
    ys = [r["Size_KB"] for r in rows]
    ax.set_xlim(0, max(xs) * 1.6)
    ax.set_ylim(0, max(ys) * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdp", type=float, default=15.0, help="Laptop CPU TDP estimate in watts")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    _, _, X_val, y_val = load_dataset()
    print(f">> Benchmarking {len(VARIANTS)} variants | {args.runs} runs after {args.warmup} warm-up | "
          f"TDP={args.tdp} W | {len(X_val)} validation windows")

    rows = []
    for name, path in VARIANTS:
        if not os.path.exists(path):
            sys.exit(f"Missing {path} - run train_model.py, convert_ptq.py, prune_quantise.py first.")
        r = {"Variant": name, **benchmark_variant(path, X_val, y_val, args.tdp, args.runs, args.warmup)}
        rows.append(r)
        print(f"   {name:<20} mean={r['Mean_Latency_ms']:.4f}ms p95={r['p95_Latency_ms']:.4f}ms "
              f"size={r['Size_KB']:.2f}KB acc={r['Accuracy_Pct']:.2f}% "
              f"E={r['Energy_mJ']:.6f}mJ C2recall={r['Class2_Recall_Pct']:.2f}%")

    pareto_front(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    png_path = os.path.join(RESULTS_DIR, "pareto_chart.png")
    plot_pareto(rows, png_path)
    print(f">> Pareto-optimal: {[r['Variant'] for r in rows if r['Pareto_Optimal']]}")
    print(f">> Wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()

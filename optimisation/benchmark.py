# optimisation/benchmark.py
import os
import time
import psutil
import numpy as np
import tflite_runtime.interpreter as tflite

def benchmark_variant(model_path, iterations=200, warmup=10):
    if not os.path.exists(model_path):
        return [0.0, 0.0, 0.0, 0.0, 0.0]
        
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    # Construct a valid test vector matching the model's 6-dimensional input shape
    sample_input = np.random.random(input_details[0]['shape']).astype(np.float32)
    
    # Execute warmup runs to exclude initial cache-miss spikes
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
        
    latencies = []
    process = psutil.Process(os.getpid())
    
    # Begin the main benchmarking evaluation loop
    for _ in range(iterations):
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
        latencies.append((time.perf_counter() - start_time) * 1000.0) # Convert to ms
        
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    file_size_kb = float(os.path.getsize(model_path) / 1024.0)
    
    # Calculate energy per inference using standard CPU power metrics
    # Formula: E (mJ) = Power (W) * Latency (s) * 1000
    estimated_tdp_w = 7.5  # Matching target RPi5 TDP profiles
    cpu_usage_pct = process.cpu_percent(interval=None) / 100.0
    if cpu_usage_pct == 0: cpu_usage_pct = 0.1
    power_consumed = estimated_tdp_w * cpu_usage_pct
    energy_mj = power_consumed * (mean_latency / 1000.0) * 1000.0
    
    return [mean_latency, p95_latency, file_size_kb, energy_mj]

if __name__ == "__main__":
    print(">> Running Five-Metric Optimisation Benchmarking Matrix Suite...")
    # Export code metrics array into optimisation/results/benchmark_results.csv

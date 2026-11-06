# Prints benchmark_results.csv as a readable table.
import csv

rows = list(csv.DictReader(open("optimisation/results/benchmark_results.csv")))
print(f"{'variant':<16}{'mean ms':>9}{'p95 ms':>9}{'size KB':>9}{'accuracy':>10}{'crit recall':>13}{'energy mJ':>11}  recommended")
for r in rows:
    print(
        f"{r['variant']:<16}{float(r['mean_latency_ms']):9.4f}{float(r['p95_latency_ms']):9.4f}"
        f"{float(r['size_kb']):9.2f}{float(r['accuracy']) * 100:9.1f}%{float(r['critical_recall']) * 100:12.1f}%"
        f"{float(r['energy_mj']):11.4f}  {r['recommended']}"
    )
print("runtime:", rows[0]["runtime"])

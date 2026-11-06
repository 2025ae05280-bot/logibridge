#!/bin/bash
# Segment 5: Component F benchmark of M1/M2/M3 and the Pareto chart.
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh

step "Benchmark: 10 warm-up + 200 timed runs per variant" '$PY optimisation/benchmark.py 2>/dev/null'
step "Results table" '$PY demo/show_benchmark.py'
step "Pareto chart" 'open optimisation/results/pareto_chart.png'
say "Segment 5 done."

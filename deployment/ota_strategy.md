# OTA strategy

The rubric's planning estimate uses a 280 KB INT8 model: one full rollout is `280 KB × 85` trucks, approximately ₹2.32 at ₹0.10/MB. Canary and shadow use the same model transfer; canary adds rollback transfers and shadow adds comparison-log uplink. The measured value must come from `optimisation/results/benchmark_results.csv`, not this planning estimate.

Canary is the default for a safety system: deploy to a small truck cohort, verify recall, connectivity, and watchdog health, then expand. Full rollout is simplest but gives the largest simultaneous failure domain. Shadow is useful for comparing a retrained model but should not control alerts until it passes the canary gate.

# Hardware justification

## Constraint triangle and selection

| Option | Performance and 90-second SLA | AI power versus 10 W budget | Pilot cost (85) | Fleet cost (265) |
|---|---|---|---:|---:|
| Raspberry Pi 5 + AI HAT+ | Ample for the six-feature model; local inference avoids cellular RTT | 7.5 W: within budget | ₹12.75 lakh | ₹39.75 lakh |
| Jetson Orin Nano Super | More than required for this small classifier | 15 W at moderate load: exceeds budget; its 7 W mode must be benchmarked for the workload | ₹38.25 lakh | ₹119.25 lakh |
| STM32H7 | Model execution may fit, but the full Linux/MQTT/Docker/SQLite service does not | 0.4 W: within budget | ₹2.98 lakh | ₹9.28 lakh |

The constraint triangle is dominated by fitting useful edge performance into the power envelope at a viable fleet cost. The Pi 5 option is the pilot choice: its quoted 7.5 W is below 10 W, and 85 pilot units cost substantially less than Jetson while retaining the Linux resources needed for local MQTT, TFLite, durable alert storage, and OTA management. At 265 vehicles it costs about ₹39.75 lakh. Jetson offers headroom that this six-input MLP does not need and its 15 W moderate-load point violates the specified budget; the 7 W mode is a possible alternative only after measuring performance and power on the deployed workload. STM32H7 is the lowest-cost, lowest-power device, but selecting it would require redesigning the service stack instead of simply deploying this Python container.

The full application must detect and alert within 90 seconds, including window formation, decision debounce, and local publish. Consequently, inference is not allowed to wait for cloud RTT. The current container benchmark is useful for software smoke testing, but the assignment's hardware claim must use a measurement on the selected Pi.

## Arithmetic intensity and Roofline

Using the assignment's stated estimates, arithmetic intensity is `45 MFLOP / 18 MB = 2.5 FLOP/byte`. The Pi CPU ridge point is `16 GFLOP/s / 12 GB/s ≈ 1.33 FLOP/byte`. Since 2.5 exceeds 1.33, the specified workload is compute-bound in the Roofline model; reducing operations or using a faster compute path is the first latency optimization to investigate, while quantisation may additionally reduce memory traffic. These are calculations from the brief's estimates, not a claim that the measured model performs 45 MFLOP. The trained MLP is much smaller, so the final device recommendation should be based on the benchmark and power measurements of the actual artifact.

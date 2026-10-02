# Component B — Hardware Selection and Justification
## Task B1 — Constraint Triangle Application

### 1. Dominant Constraint Vertex
The dominant vertex of the Edge AI Constraint Triangle for FreightBridge's cold-chain deployment is Power/Thermal Management (TDP), tightly bounded by Fleet-Scale Unit Cost. Operating within an unventilated vehicle cabin drawing power from a 12V truck battery bus via a DC-DC step-down converter, the hardware stack must respect a rigid 10W AI power budget. 

### 2. Comparative Matrix
* Raspberry Pi 5 + Hailo-8L (~₹15,000 / truck | 7.5W TDP): Fits the strict 10W envelope and balances full-scale 265-truck capitalization metrics comfortably at ₹39.75 Lakhs.
* Jetson Orin Nano (~₹45,000 / truck | 15W TDP): Fails the power envelope, requiring heavy active structural integration, and driving fleet cap costs to an excessive ₹1.19 Crores.
* STM32H7 MCU (~₹3,500 / truck | 0.4W TDP): Fails on performance; unable to deploy native Linux, local brokers, Docker container stacks, or sliding feature window buffers.

### 3. Conclusion
The Raspberry Pi 5 + Hailo-8L co-processor setup is selected. The Hailo-8L delivers 13 TOPS of accelerated matrix compute, guaranteeing sub-millisecond execution times that satisfy our real-time 90-second SLA while drawing only 7.5W.

## Task B2 — Arithmetic Intensity and Roofline Analysis
Standardizing our modeling specifications:
* Model Compute Load (W): 45 MFLOPs = 45 × 10^6 FLOPs
* Memory Access Traffic (Q): 18 MB = 18 × 10^6 Bytes
* Peak Compute Bandwidth (P_max): 16 GFLOP/s (Pi 5 NEON SIMD)
* Memory Read Bandwidth (B_max): 12 GB/s (LPDDR4X Bus)

### 1. Calculations
* Operational Arithmetic Intensity (I) = W / Q = (45 × 10^6) / (18 × 10^6) = 2.50 FLOP/Byte
* Hardware Inherent Ridge Point = P_max / B_max = (16 × 10^9) / (12 × 10^9) = 1.33 FLOP/Byte

### 2. Roofline Classification
Since our Operational Intensity (2.50) is greater than the Ridge Point (1.33), the model is strictly **Compute-Bound**. Performance is capped by CPU clock speeds rather than bus transfer delays. To reduce execution latency, we must implement compute-focused optimizations like Structured Weight Pruning (removing 35% of low-magnitude filter paths), shifting processing cycles down the compute ceiling without hitting memory bottlenecks.

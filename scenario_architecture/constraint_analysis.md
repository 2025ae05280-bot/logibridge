# Component A — System Architecture and Justification
## Task A1 — Constraint Analysis

### 1. Latency Constraints & Cloud Feasibility
The thermal dynamics of FreightBridge’s pharmaceutical cargo impose a strict safety threshold: a refrigeration unit failure can escalate compartment temperatures by 1°C per minute, requiring system detection and escalation within 90 seconds of an anomaly's signature. Offloading this inference payload to a centralized cloud architecture is structurally unfeasible.

In rural Indian cellular environments (covering transit zones across rural Maharashtra and Andhra Pradesh hill tracts), round-trip time (RTT) latency profiles are highly volatile and non-deterministic. Network latencies regularly surge from a baseline of 150ms to over 15,000ms, accompanied by persistent packet drops and unbounded TCP retransmission timeouts. Cloud inference cannot guarantee a deterministic real-time response under these conditions.

By executing the preprocessing and model evaluation loops entirely locally on the truck's edge node, inference is decoupled from network fluctuations. This guarantees a deterministic execution latency of sub-100 milliseconds, enabling immediate safety triggers well within the mandated 90-second SLA.

### 2. Bandwidth Bottlenecks & Economic Quantification
Each of the 85 refrigerated trucks generates continuous telemetry across three sensor streams: temperature at 1 Hz, 3-axis vibration at 500 Hz, and discrete door state events. Assuming standard 32-bit floating-point formatting (4 bytes per sample), the raw data generated per truck is calculated below:
* Temperature: 1 Hz × 4 bytes = 4 bytes/sec
* Vibration (3-axis): 500 Hz × 3 axes × 4 bytes = 6,000 bytes/sec
* Total Stream Volume: 6,004 bytes/sec

Over a continuous 24-hour operational cycle, a single vehicle accumulates: 6,004 bytes/sec × 86,400 seconds/day = 518,745,600 bytes ≈ 518.75 MB/truck/day

At the current M2M SIM data tariff of ₹0.10 per MB, transmitting raw telemetry to the cloud costs ₹51.88 per truck everyday. For the planned 265-truck fleet scale-up, this equates to a recurring bandwidth expenditure of ₹13,748 per day, or approximately ₹50.18 Lakhs annually.

In contrast, our Edge AI architecture processes raw streams locally, restricting cellular transmission to small, discrete MQTT alert packets (approx. 1 KB per alert state change). Assuming a conservative average of 10 alerts per truck daily, the fleet-wide data transmission plummets to less than 3 MB per day, reducing annual cellular operational costs to under ₹150.

### 3. Connectivity Reliability & Offline Resiliency
The critical Nashik–Aurangabad transit corridor contains seven documented cellular blind spots where signal coverage drops completely for 35 to 90 minutes. In a cloud-dependent architecture, a signal drop immediately halts the data pipeline: the cloud backend loses visibility, no inference can occur, and any mid-route thermal breach or mechanical failure goes completely undetected until cellular connection returns—directly risking massive cargo spoilage events like the historical ₹28 Lakh vaccine loss.

The LogiEdge decentralized architecture resolves this by isolating the core pipeline on the edge node. During network dropouts, sensor ingestion, feature extraction, and model classifications carry on uninterrupted. If a Warning or Critical state is flagged, the system appends the alert along with its corresponding window telemetry to an encrypted local SQLite/LevelDB alert log. When cellular coverage returns, a persistent edge sync manager re-establishes the uplink and pushes the backlogged alert queue to the operations centre using MQTT persistent sessions with a Quality of Service (QoS) Level 1 tier.

### 4. Data Privacy & Regulatory Compliance
FreightBridge’s pharmaceutical clients demand absolute cryptographic and operational verification that sensitive cargo metrics cannot be intercepted, manipulated, or viewed by unauthorized third parties during transit.

Transmitting raw, unencrypted time-series data over public cellular infrastructure introduces a wide attack surface, exposing the fleet to Man-in-the-Middle (MitM) positioning exploits, data scraping, and corporate espionage. On-device edge inference contains the raw sensor data completely within the physical boundaries of the vehicle's hardware node.

Only high-level status labels and system health updates leave the truck, drastically reducing the threat landscape. This architecture ensures complete compliance with strict cold-chain custody frameworks, giving clients solid proof that their environmental data remains highly secure and private.
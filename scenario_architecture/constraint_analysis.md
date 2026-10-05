# Constraint analysis

The cold-chain fault model allows roughly 1 °C/min of temperature rise and requires detection plus escalation within 90 seconds. A 30-second feature window, a 10-second emit step, and a two-of-three debounce consume most of that budget; model execution is therefore required to be much less than one second. Running ingestion and inference on the truck removes cellular RTT from the safety path.

The planned coverage gaps are 35–90 minutes. A cloud-only design detects 0% of faults that begin wholly inside such a gap within the 90-second SLA. The edge node continues classifying locally and stores alerts plus compact per-window summaries in SQLite; the sync worker forwards unsynced alert IDs after coverage returns.

For the bandwidth estimate, 1 Hz temperature at 4 bytes/sample plus 500 Hz, three-axis vibration at 4 bytes/sample is 6,004 bytes/s, or about 518.75 MB/day before JSON, timestamps, and door-event overhead. At ₹0.10/MB that is ₹51.88 per truck/day. The 85-truck pilot is about ₹4,410/day, or ₹16.1 lakh/year; the 265-truck scale-up is about ₹13,748/day, or ₹50.18 lakh/year. Feature windows and state-change alerts are far smaller than raw waveform upload.

The broker is bound to loopback by Compose and persistence is enabled. The lab configuration allows anonymous local connections so the clean-clone demo starts; production deployment must add a password file, ACLs, and TLS. The SQLite file is plain SQLite in the lab; filesystem encryption or SQLCipher is a production requirement. These are explicit boundaries, not claims that the lab already provides cryptographic protection.

An edge node also avoids making a safety decision after a reconnect. The trade-off is local storage and OTA responsibility: model, stats, reference distribution, health state, and alert schema must be versioned together. The deployment playbook copies those artifacts idempotently and uses a health check before declaring the container ready.

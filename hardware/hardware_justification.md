# Hardware justification

The 90-second SLA sets the useful latency budget: after the physical ramp, window lag, and N-of-M confirmation, each inference should be comfortably below one second. A Pi 5 has ample RAM and storage for this small MLP plus the Python/MQTT/SQLite stack. For an 85-truck pilot, the planning costs are ₹12.75 lakh for Pi 5, ₹38.25 lakh for Jetson Orin Nano, and ₹2.98 lakh for STM32H7.

Jetson's 7 W mode is valid, but its cost and performance are excessive for a six-feature MLP. STM32H7 has about 2 MB Flash and 1 MB SRAM, enough for the model, but not for the Linux container, broker, OTA, and SQLite runtime required by this project. Hailo-8L would require a compiled HEF; it does not execute this TFLite file directly.

The brief's 45 MFLOP estimate is a planning figure. The actual dense model is closer to 1.5 kFLOP per inference, so measured benchmark output—not the planning figure—must drive the final hardware claim.

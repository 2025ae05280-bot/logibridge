# Variant recommendation

Run `python optimisation/benchmark.py` after generating M1–M3. It uses 10 warm-up invocations and 200 timed invocations per variant, then evaluates accuracy and Critical recall on the held-out validation split. The CSV is the only source for latency, size, accuracy, Critical recall, energy, and runtime. Choose the smallest/fastest variant that still passes validation accuracy > 0.88 and Critical recall > 0.95; do not hand-enter benchmark values.

The 90-second budget is dominated by the physical fault ramp and window/debounce confirmation. Inference itself must remain far below one second. The model is small enough for Pi storage/RAM and STM32H7 Flash/SRAM; the STM32 is rejected for the Linux, Docker/OTA, broker, and SQLite stack rather than model memory.

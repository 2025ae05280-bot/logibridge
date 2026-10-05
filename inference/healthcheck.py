import os
import time
from pathlib import Path

heartbeat = Path(os.getenv("HEARTBEAT_FILE", "/tmp/logibridge-heartbeat"))
if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 60:
    raise SystemExit(1)

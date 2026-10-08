"""Fail if a report number is not listed in results/RESULTS.md."""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = (root / "results" / "RESULTS.md").read_text() if (root / "results" / "RESULTS.md").exists() else ""
numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", results))
missing = []
for path in (root / "reports").rglob("*") if (root / "reports").exists() else []:
    if path.is_file() and path.suffix in {".md", ".txt"}:
        for number in set(re.findall(r"\b\d+(?:\.\d+)?\b", path.read_text())):
            if number not in numbers:
                missing.append(f"{path}: {number}")
if missing:
    raise SystemExit("untracked report numbers:\n" + "\n".join(missing))

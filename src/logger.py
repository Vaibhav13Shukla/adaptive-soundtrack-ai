"""
Lightweight JSON experiment logger.
No MLflow, no W&B. Just a JSON file that survives crashes.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class ExperimentLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []

        # Resume if log exists
        if self.log_path.exists():
            with open(self.log_path) as f:
                self.records = json.load(f)
            print(f"✓ Resumed log with {len(self.records)} records")

    def log(self, **kwargs):
        rec = {"timestamp": datetime.now().isoformat(), **kwargs}
        self.records.append(rec)
        # Write every time — safe against crashes
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def best(self, metric: str = "val_loss", mode: str = "min") -> Optional[dict]:
        if not self.records:
            return None
        valid = [r for r in self.records if metric in r]
        if not valid:
            return None
        fn = min if mode == "min" else max
        return fn(valid, key=lambda r: r[metric])

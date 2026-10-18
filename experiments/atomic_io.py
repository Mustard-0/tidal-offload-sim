"""Bounded retries for transient Windows readers blocking atomic replacement."""
from pathlib import Path
import time


def replace_file(source, target, attempts=10, delay_s=.05):
    if attempts < 1:
        raise ValueError("At least one replacement attempt is required")
    for attempt in range(attempts):
        try:
            return Path(source).replace(target)
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt+1 == attempts:
                raise
            time.sleep(delay_s)

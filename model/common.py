"""Portable IO and immutable run manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def atomic_replace(tmp, path):
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.05)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    atomic_replace(tmp, path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
    atomic_replace(tmp, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))}


def seed_all(seed, threads=2):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def manifest(config, job):
    return {"config": config, "job": job, "config_hash": digest(config),
            "source_hashes": source_hashes(), "python": platform.python_version(),
            "platform": [platform.system(), platform.machine(), list(platform.libc_ver())],
            "numpy_version": np.__version__, "evidence": "single-training-seed pilot",
            "privacy_scope": "one fixed-public-membership region batch, one report per user; not whole-trajectory DP"}

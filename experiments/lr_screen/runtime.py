"""Common imports, atomic evidence and complete CPU/CUDA RNG persistence."""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".runtime"), str(ROOT / "model")]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def source_hashes():
    files = list((ROOT / "model").glob("*.py")) + [ROOT / "model/config.json"]
    files += list((ROOT / "experiments/lr_screen").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(files)}


def runtime_info(device):
    import scipy
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                device=str(device), torch_threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                # CPU checkpoints must not initialize a CUDA context just to
                # collect unused GPU RNG states.
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def rng_load(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])


def checkpoint_write(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def available_ram_gib():
    if os.name != "nt":
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(fields["MemAvailable"].split()[0])*1024/2**30
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong) for k in ("total", "available", "total_page", "available_page",
                                            "total_virtual", "available_virtual", "extended")]
    s = MemoryStatus()
    s.length = ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise OSError("Cannot inspect memory")
    return s.available/2**30

"""Fail closed unless a source/plan/runtime-locked successful preflight exists."""
import json
from .protocol import ROOT, plan, digest

APPROVAL = ROOT / "evidence/integration_20261026/formal_preflight.json"


class TransferUnavailable(RuntimeError):
    pass


def require_transfer():
    from .runtime import source_hashes, runtime_info, torch
    torch.set_num_threads(1)
    if not APPROVAL.exists():
        raise TransferUnavailable("Transfer received; formal training awaits successful source-locked preflight")
    result = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if (result.get("passed") is not True or result.get("source_hashes") != source_hashes()
            or result.get("plan_hash") != digest(plan())
            or result.get("runtime") != runtime_info("cpu")):
        raise TransferUnavailable("Missing, failed or stale source/config/runtime preflight")
    return result

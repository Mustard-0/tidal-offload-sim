"""Frozen six-baseline paired experiment; no TT-MAPPO training or tuning."""

import hashlib
from pathlib import Path

import fresh_baseline_protocol as archived


BASE = "8302d78daa61f06df3572e9b1fa5dc7a055fabd4"
PARENT_RUN = 44993119344
CAMPAIGN = "stin-public-baseline-updates-20261122"
BRANCH = "experiment/" + CAMPAIGN
WORKFLOW = ".github/workflows/stin_public_baseline_updates_20261122.yml"
LEARNED = ("PPO", "SD-SAC-Adapted", "MARATD3-Adapted")
RULES = archived.RULES
SEEDS = archived.SEEDS
CELLS = tuple((m, s) for m in LEARNED for s in SEEDS) + tuple((m, 0) for m in RULES)
POPS = (20, 40, 60, 80, 100)
TRACES, WARM, MEASURE = archived.TRACES, archived.WARM, archived.MEASURE
TRAIN_USERS, TRAIN_ADMIT, TRAIN_SLOTS = archived.TRAIN_USERS, archived.TRAIN_ADMIT, archived.TRAIN_SLOTS
BLOCK, RATIOS = archived.BLOCK, archived.RATIOS
WALL = 1200
INHERITED = tuple(dict.fromkeys((*archived.INHERITED, "fresh_baseline_learners.py",
                               "fresh_baseline_protocol.py", "fresh_baseline_worker.py")))
cell_id, settings = archived.cell_id, archived.settings
digest, packet_body = archived.digest, archived.packet_body


def read_packet(release, commit):
    assert release["tag_name"] == CAMPAIGN
    return archived.read_packet(dict(release, tag_name=archived.CAMPAIGN), commit)


def plan(project=None, overlay=None):
    root = Path(project) if project else Path(__file__).resolve().parent
    overlay = Path(overlay) if overlay else root

    def fingerprint(name):
        p = overlay / name
        data = (p if p.is_file() else root / name).read_bytes()
        return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()

    prior = archived.plan(project, overlay)
    return dict(schema="public-baseline-updates-v1", parent=BASE, anchor=PARENT_RUN, cells=CELLS,
        train=prior["train"], test=dict(prior["test"], users=POPS, reused_for_exploratory_comparison=True),
        same_physics=True, same_original_deadlines=True, gpu=False, paid=False, wall=WALL,
        changes={
            "PPO": "CleanRL policy/value objectives; value fitting on both existing actor epochs",
            "SD-SAC-Adapted": "discrete SDSAC batchwise critic max and expected Q plus entropy",
            "MARATD3-Adapted": "TD3 twin minimum and clipped score noise before legal discrete projection; tau .005",
            "rules": "original Local-Only and shared Greedy policy; unchanged",
        },
        inherited_review_hashes={p: fingerprint(p) for p in INHERITED})

"""Matched replay of the original seven-baseline grid after two scoped fixes."""

from hashlib import sha256
from pathlib import Path

import seven_scale_protocol as prior
from seven_scale_protocol import (
    BLOCK, CELLS, EVAL_CPUS, FULL_MEASURE, FULL_TRACES, FULL_WALL, FULL_WARM, LEARNED,
    PART_ENDS, POPS, PREVIEW_MEASURE, PREVIEW_TRACES, PREVIEW_WALL, PREVIEW_WARM, RATIOS, RULES,
    SEEDS, TRAIN_ADMIT, TRAIN_SLOTS, TRAIN_USERS, cell_id, digest, packet_body, receipt_name, settings,
)


BASE = "d05822e13b537826b1c1cd45255a19aaa8f1065b"
PARENT_RUN = 45026002944
CAMPAIGN = "stin-baseline-route-repair-20261123"
BRANCH = "experiment/" + CAMPAIGN
WORKFLOW = ".github/workflows/stin_baseline_route_repair_20261123.yml"
ADDED_INHERITED = ("seven_scale_protocol.py", "seven_scale_worker.py")
INHERITED = (*prior.INHERITED, *ADDED_INHERITED)


def read_packet(release, commit):
    assert release["tag_name"] == CAMPAIGN
    return prior.read_packet(dict(release, tag_name=prior.CAMPAIGN), commit)


def plan(project=None, overlays=()):
    state = prior.plan(project, overlays)
    roots = tuple(map(Path, overlays)) + (Path(project) if project else Path(__file__).resolve().parent,)
    for name in ADDED_INHERITED:
        p = next(r / name for r in roots if (r / name).is_file())
        state["inherited_review_hashes"][name] = sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return dict(state, schema="baseline-task-route-and-telemetry-repair-v1", parent=BASE, anchor=PARENT_RUN,
        changes={"MARATD3-Adapted": "GRU actor reads each decision's live task inputs and masks; replay retains them",
                 "all_methods": "queue and utilization counters sampled exactly once per measured slot",
                 "unchanged": "actions available, arrivals, capacities, team reward, deadlines, seeds, rollout/optimizer budgets"})

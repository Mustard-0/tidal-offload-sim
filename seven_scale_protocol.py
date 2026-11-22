"""One fixed seven-method population comparison, with separate preview and full-tide inputs."""

from hashlib import sha256
from pathlib import Path

import public_baseline_protocol as prior


BASE = "c142691357ccaa9447a04065ee10bb4fce5697ea"
PARENT_RUN = 45017032913
CAMPAIGN = "stin-seven-baselines-fullscale-20261122"
BRANCH = "experiment/" + CAMPAIGN
WORKFLOW = ".github/workflows/stin_seven_baselines_fullscale_20261122.yml"
RULES, SEEDS = prior.RULES, prior.SEEDS
LEARNED = ("IPPO", *prior.LEARNED)
CELLS = tuple((m, s) for m in LEARNED for s in SEEDS) + tuple((m, 0) for m in RULES)
POPS = (20, 40, 60, 80, 100, 125, 150)
PREVIEW_TRACES, PREVIEW_WARM, PREVIEW_MEASURE = prior.TRACES, prior.WARM, prior.MEASURE
FULL_TRACES = ((2630101, 0.), (2630109, 0.))
FULL_WARM, FULL_MEASURE = 5., 300.
TRAIN_USERS, TRAIN_ADMIT, TRAIN_SLOTS = prior.TRAIN_USERS, prior.TRAIN_ADMIT, prior.TRAIN_SLOTS
BLOCK, RATIOS = prior.BLOCK, prior.RATIOS
PREVIEW_WALL, FULL_WALL = 1200, 7200
EVAL_CPUS = 4
PART_ENDS = (2, 5)
INHERITED = (*prior.INHERITED, "public_baseline_protocol.py",
             "public_baseline_updates.py", "public_baseline_worker.py")
cell_id = prior.cell_id
digest, packet_body = prior.digest, prior.packet_body


def settings(method, n):
    assert n in POPS
    if n != 125:
        return prior.settings(method, n)
    # The prior protocol whitelists multiples of ten. Extend its population partition only.
    counts = [n // 3 + int(i < n % 3) for i in range(3)]
    return dict(prior.settings(method, 20), users_per_region=max(counts), region_user_counts=counts)


def receipt_name(index, tier, trace_index=None, part=None):
    method, seed = CELLS[index]
    name = cell_id(method, seed)
    if tier == "preview" and trace_index is None and part is None:
        return name + ".preview"
    assert tier == "full" and trace_index in range(len(FULL_TRACES))
    return f"{name}.trace{trace_index}.full" + (f".part{part}" if part is not None else "")


def read_packet(release, commit):
    assert release["tag_name"] == CAMPAIGN
    return prior.read_packet(dict(release, tag_name=prior.CAMPAIGN), commit)


def plan(project=None, overlays=()):
    roots = tuple(map(Path, overlays)) + (Path(project) if project else Path(__file__).resolve().parent,)

    def fingerprint(name):
        p = next((root / name for root in roots if (root / name).is_file()), None)
        assert p is not None, name
        return sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    return dict(schema="seven-baselines-fullscale-v1", parent=BASE, anchor=PARENT_RUN,
        cells=CELLS, seed_selection="fixed before dispatch",
        train=dict(users=TRAIN_USERS, admit=TRAIN_ADMIT, steps=TRAIN_SLOTS, block=BLOCK,
                   phase=55., trace_offset=4300000, from_random=True,
                   note="one fit per method and seed; both evaluation tiers reuse the same parameters in the same VM"),
        preview=dict(users=POPS, traces=PREVIEW_TRACES, warmup_s=PREVIEW_WARM,
                     measured_s=PREVIEW_MEASURE, wall_s=PREVIEW_WALL, jobs=len(CELLS)),
        full=dict(users=POPS, traces=FULL_TRACES, warmup_s=FULL_WARM,
                  measured_s=FULL_MEASURE, wall_s=FULL_WALL, checkpoint_after_n=PART_ENDS,
                  jobs=len(CELLS), processes_per_vm=EVAL_CPUS),
        deterministic_eval=True, same_physics=True, original_deadlines=True,
        strict_qos_report_only=.95, gpu=False, paid=False, rerun=False,
        inherited_review_hashes={p: fingerprint(p) for p in INHERITED})

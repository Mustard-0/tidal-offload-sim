"""Pair native policy execution with the previously verified peak/trough study."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import paper_load_protocol as parent


BASE = 'd2f163df3548952bff7359eebf85ee9e4e4c2dca'
PARENT_RUN = 44940773550
WORKFLOW = '.github/workflows/tt_mappo_native_policies_20261122.yml'
BRANCHES = {name: f'experiment/tt-mappo-native-policies-{name}-20261122' for name in ('mid', 'high')}
INHERITED = parent.INHERITED + ('paper_load_protocol.py', 'paper_load_worker.py')


def stage(name):
    old = parent.stage(name)
    branch = BRANCHES[name]
    campaign = branch.split('/', 1)[1]

    def plan(project=None, overlay=None):
        root = Path(project) if project else Path(__file__).resolve().parent

        def fingerprint(path):
            f = (Path(overlay) if overlay else root) / path
            data = (f if f.is_file() else root / path).read_bytes()
            return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()

        return dict(schema='native-policy-paired-v1', stage=name, parent=BASE, anchor=PARENT_RUN,
            cases=old.CASES, arms=old.ARMS, seed=old.SEED, mix=old.MIXES, warmup=old.WARM,
            window=old.MEASURE, ratios=old.RATIOS, wall_s=old.WALL, train=False, gpu=False, paid=False,
            controller='Native checkpoint inference; no evaluation-only logits, heavy-task fallback, '
                       'route repair, special local governor, or queue reordering.',
            greedy='Shared weighted latency/energy/miss quote; full precision after report. '
                   'Before release, public location prior and actual report cost and slot delay.',
            diagnostics='Executed decisions and number of feasible actions; '
                        'task outcome and processing stage, 1-ms completed latency bins, energy and QoS.',
            selection='Same cases, inputs and model hashes as the previous study. No outcome-based exclusions.',
            inherited_review_hashes={p: fingerprint(p) for p in INHERITED})

    def read_packet(release, commit):
        assert release['tag_name'] == campaign
        return old.read_packet(dict(release, tag_name=old.CAMPAIGN), commit)

    return SimpleNamespace(**{**vars(old), 'CAMPAIGN': campaign, 'BRANCH': branch, 'BASE': BASE,
        'WORKFLOW': WORKFLOW, 'INHERITED': INHERITED, 'plan': plan, 'read_packet': read_packet})

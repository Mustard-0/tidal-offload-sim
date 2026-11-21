"""Frozen peak/trough evaluation of the eight-method rescue-v2 snapshot."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import rescue_main_protocol as prior


BASE = 'be0efb445d727ca5b763bcaa3a5c26a071cfc38e'
PARENT_RUN = 44938773068
WORKFLOW = '.github/workflows/tt_mappo_paper_load_20261122.yml'
BRANCHES = {k: f'experiment/tt-mappo-paper-load-{k}-20261122' for k in ('mid', 'high')}
# Start 11 s before the 75 s and 225 s aggregate extrema (5 s warmup + 6 s midpoint).
TRACES = ((1212221, 64.), (1212229, 214.))
MEASURE = 12.
WALL = 3600
INHERITED = prior.INHERITED + ('deadline_rescue.py', 'rescue_main_protocol.py', 'rescue_main_worker.py')


def stage(name):
    old = prior.stage(name)
    branch = BRANCHES[name]
    campaign = branch.split('/', 1)[1]
    cases = tuple(('original', n, t, phase) for n in old.POPS for t, phase in TRACES)

    def plan(project=None, overlay=None):
        root = Path(project) if project else Path(__file__).resolve().parent
        def fingerprint(p):
            f = Path(overlay) / p if overlay else root / p
            return hashlib.sha256((f if f.is_file() else root / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        return dict(schema='paper-load-peak-trough-v1', stage=name, parent=BASE, anchor=PARENT_RUN,
            cases=cases, arms=old.ARMS, seed=old.SEED, mix=old.MIXES, warmup=old.WARM,
            window=MEASURE, ratios=old.RATIOS, wall_s=WALL, train=False, gpu=False, paid=False,
            policy='Frozen rescue-v2 including dispatch and Local-Only; same fixed capacity and original deadlines.',
            report='Completed-only latency histogram at 1 ms upper bin edges, plus failures and original/95% QoS.',
            selection='Fixed traces and all 14 populations; evaluate peak and trough separately and retain every case.',
            inherited_review_hashes={p: fingerprint(p) for p in INHERITED})

    def read_packet(release, commit):
        assert release['tag_name'] == campaign
        return old.read_packet(dict(release, tag_name=old.CAMPAIGN), commit)

    return SimpleNamespace(**{**vars(old), 'CAMPAIGN': campaign, 'BRANCH': branch, 'BASE': BASE,
        'WORKFLOW': WORKFLOW, 'INHERITED': INHERITED, 'TRACES': TRACES, 'CASES': cases,
        'MEASURE': MEASURE, 'WALL': WALL, 'plan': plan, 'read_packet': read_packet})

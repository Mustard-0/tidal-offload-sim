"""Frozen, finite development screen and targeted accounting replay."""

import hashlib
from pathlib import Path
from statistics import mean

from review_protocol import digest
from scalability_protocol import CELLS, RULES, TRAIN_CELLS, TRACES


CAMPAIGN = 'energy-qos-greedy-audit-20261121'
BRANCH = 'experiment/' + CAMPAIGN
WORKFLOW = '.github/workflows/energy_qos_greedy_20261121.yml'
BASE_COMMIT = 'd7491d5be11d7737c4a4320bae51877189b7b36c'
PARENT_RUN = 44530737458
PARENT_CAMPAIGN = 'tt-mappo-matched-scale-up-20261119-gatefix'
PARENT_RELEASE = 451362024
PARENT_INPUTS = 'cd3735c055ed3f6cd97c2557d53d2ffa87faaa4d81305ff3f3bb7e28176b9ec4'
PARENT_PLAN = 'a2500a58e993f9051f29b2863ff2dfbb0526be8ae4cd747809b4b931cb0a1ced'
GREEDY = ('Greedy-NoPrivacy', 'Greedy-GeoI')
OPT = '_capacity'
DEV_CELLS = CELLS + tuple(g + OPT for g in GREEDY)
DEV_POP = (20, 40, 125, 150)
DEV_TRACES = (1202111, 1202119)
DEV_PHASES = (30., 150.)
DEV_KEYS = {(n, t, p) for n in DEV_POP for t in DEV_TRACES for p in DEV_PHASES}
REPLAY_CELLS = tuple(c for c in TRAIN_CELLS if c.split('_seed')[0] in
                     ('PPO', 'MARATD3-Adapted', 'SD-SAC-Adapted'))
REPLAY_POP = (125, 150)
REPLAY_SHARDS = tuple((cell, trace) for cell in REPLAY_CELLS for trace in TRACES)
INHERITED = ('review_environment.py', 'review_checks.py', 'review_protocol.py', 'pilot_protocol.py',
             'pilot_worker.py', 'optimization2_protocol.py', 'optimization2_core.py',
             'optimization2_worker.py', 'scalability_protocol.py', 'scalability_worker.py',
             'scalability_protocol_v2.py')
WARMUP, DEV_MEASURE, REPLAY_MEASURE = 5., 12., 300.
DEV_WALL, REPLAY_WALL, JOBS = 3500, 5700, 2 + len(DEV_CELLS) + len(REPLAY_SHARDS)
SCENES = len(DEV_CELLS) * len(DEV_KEYS) + len(REPLAY_SHARDS) * len(REPLAY_POP)


def plan(root=None):
    root = Path(__file__).resolve().parent if root is None else Path(root)
    return dict(schema='energy-qos-greedy-audit-v1', source_base=BASE_COMMIT,
        inherited_review_hashes={n: hashlib.sha256((root/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                 for n in INHERITED},
        parent=dict(run=PARENT_RUN, release=PARENT_RELEASE, commit=BASE_COMMIT,
                    inputs=PARENT_INPUTS, plan_hash=PARENT_PLAN),
        development=dict(cells=DEV_CELLS, traces=DEV_TRACES, phases_s=DEV_PHASES,
                         populations=DEV_POP, warmup_s=WARMUP, measure_s=DEV_MEASURE),
        replay=dict(shards=REPLAY_SHARDS, populations=REPLAY_POP, warmup_s=WARMUP,
                    measure_s=REPLAY_MEASURE, phase_s=0., selection_uses_replay=False),
        new_qos_ratio=.95, original_qos_unchanged=True, keep_all_seeds=True,
        choice=dict(min_average_cost_gain=.01, max_completion_loss=.002,
                    max_qos_increase_per_population=.005, same_variant_for_both_reporters=True),
        budget=dict(attempt=1, max_jobs=JOBS, max_scenes=SCENES, max_parallel=6,
                    stage_admission_hours=24, development_wall_s=DEV_WALL, replay_wall_s=REPLAY_WALL,
                    train=False, gpu=False, paid=False))


def _rates(rows):
    n = sum(r['arrivals'] for r in rows)
    return {k: sum(r[k] * r['arrivals'] for r in rows) / n
            for k in ('mean_task_cost', 'completion_rate', 'qos_rate', 'qos_95_rate')}


def choose(cells):
    needed = {g + marker for g in GREEDY for marker in ('', OPT)}
    missing = sorted(c for c in needed if not cells.get(c, {}).get('completed'))
    if missing:
        return dict(complete=False, selected=None, missing=missing)
    signatures = {}
    for name in sorted(needed):
        rows = cells[name]['rows']
        assert len(rows) == len(DEV_KEYS)
        assert {(r['users'], r['trace_seed'], r['phase_s']) for r in rows} == DEV_KEYS
        for row in rows:
            key = row['users'], row['trace_seed'], row['phase_s']
            sig = row['full_input_hash'], row['arrivals']
            assert signatures.setdefault(key, sig) == sig
            assert row['number_queued_pending'] == 0
    checks = []
    for g in GREEDY:
        base, alt = cells[g]['rows'], cells[g+OPT]['rows']
        b, a = _rates(base), _rates(alt)
        gain = (b['mean_task_cost'] - a['mean_task_cost']) / b['mean_task_cost']
        worst = max(_rates([r for r in alt if r['users'] == n])['qos_rate'] -
                    _rates([r for r in base if r['users'] == n])['qos_rate'] for n in DEV_POP)
        passed = gain >= 0 and a['completion_rate'] >= b['completion_rate']-.002 and worst <= .005
        checks.append(dict(reporter=g, base=b, candidate=a, gain=gain,
                           worst_population_qos_delta=worst, passes_non_regression=passed))
    avg = mean(c['gain'] for c in checks)
    selected = 'capacity' if avg >= .01 and all(c['passes_non_regression'] for c in checks) else 'control'
    return dict(complete=True, selected=selected, checks=checks, mean_gain=avg,
                paired_inputs=len(signatures), replay_used=False)

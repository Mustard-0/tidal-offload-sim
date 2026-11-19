"""Pre-register the three complementary arms of the full-tide scale-up."""
import hashlib
from pathlib import Path

from review_protocol import digest
from scalability_protocol import (CELLS, DEVELOPMENT_SEEDS, EVAL_CELLS, LEARNERS,
    MEASURE, POPULATIONS, PRIOR, RULES, TOTAL_STEPS, TRAIN_CELLS, TRACES, WARMUP,
    settings as matched_settings, split_cell)


CAMPAIGN = 'eight-method-full-tide-pressure-20261119'
BRANCH = 'experiment/' + CAMPAIGN
BASE_COMMIT = 'd7491d5be11d7737c4a4320bae51877189b7b36c'
PARENT_RUN = 44530737458
PARENT_CAMPAIGN = 'tt-mappo-matched-scale-up-20261119-gatefix'
PARENT_RELEASE = 451362024
PARENT_INPUTS = 'cd3735c055ed3f6cd97c2557d53d2ffa87faaa4d81305ff3f3bb7e28176b9ec4'
PARENT_PLAN = 'a2500a58e993f9051f29b2863ff2dfbb0526be8ae4cd747809b4b931cb0a1ced'
FACTORS = {'reference': (1., 1.), 'arrivals_x1.25': (1.25, 1.),
    'remote_cpu_x0.65': (1., .65), 'combined': (1.25, .65)}
SHARDS = tuple((scenario, cell, trace) for scenario in tuple(FACTORS)[1:] for cell, trace in EVAL_CELLS)
INHERITED = ('review_environment.py', 'review_checks.py', 'review_protocol.py', 'pilot_protocol.py',
    'pilot_worker.py', 'optimization2_protocol.py', 'optimization2_core.py', 'optimization2_worker.py',
    'scalability_protocol.py', 'scalability_worker.py', 'scalability_protocol_v2.py')
EVAL_WALL = 10100
PARENT_WAIT_WALL = 7200


def settings(scenario, method, seed, n):
    cfg, train = matched_settings(method, seed, n)
    a, cpu = FACTORS[scenario]
    cfg['arrival_per_user_s'] *= a
    cfg['cpu_bs'] *= cpu
    cfg['cpu_sat'] *= cpu
    return cfg, train


def plan():
    root = Path(__file__).resolve().parent
    return dict(schema='matched-full-tide-pressure-arms-v1', source_base=BASE_COMMIT,
        inherited_review_hashes={n: hashlib.sha256((root/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in INHERITED},
        parent=dict(run=PARENT_RUN, release=PARENT_RELEASE, commit=BASE_COMMIT, plan_hash=PARENT_PLAN,
            inputs_sha256=PARENT_INPUTS, shared_terminal_weights=True, reference_recomputed=False),
        cohorts=dict(seeds=DEVELOPMENT_SEEDS, traces=TRACES, populations=POPULATIONS,
            warmup_s=WARMUP, measure_s=MEASURE, phase_s=0., natural_drain=True),
        factors=FACTORS, shards=SHARDS, scene_count=len(SHARDS)*len(POPULATIONS),
        eval_wall_s=EVAL_WALL, parent_wait_wall_s=PARENT_WAIT_WALL, maximum_concurrent=6, stage_admission_hours=24,
        no_retraining=True, no_test_driven_selection=True, comparison='full-window robustness of fixed policies',
        controls='Hold deadlines, privacy, rewards, channels, UE capacity, and all policy settings fixed. '
            'Only shared arrival intensity and remote CPU change. No ranks, trends or seed rejection gates.')

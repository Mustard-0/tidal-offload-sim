"""One matched 302-update scale-up study on previously reserved inputs."""
import hashlib
from pathlib import Path

from optimization2_protocol import (ADMISSION_STEPS, BLOCK, CAMPAIGN as PRIOR_CAMPAIGN,
    DEVELOPMENT_SEEDS, RESERVED_TEST_TRACES as TRACES, TOTAL_STEPS, settings as tt_settings)
from pilot_protocol import settings as prior_settings
from review_protocol import LEARNERS, POPULATIONS, RULES, digest


CAMPAIGN = 'tt-mappo-matched-scale-up-20261119'
BRANCH = 'experiment/' + CAMPAIGN
BASE_COMMIT = '6b6a55f072bc36d3e7ab4dfa9b34ff96c12389e4'
PRIOR_RUN = '44466767988'
PRIOR_INPUTS = 'b6586b15a5405f8a83d4c02b094327ac1c3d3d2253bda799be2cd564384ffc22'
PRIOR_PLAN = '4c2fffd3960371b493fa7b68987634c4c42c9c63cf69ecbf5ef3758186b633d7'
PRIOR = {
    '871901': (514263, 'fda47887ff24434290cb1887d9ae4414515b9902b25c3b0d6550ff783ce9afa7',
        '2ebe32c3e0131ac2a0834938828d296a481b99ed4e599275c366eb8ff59a488f'),
    '871909': (513687, '35b314b91cb48f2ea09a824d50213bf6e2b25569aeec6b33eaa560707094770b',
        'a1dbe6e987ec1fc20840602ba483eb92c3d4770a65625fe5ee0daf8ea27d82c8'),
}
LEARNED_CELLS = tuple(f'{m}_seed{s}' for m in LEARNERS for s in DEVELOPMENT_SEEDS)
TRAIN_CELLS = tuple(c for c in LEARNED_CELLS if not c.startswith('TMAPO_'))
CELLS = LEARNED_CELLS + RULES
EVAL_CELLS = tuple((c, trace) for c in CELLS for trace in TRACES)
INHERITED = ('review_environment.py', 'review_protocol.py', 'review_checks.py', 'pilot_protocol.py',
    'pilot_worker.py', 'optimization2_protocol.py', 'optimization2_worker.py', 'optimization2_core.py')
WARMUP, MEASURE = 5., 300.
TRAIN_WALL, EVAL_WALL = 8500, 10100


def split_cell(cell):
    if cell not in CELLS:
        raise ValueError('Unregistered cell')
    if cell in RULES:
        return cell, None
    method, seed = cell.rsplit('_seed', 1)
    return method, int(seed)


def settings(method, seed, n):
    if method == 'TMAPO':
        # The same TT-MAPPO architecture and privacy pipeline at every population.
        if n in (20, 80, 150):
            return tt_settings('full35', seed, n)
        _, _, cfg, train = prior_settings(method, seed, n)
        _, fixed = tt_settings('full35', seed, 150)
        train.update(fixed)
        return cfg, train
    _, _, cfg, train = prior_settings(method, seed or DEVELOPMENT_SEEDS[0], n)
    train.update(updates=TOTAL_STEPS // BLOCK, rollout_steps=BLOCK, episodes=1,
        learning_rate_schedule_horizon=1208)
    return cfg, train


def plan():
    root = Path(__file__).resolve().parent
    return dict(schema='matched-scale-up-v1', source_base=BASE_COMMIT,
        inherited_code={name: hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in INHERITED},
        tt_reference=dict(run=PRIOR_RUN, campaign=PRIOR_CAMPAIGN, inputs=PRIOR_INPUTS,
            plan_hash=PRIOR_PLAN, verified_checkpoints=PRIOR, variant='full35', retrained=False),
        train_cells=TRAIN_CELLS, cells=CELLS, shards=EVAL_CELLS, populations=POPULATIONS,
        training_seeds=DEVELOPMENT_SEEDS, training_population=150, rate_per_user=9.,
        training=dict(admission_slots=ADMISSION_STEPS, steps=TOTAL_STEPS, block_slots=BLOCK,
            updates=TOTAL_STEPS // BLOCK, offpolicy_steps_per_block=24, baseline_credit=0.,
            schedule_horizon=1208, training_trace_offset=1191900),
        test=dict(traces=TRACES, warmup_s=WARMUP, measure_s=MEASURE, phase_s=0., natural_drain=True,
            evaluation_policy='fixed deterministic; no training in evaluation'),
        budget=dict(train_wall_s=TRAIN_WALL, eval_wall_s=EVAL_WALL, attempts=1,
            maximum_cloud_jobs=1 + len(TRAIN_CELLS) + len(EVAL_CELLS), gpu=False, paid=False),
        limits='Matched interactions and inputs, not equal tuning or optimizer compute. '
            'Independent GeoI baselines versus TT-MAPPO privacy pipeline: system comparison. '
            'Recent baselines are architecture adaptations. No automatic expansions or retries.')

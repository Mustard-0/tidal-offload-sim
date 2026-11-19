"""Frozen full-tide TMAPO development screen, no ranking-driven input search."""
import hashlib
from pathlib import Path
from pilot_protocol import settings as prior_settings
from review_protocol import DEVELOPMENT_SEEDS, digest

CAMPAIGN = 'tmapo-full-tide-development-20261119'
BRANCH = 'experiment/' + CAMPAIGN
BASE_COMMIT = 'b28d915c157cb3606051063f887e1b962a2ec458'
VARIANTS = {'full70': (True, .7), 'nopred70': (False, .7),
            'full35': (True, .35), 'nopred35': (False, .35)}
CELLS = tuple(v + '_seed' + str(s) for v in VARIANTS for s in DEVELOPMENT_SEEDS)
VALIDATION_TRACES = (1193901, 1193909)
RESERVED_TEST_TRACES = (1194901, 1194909)
VALIDATION_POPULATIONS = (80, 150)
VALIDATION_PHASES = (30., 150.)
CHECKPOINT_UPDATES = (0, 151, 302)
TOTAL_STEPS = 30200
ADMISSION_STEPS = 30000
BLOCK = 100


def settings(variant, seed, n=150):
    if variant not in VARIANTS or seed not in DEVELOPMENT_SEEDS or n not in VALIDATION_POPULATIONS + (20,):
        raise ValueError('Outside finite development configuration')
    _, _, cfg, train = prior_settings('TMAPO', seed, n)
    train.update(local_credit_fraction=VARIANTS[variant][1], updates=302, rollout_steps=100,
                 episodes=1, learning_rate_schedule_horizon=1208)
    return cfg, train


def plan():
    root = Path(__file__).resolve().parent
    return dict(schema='full-tide-tmapo-development-v1', source_base=BASE_COMMIT,
        variants=VARIANTS, cells=CELLS, training_seeds=DEVELOPMENT_SEEDS,
        train_population=150, region_counts=(50, 50, 50), arrival_per_user_s=9.,
        admission_slots=ADMISSION_STEPS, fixed_drain_slots=200,
        total_env_steps=TOTAL_STEPS, updates=302, rollout_slots=BLOCK,
        schedule_horizon=1208, from_scratch=True, old_weights_reused=False,
        validation=dict(traces=VALIDATION_TRACES, populations=VALIDATION_POPULATIONS,
            phases=VALIDATION_PHASES, checkpoints=CHECKPOINT_UPDATES,
            warmup_s=5., measure_s=10., natural_drain=True),
        reserved_test_traces=RESERVED_TEST_TRACES, final_test_used=False,
        inherited_review_hashes={n: hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for n in ('review_protocol.py','review_environment.py','review_checks.py',
                      'pilot_protocol.py','pilot_worker.py')},
        selection=dict(reference='full70', minimum_pooled_cost_improvement=.02,
            minimum_each_seed_cost_improvement=0., maximum_qos_increase=.005,
            maximum_completion_decrease=.005, no_energy_increase=True,
            no_completed_delay_increase=True, maximum_late_cost_ratio=1.05,
            all_eight_cells_required=True, uses_validation_only=True),
        controls='Tasks/deadlines/physical resources/reward/privacy/wait cap are frozen; only forecast-feature visibility and credit fraction vary',
        budget=dict(maximum_parallel=8, maximum_parts=2, seconds_per_part=9000,
            maximum_campaign_hours=24, free_public_cpu_only=True, paid=False, gpu=False),
        resume='Only a normally saved paused state; restore environment, optimizer, normalizer and all RNG state after source/runtime identity checks. Completed part1 -> part2 is a no-new-training reference.',
        stop='One fixed 8-cell screen. No failure rerun, input replacement, seed dropping, arbitrary third part, or automatic formal/main-baseline expansion.',
        scope='TT-MAPPO optimization screen only. Do not mix these longer-trained models with the previous short-trained baselines to claim superiority.')

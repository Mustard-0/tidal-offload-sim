import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE_COMMIT='d0e67b6199915a0daf2773256ca55d9ef892196e'
METHODS=('TMAPO','IPPO','PPO','SD-SAC-Adapted','MARATD3-Adapted')
SEEDS=(87021,87022)
VALIDATION_SEEDS=(9791101,9791109)
TEST_SEEDS=(9792101,9792109)
VARIANTS={
    'control':dict(actor_lr=1e-4,critic_lr=3e-4,gae_lambda=.98,entropy=.01),
    'lr':dict(actor_lr=3e-4,critic_lr=1e-3,gae_lambda=.98,entropy=.01),
    'horizon':dict(actor_lr=1e-4,critic_lr=3e-4,gae_lambda=.995,entropy=.001),
    'lr_horizon':dict(actor_lr=3e-4,critic_lr=1e-3,gae_lambda=.995,entropy=.001),
}
SCREEN=[dict(id=f'TMAPO_{v}_seed{s}',method='TMAPO',seed=s,variant=v)
    for v in VARIANTS for s in SEEDS]+[dict(id=f'{m}_control_seed87021',method=m,seed=87021,variant='control') for m in METHODS[1:]]
FORMAL=[dict(id=f'{m}_seed{s}',method=m,seed=s) for m in METHODS for s in SEEDS]


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configuration(method,seed=87021,n=150,variant='control',screen=False):
    from pressure9.protocol import configuration as base
    c,t=base(method,seed,n)
    if method=='TMAPO':t.update(VARIANTS[variant])
    if screen:t.update(updates=302,episodes=1,validation_updates=(151,302))
    t['hyperparameter_variant']=variant if method=='TMAPO' else 'control'
    return c,t


def plan():
    return dict(schema='finite-night-validation-review-v1',base_commit=BASE_COMMIT,
        original_frozen_run='40715374452',old_queue_not_modified=True,
        manuscript_status='Located historical v48 draft differs from latest human-authorized bounded-privacy setup; do not label this a v48 reproduction or silently restore its K-anonymity/sliding-budget model',
        classification='Versioned finite learning screen and conditional full-budget recheck under latest approved environment, not a final multi-seed paper conclusion',
        hypotheses=['Actor/critic LR may under-update within finite budget',
          'Longer GAE horizon and lower entropy may improve delayed physical credit; bundled hypothesis, not isolated causal attribution'],
        variants=VARIANTS,screen=SCREEN,formal=FORMAL,cells=FORMAL,
        screen_updates=302,formal_updates=1208,rollout_slots=100,
        screen_interactions_per_cell=30200,formal_interactions_per_cell=120800,
        maximum_screen_cells=12,maximum_formal_cells=10,maximum_new_interactions=1570400,
        gate_component_interactions_additional_maximum=6000,
        maximum_rounds=1,maximum_campaign_hours=24,
        shared_environment='UNCHANGED N150=50/50/50 rate9, tasks/resources/deadlines/reward/privacy/wait20ms; all methods same',
        tmapo_modules='UNCHANGED .7 local physical credit/CTDE/causal prediction/conditional wait, not copied into baselines',
        learning_schedule='First302 updates of declared1208-update LR schedule in screens; formal starts from scratch, no weights reuse',
        validation=dict(seeds=VALIDATION_SEEDS,population=150,phases=(30.,150.),warmup_s=5.,measure_s=10.,checkpoints=(0,151,302)),
        test=dict(seeds=TEST_SEEDS,populations=(20,40,60,80,100,125,150),warmup_s=5.,measure_s=300.,natural_drain=True,
          policy='All categorical argmax, native deterministic MAR; fixed before any new scores'),
        selection=dict(target='TMAPO two-seed mean terminal validation physical cost only; never final-test ranking',
          improvement_vs_control_minimum=.02,improvement_vs_initial_each_seed_minimum=.10,
          qos_increase_vs_control_maximum=.005,late_validation_relative_cost_increase_maximum=.05,
          require_all_12_screens_complete=True,no_retry=True),
        formal_condition='Only if selection gate passes; all5methods x2seeds same120800 interactions, fresh initialization and test seeds',
        fairness='Extra8 TM tuning cells versus1 diagnostic per other method disclosed; not equal hyperparameter search or final significance',
        budget=dict(maximum_parallel_screen=6,maximum_parallel_formal=8,free_public_cpu_only=True,
          seconds_per_screen=5400,maximum_formal_training_seconds=18000,evaluation_seconds=10800),
        stop='No blind rerun for failure/numerical/unknown; no gate pass -> stop after evidence; no improvement-ranking-driven third phase')


def source_hashes():
    files=[]
    for folder in ('model','experiments/lr_screen','study','pressure9','nightstudy'):
        files+=list((ROOT/folder).rglob('*.py'))
    files+=[ROOT/'model/config.json']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(files)}

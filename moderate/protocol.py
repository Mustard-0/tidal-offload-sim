"""Freeze one moderate shared environment BEFORE new training/test outcomes."""
import hashlib
import json
from . import ROOT

METHODS=('TMAPO','IPPO','PPO','SD-SAC-Adapted','MARATD3-Adapted')
SEEDS=(87021,87022)
POPULATIONS=(20,40,60,80,100,125,150)
VALIDATION_SEEDS=(8298101,8298109)
TEST_SEEDS=(9298101,9298109)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configuration(method, seed=87021, n=150):
    if method not in (*METHODS,'Greedy','GeoI-Greedy') or seed not in SEEDS or n not in POPULATIONS:
        raise ValueError('Outside finite moderate protocol')
    from study.protocol import configuration as original
    c,t=original(method if method in ('TMAPO','IPPO','PPO') else 'TMAPO',seed,n)
    for unused in ('arrival_steps','episode_max_steps','checkpoint_episodes'):
        t.pop(unused,None)
    c.update(arrival_per_user_s=6.,observation_horizon_slots=30000)
    t.update(episodes=4,updates=1208,cycle_steps=30200,admission_steps=30000,rollout_steps=100,
        gamma=.999,local_credit_fraction=.7 if method=='TMAPO' else 0.,
        actor_cost_prior_scale=0.,checkpoint_updates=10,checkpoint_seconds=120.,
        validation_updates=(302,604,906,1208),actor_lr=1e-4,critic_lr=3e-4,
        torch_threads=1,offpolicy_updates_per_block=6,history=8)
    if method=='SD-SAC-Adapted':
        t.update(actor_lr=3e-4,critic_lr=3e-4,batch=64,replay_capacity=20000)
    elif method=='MARATD3-Adapted':
        t.update(actor_lr=1e-4,critic_lr=1e-4,batch=8,replay_capacity=1024)
    return c,t


def plan():
    return dict(schema='moderate-six-arrivals-full-cycle-v1',
        base_commit='0d1553cb789c399f3186cb1327d143684060032e',
        environment_selection='Common arrival rate6 from completed rule-only pressure discovery; moderate congestion, not selective weakening of Greedy',
        development_run='40638214908',development_traces_excluded=[7298101,7298109],
        cells=[dict(id=f'{m}_seed{s}',method=m,seed=s) for m in METHODS for s in SEEDS],
        maximum_training_cells=10,learning_rate_scan=False,seed_selection=False,
        actual_training_population=[50,50,50],arrival_per_user_s=6.,
        training=dict(cycles=4,admission_slots_per_cycle=30000,fixed_drain_slots=200,
            environment_steps=120800,rollout_slots=100,updates=1208,
            no_reset_between_updates=True,phase_s=0.,
            plot='100-slot rollout return vs env_steps; MA50 UPDATES, never mislabeled episodes'),
        validation=dict(seeds=VALIDATION_SEEDS,population=150,phases=(30.,150.),warmup_s=5.,measure_s=10.,
            checkpoints=(0,302,604,906,1208),selection='terminal only; no test tuning or best-seed/checkpoint selection'),
        test=dict(seeds=TEST_SEEDS,populations=POPULATIONS,phase_s=0.,warmup_s=5.,measure_s=300.,natural_drain=True,
            mode='sampled categorical as trained; native deterministic TD3 without exploration, one declared mode per family'),
        reward='UNCHANGED .25*resolution_seconds/.5 + .25*system_energy_j/.24 + QoS failure; QL diagnostic',
        credit='TMAPO .7 mean-conserving local physical credit, others scalar team reward; same reported physical objective, NOT identical per-agent training vectors',
        algorithms=dict(TMAPO='local actors + centralized local/global critic, causal forecast, conditional <=20ms wait',
            IPPO='local actors and local critic; no forecast or cost logit prior',
            PPO='central joint-action likelihood and global critic; no forecast or cost logit prior',
            SAC='SD-SAC-Adapted central masked discrete SAC, twin Q, targets, event-chain replay, auto-temperature',
            TD3='MARATD3-Adapted 2-layer LSTM actor/critics, 2-head attention, twin Q, delayed updates and discrete intent decoding'),
        recent_baselines='Architecture adaptations, not author code or exact continuous resource-allocation reproductions',
        removed_from_all_learners='analytic actor cost prior and objective-weighted user action hints; no TM forecast supplied to baselines',
        privacy='Unchanged bounded tight R500/epsilon1/delta1e-6/shrink.5, actual public-region/class distinct users; GeoI rule is explicitly different mechanism',
        waiting='Baselines release immediately; TMAPO conditional physical-gain gate <=20ms; zero waits may be correct, never force waits',
        unchanged='task profiles/deadlines/lifetime/CPU/bandwidth/queues/energy coefficients/physical objective',
        hyperparameters='PPO-family actor1e-4 critic3e-4; SAC3e-4/3e-4; MAR1e-4/1e-4 with LSTM .1x; no rate selection. Hidden32, common slot gamma.999; existing six replay gradients/100 slots',
        budget=dict(maximum_parallel=12,maximum_training_segments=2,seconds_per_segment=9000,
            maximum_training_wall_s=18000,standard_public_cpu_only=True,paid=False,gpu=False,
            evaluation_seconds_per_cell=10800,no_failed_or_unknown_rerun=True,maximum_campaign_days=7),
        stop='Finite queue only; no ranking-triggered modifications or extra training; incomplete/unstable outcomes retained')


def source_hashes():
    files=list((ROOT/'model').glob('*.py'))+[ROOT/'model/config.json']
    files+=list((ROOT/'experiments/lr_screen').rglob('*.py'))+list((ROOT/'study').glob('*.py'))
    files+=list((ROOT/'moderate').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(files)}

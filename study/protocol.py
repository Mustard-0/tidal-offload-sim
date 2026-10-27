"""One LR, two fixed seeds, three genuine algorithms; no ranking-dependent edits."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=('TMAPO','IPPO','PPO')
SEEDS=(87021,87022)
POPULATIONS=(20,40,60,80,100,125,150)
VALIDATION_SEEDS=(6294101,6294109)
TEST_SEEDS=(6295101,6295109)
PHASES=(30.,90.,150.,210.,270.)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configuration(method,seed=87021,n=100):
    if method not in METHODS or seed not in SEEDS or n not in POPULATIONS:
        raise ValueError('Outside the finite declared study')
    base=json.loads((ROOT/'model/config.json').read_text())
    c,t=base['environment'],base['training']
    counts=[n//3+int(i<n%3) for i in range(3)]
    c.update(users_per_region=max(counts),region_user_counts=counts,
        region_radius_m=500.,epsilon_req=1.,delta_req=1e-6,report_shrink=.5,
        project_public_region=False,mechanism='bounded_tight_actual_batch_shrink0.5',
        weight_time=.25,weight_energy=.25,weight_quality=0.,weight_violation=1.,
        time_ref_s=.5,energy_ref_j=.24,quality_ref_m=500.,
        wait_max_s=.02,prediction_window_s=.1,prediction_horizon_s=.02,
        active_wait_mode='causal_physical_gain_guard_v1',observation_horizon_slots=100,
        min_service_reserve_s=.04,wait_min_history_s=.1,wait_history_window_s=.2,
        wait_fill_probability=.8,wait_model_samples=16,wait_model_margin=.002,
        wait_confidence_z=1.645,baseline_release='immediate')
    t.update(seed=seed,episodes=2000,updates=2000,arrival_steps=100,episode_max_steps=303,
        hidden=32,actor_lr=1e-4,critic_lr=3e-4,gamma=1.,gae_lambda=.98,
        epochs=2,minibatch=256,ppo_clip=.2,target_kl=.02,entropy=.01,
        value_normalization=True,local_credit_fraction=0.,max_grad_norm=.5,
        actor_cost_prior_scale=0.,linear_lr_decay=True,lr_final_fraction=.1,
        checkpoint_episodes=10,checkpoint_seconds=120.,torch_threads=1)
    return c,t


def plan():
    return dict(schema='N100-basic-and-causal-wait-20261028-v1',
        cells=[dict(id=f'{m}_seed{s}',method=m,seed=s) for m in METHODS for s in SEEDS],
        episodes_per_cell=2000,arrival_slots=100,natural_drain=True,maximum_episode_slots=303,
        actual_training_users=[34,33,33],evaluation_populations=POPULATIONS,
        actor_lr=1e-4,critic_lr=3e-4,lr_grid=False,seed_selection=False,
        reward='UNCHANGED N100 physical cost .25*seconds/.5 + .25*joules/.24 + QoS; QL diagnostic only',
        task_resource_change=False,privacy_change=False,
        algorithms=dict(TMAPO='local actors, V(local,public-global), causal forecast features, physically gated optional regional wait',
            IPPO='local actors, V(local), NO forecast features/analytic prior, immediate batch release',
            PPO='central joint-action controller and V(public-global), NO forecast features/analytic prior, immediate batch release'),
        common='same tasks, links, compute, deadline rules, bounded actual-batch privacy, feasible routes, optimizers and reward; no truth available to policy',
        removed_from_all='analytic actor logits prior and objective-weighted local/offload hint features; no TMAPO weight transfer',
        waiting='User WAIT masked. Regional wait<=20ms, only TMAPO, optional after historical distinct-fill probability and public-prior physical gain confidence checks. Baselines immediate, never forced blind waits',
        small_batch_limit='epsilon0 remains1 for n1..20 at current accountant; small populations may correctly yield ZERO active waits; no fabricated wait advantage',
        prediction='100ms observation windows replace old1s window that updated only at final admission slot; task tide period stays300s',
        initial_and_terminal_validation=dict(seeds=VALIDATION_SEEDS,phases=(30.,150.),population=100),
        test=dict(seeds=TEST_SEEDS,phases=PHASES,slots=100,
                  primary='sampled as trained, keyed policy RNG per scene',secondary='deterministic argmax for ALL methods'),
        selection='terminal checkpoint only, single common LR, retain both seeds; no final-test tuning or superiority guarantee',
        budget=dict(maximum_parallel=6,maximum_wall_seconds_per_cell=10800,maximum_segments_per_cell=1,
                    no_automatic_rerun=True,maximum_campaign_days=7),
        limitations='short cohorts at fixed phases are NOT full tidal cycles; conditional per-batch privacy is NOT trajectory DP; architecture comparison need not favor TMAPO')


def source_hashes():
    files=list((ROOT/'model').glob('*.py'))+[ROOT/'model/config.json']
    files+=list((ROOT/'experiments/lr_screen').rglob('*.py'))+list((ROOT/'study').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(files)}

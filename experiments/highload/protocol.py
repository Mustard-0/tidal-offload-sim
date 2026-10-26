"""Prospectively frozen finite high-load study, including matched ablations."""
import hashlib
import json
from pathlib import Path
from . import ROOT

LRS = {'TMAPO': ((3e-5,9e-5),(1e-4,3e-4),(2e-4,6e-4),(3e-4,9e-4),(5e-4,1.5e-3)),
       'IPPO': ((3e-5,9e-5),(1e-4,3e-4)), 'PPO': ((3e-5,9e-5),(1e-4,3e-4)),
       'SD-SAC-Adapted': ((1e-4,1e-4),(3e-4,3e-4)),
       'MARATD3-Adapted': ((1e-4,1e-4),(3e-4,3e-4))}
SEEDS = {'TMAPO': (87021,87022,87023), 'IPPO': (87021,87022), 'PPO': (87021,87022),
         'SD-SAC-Adapted': (87021,87022), 'MARATD3-Adapted': (87021,87022)}
ABLATIONS = ('TMAPO-NoForecast','TMAPO-NoPrior','TMAPO-LocalCritic')
POPULATIONS = (100,200,300,400)
CYCLE_POPULATIONS = (300,200,400,300,100,400,200,300,400,300)
VALIDATION_SEEDS = (814001,814019)
TEST_SEEDS = (9753101,9753109,9753133,9753167)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configuration(method, lr=1, seed=87021):
    family = 'TMAPO' if method in ABLATIONS else method
    if family not in LRS or not 1 <= lr <= len(LRS[family]) or seed not in SEEDS[family]:
        raise ValueError('Outside finite declared study')
    c = json.loads((ROOT/'model/config.json').read_text())['environment']
    c.update(users_per_region=100,region_user_counts=[100,100,100],
        region_radius_m=500.,epsilon_req=1.,delta_req=1e-6,report_shrink=.5,
        wait_max_s=.02,prediction_horizon_s=.02,project_public_region=False,
        mechanism='bounded_tight_actual_batch_shrink0.5',quality_ref_m=500.,
        weight_quality=0.,weight_time=1.,time_ref_s=.1,
        weight_energy=.1,energy_ref_j=.1,weight_violation=5.,
        active_wait_mode='immediate_until_physical_benefit_validated',
        min_service_reserve_s=.04,wait_fill_probability=.8,wait_min_history_s=1.,
        observation_horizon_slots=30000)
    a,v = LRS[family][lr-1]
    t = dict(seed=seed,hidden=32,actor_lr=a,critic_lr=v,gamma=.999,
        gae_lambda=.98,value_normalization=True,local_credit_fraction=0.,
        ppo_clip=.2,target_kl=.02,entropy=.01,epochs=2,minibatch=256,
        max_grad_norm=.5,actor_cost_prior_scale=0. if method=='TMAPO-NoPrior' else 2.,
        linear_lr_decay=True,lr_final_fraction=.1,torch_threads=1,
        updates=3000,rollout_steps=100,cycle_steps=30000,drain_slots=200,
        populations=list(CYCLE_POPULATIONS),checkpoint_updates=10,
        checkpoint_seconds=300.,validation_updates=[0,1000,2000,3000])
    return c,t


def cell(method, lr, seed):
    c,t = configuration(method,lr,seed)
    return dict(id=f'{method}_LR{lr}_seed{seed}',algorithm=method,lr_index=lr,seed=seed,
        actor_lr=t['actor_lr'],critic_lr=t['critic_lr'],env_steps=300000,
        phase='ablation' if method in ABLATIONS else 'screen',
        conditional_on_selected_TMAPO_lr=method in ABLATIONS)


def plan():
    return dict(schema='highload-causal-continuing-20261027-v1',
        cells=[cell(m,lr,s) for m in LRS for lr in range(1,len(LRS[m])+1) for s in SEEDS[m]],
        conditional_ablation_cells=[cell(m,lr,s) for m in ABLATIONS for lr in range(1,6) for s in SEEDS['TMAPO']],
        conditional_rule='Exactly one validation-selected TMAPO LR activates 3 ablations x3 seeds=9; other conditional cells NEVER run',
        maximum_training_cells=40,main_cells=31,ablation_cells=9,
        training_env_steps_per_cell=300000,training_updates=3000,rollout_steps=100,
        actual_full_cycles_per_cell=10,cycle_steps=30000,admission_steps_per_cycle=29800,
        cycle_populations=CYCLE_POPULATIONS,evaluation_populations=POPULATIONS,
        mechanism='same bounded actual-region/class distinct-user batch calibration, R500, epsilon1, delta1e-6, shrink0.5 for all except explicit GeoI reference',
        active_wait='ZERO: do not force batching or silently add delays',
        reward='-(delay_seconds/.1 + .1*energy_j/.1 +5*QoS_failure); QL diagnostic only; normalized identical scalar team slot cost',
        priority='100ms delay has cost1;100mJ energy has cost0.1; a violation costs5. Latency-first physical objective, not rank-dependent',
        execution='categorical learned policies sampled as trained; deterministic projection secondary only. TD3 uses its deterministic intent policy, no added exploration at evaluation. Greedy minimizes instantaneous reported-state cost, no future arrivals',
        initial_policy='evaluate before any update; common physical-cost prior held equal except NoPrior',
        validation=dict(seeds=VALIDATION_SEEDS,populations=(200,300,400),phases=(30.,150.),
            warmup_s=3.,measure_s=3.,updates=(0,1000,2000,3000),
            selection='LR minimizes terminal validation pooled physical cost across all declared seeds. No best seed or best intermediate checkpoint. Same rule for every method'),
        final_test=dict(seeds=TEST_SEEDS,populations=POPULATIONS,phase=0.,
            warmup_s=5.,measure_s=300.,checkpoint='terminal300000 interactions',
            held_out=True,repetitions='one keyed policy RNG per trace/training seed'),
        ablations=dict(no_forecast='Mask forecast/error/trend-derived policy inputs ONLY, retain true workload and queues',
            no_prior='Same networks/observations, analytic actor cost prior scale0',
            local_critic='Same actor and reward, critic(local) replaces critic(local,global). Same selected full-model LR/3 seeds',
            no_wait='NOT a separate curve: full method already immediate; avoid identical fake ablation'),
        budget=dict(maximum_parallel=16,maximum_segments=96,maximum_segment_seconds=10800,maximum_days=7),
        labels='100 steps = update/rollout chunk, NOT a fully drained episode. Preserve queues/history; terminal only at fixed fully drained300s cycle boundary',
        fairness='Same300000 total environment steps including drainage and same keyed arrival streams per seed/cycle. Validation and final test not used for learning',
        limitations='New training from scratch; not a continuation or repair of old N100 results. Architecture adaptations, not exact author-paper resource/topology reproduction; no superiority guarantee')


def source_hashes():
    files=list((ROOT/'model').glob('*.py'))+[ROOT/'model/config.json']
    files+=[p for p in (ROOT/'experiments').rglob('*.py')
            if p.name not in ('cloud_worker.py','evaluate_entry.py')]
    files+=list((ROOT/'experiments/lr_screen/vendor').glob('LICENSE*'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(set(files))}


def population(cfg,n):
    result=dict(cfg)
    counts=[n//3+int(g<n%3) for g in range(3)]
    result.update(users_per_region=max(counts),region_user_counts=counts)
    return result

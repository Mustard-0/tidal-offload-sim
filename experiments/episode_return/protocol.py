"""N100 split 34/33/33; 100 arrival slots followed by natural task draining."""
import hashlib
import json
from pathlib import Path
from experiments.lr_screen.protocol import configuration as old_configuration, digest, LRS

ROOT = Path(__file__).resolve().parents[2]
SEEDS = {'TMAPO': (87021,87022,87023), 'IPPO': (87021,87022), 'PPO': (87021,87022)}
LR_COUNTS = {'TMAPO': 5, 'IPPO': 2, 'PPO': 2}


def configuration(method, lr=1, seed=87021):
    if (method not in SEEDS or not 1<=lr<=LR_COUNTS[method]
            or seed not in SEEDS[method]):
        raise ValueError('Outside the predeclared 23-cell LR/seed grid')
    cfg, train = old_configuration(method, lr)
    cfg.update(users_per_region=34, region_user_counts=[34,33,33], observation_horizon_slots=100)
    train.update(episodes=2000, arrival_steps=100, episode_max_steps=303,
                 updates=2000, rollout_steps=100, seed=seed,
                 checkpoint_episodes=5, checkpoint_seconds=300.)
    train.pop('episode_steps',None)
    for key in ('validation_interval', 'validation_seeds', 'validation_phases',
                'validation_warmup_s', 'validation_measure_s', 'checkpoint_interval'):
        train.pop(key, None)
    return cfg, train


def plan():
    cfg, train = configuration('TMAPO')
    return dict(schema='N100-arrival100-drain-multiseed-20261026', environment=cfg, common_training=train,
        cells=[dict(id=f'{method}_N100_A100_LR{lr}_seed{seed}', algorithm=method, lr_index=lr,
                    seed=seed,actor_lr=LRS[method][lr-1][0],critic_lr=LRS[method][lr-1][1],
                    episodes=2000, arrival_steps=100,
                    maximum_env_steps=606000)
               for method in SEEDS for lr in range(1,LR_COUNTS[method]+1) for seed in SEEDS[method]],
        reward='sum of unchanged normalized team rewards over all arrival AND terminal drain slots',
        plot='x=completed episode (1..2000), y=full episode return including drainage',
        selection='Show every predeclared LR/seed; no best-seed selection, no ranking-driven edits. Training returns are not held-out validation.',
        restart='new N100 and 100-arrival-slot episode protocol; all three start from initialization',
        removed='periodic validation, validation task CSV, best-validation checkpoint, per-update CSV rewrites',
        unchanged='task sizes/deadlines, physical resources, privacy mechanism, RL action logic and physical-cost reward',
        fairness='identical episode arrival seeds and 100-slot admission windows; actual drain/interaction counts vary by policy and are reported, not claimed equal',
        scope='23 fixed exploratory LR/seed cells, no algorithm superiority guarantee',
        lr_rule='actor and critic scale together at a fixed critic/actor ratio of 3',
        seeds=SEEDS,lr_counts=LR_COUNTS,
        reuse='Three completed LR1/seed87021 cells retained only after exact config and unchanged model/environment/update-core verification; no old N300 results',
        total_arrival_env_steps=4600000, maximum_total_env_steps=13938000,
        population=[34,33,33], no_phantom_users=True,
        maximum_parallel=16, maximum_segments=96, maximum_days=7)


def source_hashes():
    files = list((ROOT/'model').glob('*.py')) + [ROOT/'model/config.json']
    files += list((ROOT/'experiments/lr_screen').rglob('*.py'))
    files += list((ROOT/'experiments/episode_return').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(files)}

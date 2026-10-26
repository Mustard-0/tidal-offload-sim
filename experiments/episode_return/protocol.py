"""Three validation-selected learning rates; one existing common seed."""
import hashlib
import json
from pathlib import Path
from experiments.lr_screen.protocol import configuration as old_configuration, digest

ROOT = Path(__file__).resolve().parents[2]


def configuration(method, lr=1):
    if method not in ('TMAPO', 'IPPO', 'PPO') or lr != 1:
        raise ValueError('Only the three existing validation-selected LR1 cells')
    cfg, train = old_configuration(method, lr)
    train.update(episodes=2000, episode_steps=6000, updates=93750, seed=87021,
                 checkpoint_episodes=5, checkpoint_seconds=300.)
    for key in ('validation_interval', 'validation_seeds', 'validation_phases',
                'validation_warmup_s', 'validation_measure_s', 'checkpoint_interval'):
        train.pop(key, None)
    assert train['updates']*train['rollout_steps'] == train['episodes']*train['episode_steps']
    return cfg, train


def plan():
    cfg, train = configuration('TMAPO')
    return dict(schema='episode-return-2000-20261026', environment=cfg, common_training=train,
        cells=[dict(id=f'{method}_LR1_seed87021', algorithm=method, lr_index=1,
                    actor_lr=3e-5, critic_lr=9e-5, env_steps=12000000)
               for method in ('TMAPO', 'IPPO', 'PPO')],
        reward='sum of the existing per-slot normalized team reward over 6000 slots',
        plot='x=completed episode (1..2000), y=episode return; raw plus labeled optional smoothing',
        selection='LR1 selected from previous fixed-validation tail mean+SD; no other seed has results',
        restart='new budget and learning-rate schedule; all three start from initialization',
        removed='periodic validation, validation task CSV, best-validation checkpoint, per-update CSV rewrites',
        unchanged='environment, privacy, observations, actions, reward, optimizer and rollout length',
        scope='single-seed exploratory convergence run, no algorithm superiority guarantee',
        total_training_env_steps=36000000, maximum_parallel=3, maximum_segments=96, maximum_days=7)


def source_hashes():
    files = list((ROOT/'model').glob('*.py')) + [ROOT/'model/config.json']
    files += list((ROOT/'experiments/lr_screen').rglob('*.py'))
    files += list((ROOT/'experiments/episode_return').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(files)}

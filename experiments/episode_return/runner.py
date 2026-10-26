"""Train complete episodes, log only episode return, checkpoint at safe updates."""
import argparse
import copy
import math
import os
from pathlib import Path
import time
import traceback
from .protocol import configuration, digest, source_hashes, ROOT
from experiments.lr_screen.runtime import (np, torch, seed_all, rng_state, rng_load,
    checkpoint_write, json_write, csv_write, runtime_info, available_ram_gib)
from experiments.lr_screen.algorithms import learner_for
from experiments.lr_screen.runner import new_environment


def train(out, method, resume=False, max_seconds=10800, component=None, stop_update=None):
    cfg, training = configuration(method)
    if component:
        training.update(component)
        training['updates'] = training['episodes']*training['episode_steps']//training['rollout_steps']
    if training['episodes']*training['episode_steps'] != training['updates']*training['rollout_steps']:
        raise ValueError('Budget must finish at a complete episode and update boundary')
    if training['episode_steps'] <= math.ceil(cfg['lifetime_s']/cfg['dt_s'])+2:
        raise ValueError('Episode must drain all terminal task costs')
    if available_ram_gib() < 1.:
        raise RuntimeError('Less than 1GiB RAM available')
    out = Path(out)
    if (out/'STOP_AFTER_UPDATE').exists():
        raise RuntimeError('Explicit stop marker')
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = out/'.worker.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    try:
        return _train(out, method, cfg, training, resume, max_seconds, component, stop_update)
    finally:
        os.close(fd)
        lock.unlink()


def _train(out, method, cfg, training, resume, max_seconds, component, stop_update):
    import json
    seed_all(training['seed'])
    actual_runtime = runtime_info('cpu')
    stable_runtime = {k: actual_runtime[k] for k in
                      ('python','numpy','scipy','torch','device','torch_threads')}
    identity = dict(schema='episode-return-v1', method=method, lr=1, environment=cfg,
        training=training, runtime=stable_runtime, source_hashes=source_hashes(),
        component_only=bool(component), transfer_integrated=True)
    identity_hash = digest(identity)
    if resume:
        manifest = json.loads((out/'manifest.json').read_text())
        if manifest['identity_hash'] != identity_hash or (out/'complete.json').exists():
            raise ValueError('Resume requires identical source, config, CPU runtime and incomplete run')
        manifest.setdefault('execution_runtimes', []).append(actual_runtime)
        json_write(out/'manifest.json', manifest)
    else:
        json_write(out/'manifest.json', dict(identity, identity_hash=identity_hash,
            execution_runtimes=[actual_runtime],
            source_commit=os.environ.get('GITHUB_SHA', 'local-component-check'),
            test_set_used=False, status='component' if component else 'formal'))
    learner = learner_for(method, training, cfg, 'cpu')
    episode_index, completed, update, partial_return, env_steps = 0, 0, 0, 0., 0
    env = new_environment(cfg, training['seed'], 0, training['episode_steps'])
    rows, previous_wall, previous_train_wall = [], 0., 0.
    if resume:
        saved = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        if saved['identity_hash'] != identity_hash:
            raise ValueError('Checkpoint identity mismatch')
        learner.load(saved['learner'])
        rng_load(saved['rng'])
        env, episode_index, completed = saved['environment'], saved['episode_index'], saved['completed_episodes']
        update, env_steps = saved['update'], saved['env_steps']
        partial_return, rows = saved['partial_return'], saved['rows']
        previous_wall, previous_train_wall = saved['wall_s'], saved['training_wall_s']
        # A log can be ahead of its last atomic checkpoint after power loss.
        # Retain that evidence before restoring its exact checkpoint prefix.
        if (out/'episode_returns.csv').exists():
            recovery = out/'recovery'/str(time.time_ns())
            recovery.mkdir(parents=True)
            (recovery/'episode_returns.csv').write_bytes((out/'episode_returns.csv').read_bytes())
        if rows:
            csv_write(out/'episode_returns.csv', rows)
    started = time.monotonic()
    train_wall = previous_train_wall
    last_checkpoint = started
    checkpoint_episode = completed

    def checkpoint():
        checkpoint_write(out/'latest.pt', dict(identity_hash=identity_hash, learner=learner.state(),
            rng=rng_state(), environment=env, episode_index=episode_index,
            completed_episodes=completed, update=update, env_steps=env_steps,
            partial_return=partial_return, rows=rows,
            wall_s=previous_wall+time.monotonic()-started, training_wall_s=train_wall))
        json_write(out/'checkpoint_status.json', dict(update=update, env_steps=env_steps,
            completed_episodes=completed, partial_episode_slots=env.slot % training['episode_steps'],
            training_wall_s=train_wall, component_only=bool(component)))

    try:
        if not resume:
            checkpoint()
        for update in range(update+1, training['updates']+1):
            tick = time.monotonic()
            learner.set_progress(update, training['updates'])
            before = completed
            for _ in range(training['rollout_steps']):
                if env.slot == training['episode_steps']:
                    if env.tasks:
                        raise AssertionError('Unresolved tasks at terminal boundary')
                    episode_index += 1
                    env = new_environment(cfg, training['seed'], episode_index, training['episode_steps'])
                reward, records, _, _ = env.step(learner, arrivals=env.slot < env.arrival_stop_slot)
                if not math.isfinite(float(reward)):
                    raise FloatingPointError('Nonfinite reward')
                done = env.slot == training['episode_steps']
                learner.remember(env.agent_rewards(0.), records, done)
                partial_return += float(reward)
                env_steps += 1
                # WAIT is masked in this unchanged immediate-release protocol;
                # unused admissions older than its one-second history horizon
                # cannot enter any action. Bound checkpoint memory growth.
                cutoff = env.time-cfg['wait_min_history_s']
                for history in env.admission_history.values():
                    while history and history[0][0] < cutoff:
                        history.popleft()
                if done:
                    if env.tasks:
                        raise AssertionError('Terminal cost/queue not drained')
                    completed += 1
                    rows.append(dict(episode=completed, env_steps=env_steps,
                        episode_return=partial_return, slots=training['episode_steps'],
                        train_trace_hash=env.arrival_hash))
                    partial_return = 0.
            context, glob, *_ = env.observations()
            stats = learner.update(context, glob)
            if not all(math.isfinite(float(v)) for v in stats.values()):
                raise FloatingPointError('Nonfinite optimizer diagnostics')
            if any(not torch.isfinite(p).all() for module in (learner.actors, learner.critic)
                   for p in module.parameters()):
                raise FloatingPointError('Nonfinite parameters')
            train_wall += time.monotonic()-tick
            if completed != before:
                csv_write(out/'episode_returns.csv', rows)
                print('EPISODE', method, completed, 'env_steps', env_steps, flush=True)
            reason = ('stop marker' if (out/'STOP_AFTER_UPDATE').exists() else
                'available RAM below .5GiB' if available_ram_gib() < .5 else
                'wall budget' if time.monotonic()-started >= max_seconds else
                'component boundary' if stop_update and update >= stop_update else None)
            terminal = update == training['updates']
            if (terminal or reason or completed-checkpoint_episode >= training['checkpoint_episodes']
                    or time.monotonic()-last_checkpoint >= training['checkpoint_seconds']):
                checkpoint()
                checkpoint_episode, last_checkpoint = completed, time.monotonic()
            if reason and not terminal:
                json_write(out/'paused.json', dict(update=update, reason=reason))
                return False
        assert completed == training['episodes'] and env_steps == training['episodes']*training['episode_steps']
        assert env.slot == training['episode_steps'] and not env.tasks
        json_write(out/'complete.json', dict(updates=update, env_steps=env_steps,
            episodes=completed, component_only=bool(component)))
        if (out/'paused.json').exists():
            (out/'paused.json').unlink()
        return True
    except BaseException as error:
        json_write(out/'failure.json', dict(message=str(error), traceback=traceback.format_exc(),
            last_logged_env_steps=env_steps, recover_from='latest.pt'))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=('TMAPO','IPPO','PPO'), required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-seconds', type=float, default=10800)
    a = p.parse_args()
    train(a.out, a.method, a.resume, a.max_seconds)


if __name__ == '__main__':
    main()

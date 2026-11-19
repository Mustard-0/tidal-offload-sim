"""Train matched baselines and evaluate the frozen eight-method suite on CPU."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import time

import optimization2_worker as base
import scalability_protocol as spec
from experiments.lr_screen.github_api import download
from experiments.lr_screen.transport import sha, unseal
from review_checks import bootstrap, check_metrics


def transport():
    base.BRANCH, base.CAMPAIGN, base.plan = spec.BRANCH, spec.CAMPAIGN, spec.plan
    t = base.Transport()
    for name, digest in t.inputs['plan']['inherited_code'].items():
        assert sha(Path(name).read_bytes().replace(b'\r\n', b'\n')) == digest, name
    return t


def prior_policy(t, seed):
    from optimization2_core import deserialize
    from pressure9.policies import evaluation_policy, weights_only

    release = t.api.request('GET', base.BASE + '/releases/tags/' + spec.PRIOR_CAMPAIGN)
    assert release['id'] == 450925129 and release['target_commitish'] == spec.BASE_COMMIT
    name = f'full35_seed{seed}.part1'
    assets = {a['name']: a for a in t.api.request('GET', base.BASE + f"/releases/{release['id']}/assets?per_page=100")}
    asset = assets[name + '.checkpoint.enc']
    size, digest, train_input = spec.PRIOR[str(seed)]
    assert asset['size'] == size and asset['digest'] == 'sha256:' + digest
    data = download(asset)
    assert len(data) == size and sha(data) == digest
    raw, _ = unseal(data, t.key, dict(campaign=spec.PRIOR_CAMPAIGN, source_commit=spec.BASE_COMMIT,
        run_id=spec.PRIOR_RUN, attempt='1', inputs_sha256=spec.PRIOR_INPUTS, cell=name,
        purpose='optimization-checkpoint'))
    saved = deserialize(raw)
    identity, env = saved['identity'], saved['environment']
    assert (identity['cell'], identity['source_commit'], identity['plan_hash']) == (
        f'full35_seed{seed}', spec.BASE_COMMIT, spec.PRIOR_PLAN)
    assert env['slot'] == spec.TOTAL_STEPS and not env['tasks'] and env['full_input_hash'] == train_input
    assert len(saved['training_rows']) == spec.TOTAL_STEPS // spec.BLOCK
    cfg, train = spec.settings('TMAPO', seed, 150)
    assert env['cfg'] == cfg
    policy = evaluation_policy('TMAPO', train, cfg, saved['learner'])
    return weights_only(policy), train_input


def baseline_train(cell, value, deadline, smoke=False):
    from experiments.lr_screen.runtime import torch, seed_all
    from pilot_worker import module_hash
    from pressure9.policies import learner, set_progress, update, weights_only
    from review_environment import environment_types

    plain, _, _, _ = environment_types()
    method, seed = spec.split_cell(cell)
    cfg, train = spec.settings(method, seed, 20 if smoke else 150)
    seed_all(seed)
    env = plain(cfg, family=method, seed=1191900 + seed)
    policy = learner(method, train, cfg)
    initial = module_hash(policy)
    steps = 300 if smoke else spec.TOTAL_STEPS
    for block in range(1, steps // spec.BLOCK + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError('Baseline training wall budget')
        set_progress(policy, block, 1208)
        returns = []
        for _ in range(spec.BLOCK):
            reward, records, *_ = env.step(policy, arrivals=env.slot < spec.ADMISSION_STEPS)
            assert math.isfinite(reward)
            returns.append(float(reward))
            policy.remember(env.agent_rewards(0.), records, env.slot == steps)
        stats = update(policy, *env.observations()[:2])
        assert all(math.isfinite(float(x)) for x in stats.values())
        value['training_rows'].append(dict(update=block, env_steps=env.slot, block_return=sum(returns), **stats))
    nets = (policy.actors, policy.critic) if hasattr(policy, 'actors') else (policy.actor, policy.q)
    assert all(torch.isfinite(w).all() for net in nets for w in net.parameters())
    value.update(weights_hash=module_hash(policy), env_steps=env.slot,
        training_input_hash=env.full_input_hash, pending_tasks=len(env.tasks))
    if not smoke:
        check_metrics(env)
        assert value['weights_hash'] != initial
        assert env.full_input_hash == spec.PRIOR[str(seed)][2], 'Training inputs differ from TT-MAPPO'
    return weights_only(policy)


def eval_shard(t, cell, trace, value, deadline):
    from experiments.lr_screen.runtime import torch, seed_all
    from optimization2_core import deserialize
    from pilot_worker import module_hash
    from pressure9.policies import evaluation_policy
    from review_environment import environment_types

    method, seed = spec.split_cell(cell)
    plain, tt, greedy, local = environment_types()
    expected_model = None
    if method == 'TMAPO':
        weights, value['training_input_hash'] = prior_policy(t, seed)
    elif seed is not None:
        train = t.read_small(cell + '.train.enc', cell + '.train')
        name = cell + '.weights.enc'
        assert train['completed'] and train['env_steps'] == spec.TOTAL_STEPS
        assert train['weights']['digest'] == t.assets[name]['digest']
        weights = deserialize(t.get(name, cell, 'scale-up-terminal-weights', 4 * 1024**2))
        value['training_input_hash'] = train['training_input_hash']
        expected_model = train['weights_hash']
    assert seed is None or value['training_input_hash'] == spec.PRIOR[str(seed)][2]
    for n in spec.POPULATIONS:
        if time.monotonic() >= deadline:
            raise TimeoutError('Evaluation wall budget before next population')
        seed_all(trace)
        cfg, train = spec.settings(method, seed, n)
        window = dict(measure_start=spec.WARMUP, measure_end=spec.WARMUP + spec.MEASURE, seed=trace, phase_s=0.)
        env = tt(cfg, **window) if method == 'TMAPO' else plain(cfg, family=method, **window)
        policy = ((local() if method == 'Local-Only' else greedy(cfg)) if seed is None
            else evaluation_policy(method, train, cfg, weights))
        before = module_hash(policy) if seed is not None else None
        if expected_model is not None:
            assert before == expected_model
        slots = round((spec.WARMUP + spec.MEASURE) / cfg['dt_s'])
        with torch.no_grad():
            for slot in range(slots + 203):
                env.step(policy, arrivals=slot < slots)
                if env.slot >= slots and not env.tasks:
                    metrics = check_metrics(env)
                    break
                if slot % 100 == 99 and time.monotonic() >= deadline:
                    raise TimeoutError(f'Evaluation wall budget during N={n}; unfinished scene omitted')
            else:
                raise AssertionError('Full-tide evaluation failed to drain')
        if seed is not None:
            assert module_hash(policy) == before
        value['evaluation_rows'].append(dict(method=method, training_seed=seed, users=n, trace_seed=trace,
            phase_s=0., warmup_s=spec.WARMUP, measure_s=spec.MEASURE,
            full_input_hash=env.full_input_hash, model_hash=before, **metrics))
        print(f'Saved complete scene locally: {cell}, trace={trace}, N={n}', flush=True)


def run(action, cell=None, trace=None):
    if action == 'train':
        assert cell in spec.TRAIN_CELLS and trace is None
    if action == 'evaluate':
        assert (cell, trace) in spec.EVAL_CELLS
    bootstrap()
    from experiments.lr_screen.runtime import torch
    from optimization2_core import serialize
    torch.set_num_threads(1)
    t = transport()
    name = 'gate' if action == 'gate' else f'{cell}.train' if action == 'train' else f'{cell}.trace{trace}'
    value = dict(action=action, cell=cell, trace_seed=trace, completed=False,
        evaluation_rows=[], training_rows=[])
    started = time.monotonic()
    try:
        assert base.within_dispatch_window(t), '24-hour stage admission window expired'
        if action == 'gate':
            for seed in spec.DEVELOPMENT_SEEDS:
                prior_policy(t, seed)
            for m in spec.LEARNERS[1:]:
                proof = dict(training_rows=[])
                baseline_train(f'{m}_seed{spec.DEVELOPMENT_SEEDS[0]}', proof, started + 950, smoke=True)
                assert len(proof['training_rows']) == 3
            value['component_only'] = True
        elif action == 'train':
            weights = baseline_train(cell, value, started + spec.TRAIN_WALL)
            value['weights'] = t.upload(cell + '.weights.enc', serialize(weights), cell,
                'scale-up-terminal-weights', 4 * 1024**2)
        else:
            eval_shard(t, cell, trace, value, started + spec.EVAL_WALL)
            assert len(value['evaluation_rows']) == len(spec.POPULATIONS)
        value['completed'] = True
    except Exception as error:
        value['error'] = dict(type=type(error).__name__, message=str(error))
    value['wall_seconds'] = time.monotonic() - started
    t.small(name + '.enc', value, name)
    print(f'Scale-up outcome: {name}; completed={value["completed"]}', flush=True)
    if value['completed'] and action == 'gate':
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as file:
            file.write('train=' + json.dumps(dict(cell=spec.TRAIN_CELLS)) + '\n')
            file.write('evaluate=' + json.dumps(dict(include=[dict(cell=c, trace=s) for c, s in spec.EVAL_CELLS])) + '\n')
    return value['completed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gate', 'train', 'evaluate'))
    parser.add_argument('--cell', choices=spec.CELLS)
    parser.add_argument('--trace', type=int, choices=spec.TRACES)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.action, args.cell, args.trace) else 2)

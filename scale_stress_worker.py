"""Full-tide outcomes for pressure arms; reuse only the authenticated parent controllers."""
import argparse
import json
import math
import os
import time

import optimization2_worker as transport_runtime
import scale_stress_protocol as spec
from review_checks import bootstrap, check_metrics


def transport():
    transport_runtime.BRANCH = spec.BRANCH
    transport_runtime.CAMPAIGN = spec.CAMPAIGN
    transport_runtime.plan = spec.plan
    return transport_runtime.Transport()


def parent_link(t):
    from copy import copy
    base = transport_runtime.BASE
    run = t.api.request('GET', base + f'/actions/runs/{spec.PARENT_RUN}')
    assert run['run_attempt'] == 1 and run['head_sha'] == spec.BASE_COMMIT
    release = t.api.request('GET', base + '/releases/tags/' + spec.PARENT_CAMPAIGN)
    assert release['id'] == spec.PARENT_RELEASE and release['target_commitish'] == spec.BASE_COMMIT
    prior = copy(t)
    prior.release, prior.commit, prior.run_id = release, spec.BASE_COMMIT, str(spec.PARENT_RUN)
    prior.inputs_sha256, prior.plan_hash = spec.PARENT_INPUTS, spec.PARENT_PLAN
    prior.assets = {a['name']: a for a in t.api.request('GET', base + f"/releases/{release['id']}/assets?per_page=100")}
    assert prior.assets['inputs.enc']['digest'] == 'sha256:' + spec.PARENT_INPUTS

    def prior_header(cell, purpose):
        return dict(campaign=spec.PARENT_CAMPAIGN, source_commit=spec.BASE_COMMIT, run_id=str(spec.PARENT_RUN),
            attempt='1', inputs_sha256=spec.PARENT_INPUTS, cell=cell, purpose=purpose)

    prior.header = prior_header
    return prior


def await_parent_training(t, deadline):
    required = {'gate'} | {f'train ({cell})' for cell in spec.TRAIN_CELLS}
    api_path = transport_runtime.BASE + f'/actions/runs/{spec.PARENT_RUN}/attempts/1/jobs?per_page=100'
    previous = None
    while True:
        jobs = {r['name']: r for r in t.api.request('GET', api_path)['jobs']}
        assert required <= jobs.keys()
        failures = [n for n in required if jobs[n]['status'] == 'completed' and jobs[n]['conclusion'] != 'success']
        if failures:
            raise ValueError('Parent training failed: ' + ', '.join(sorted(failures)))
        completed = sum(jobs[n]['conclusion'] == 'success' for n in required)
        if completed == len(required):
            return
        if completed != previous:
            print(f'Waiting for pinned parent receipts: {completed}/{len(required)} jobs', flush=True)
            previous = completed
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Parent training did not finish within the dependency wait budget')
        time.sleep(min(60., remaining))


def baseline_policy(prior, cell):
    from optimization2_core import deserialize
    train = prior.read_small(cell + '.train.enc', cell + '.train')
    method, seed = spec.split_cell(cell)
    assert method not in ('TMAPO', *spec.RULES) and seed is not None
    name = cell + '.weights.enc'
    assert train['completed'] and train['env_steps'] == spec.TOTAL_STEPS and train['pending_tasks'] == 0
    assert len(train['training_rows']) == 302 and train['training_input_hash'] == spec.PRIOR[str(seed)][2]
    assert train['weights']['digest'] == prior.assets[name]['digest'] and train['weights']['size'] == prior.assets[name]['size']
    return deserialize(prior.get(name, cell, 'scale-up-terminal-weights', 4 * 1024**2)), train['weights_hash']


def evaluate(t, scenario, cell, trace, result, deadline):
    from experiments.lr_screen.runtime import torch, seed_all
    from pilot_worker import module_hash
    from pressure9.policies import evaluation_policy
    from scalability_worker import prior_policy
    from scale_diagnostics_environment import diagnostic_environment_types

    _, plain, tt, greedy, local = diagnostic_environment_types()
    method, seed = spec.split_cell(cell)
    model_hash = None
    if method == 'TMAPO':
        weights, training_hash = prior_policy(t, seed)
        assert training_hash == spec.PRIOR[str(seed)][2]
    elif seed is not None:
        weights, model_hash = baseline_policy(parent_link(t), cell)
    for n in spec.POPULATIONS:
        if time.monotonic() >= deadline:
            raise TimeoutError('Wall budget before next complete population scene')
        cfg, train = spec.settings(scenario, method, seed, n)
        seed_all(trace)
        opts = dict(seed=trace, phase_s=0., measure_start=spec.WARMUP, measure_end=spec.WARMUP+spec.MEASURE)
        env = tt(cfg, **opts) if method == 'TMAPO' else plain(cfg, family=method, **opts)
        policy = ((local() if method == 'Local-Only' else greedy(cfg)) if seed is None
            else evaluation_policy(method, train, cfg, weights))
        before = module_hash(policy) if seed is not None else None
        if model_hash is None:
            model_hash = before
        assert before == model_hash
        dt = cfg['dt_s']
        slots = round((spec.WARMUP + spec.MEASURE) / dt)
        with torch.no_grad():
            for i in range(slots+math.ceil(cfg['lifetime_s']/dt)+3):
                env.step(policy, arrivals=i < slots)
                if env.slot >= slots and not env.tasks:
                    break
                if i % 100 == 99 and time.monotonic() >= deadline:
                    raise TimeoutError(f'Wall budget at N={n}; unfinished scene excluded')
            else:
                raise AssertionError('Measured arrival cohort did not drain')
        metrics = check_metrics(env)
        assert env.pressure_samples == round(spec.MEASURE/dt)
        assert 0. <= metrics['remote_compute_utilization'] <= 1. + 1e-9
        assert seed is None or module_hash(policy) == before
        result['evaluation_rows'].append(dict(method=method, training_seed=seed, users=n, trace_seed=trace,
            scenario=scenario, warmup_s=spec.WARMUP, measure_s=spec.MEASURE, phase_s=0.,
            model_hash=before, full_input_hash=env.full_input_hash, **metrics))
        print(f'{scenario} {cell} trace {trace}: completed N={n}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('gate', 'evaluate'))
    parser.add_argument('--scenario', choices=tuple(spec.FACTORS)[1:])
    parser.add_argument('--cell', choices=spec.CELLS)
    parser.add_argument('--trace', type=int, choices=spec.TRACES)
    args = parser.parse_args()
    bootstrap()
    from experiments.lr_screen.runtime import torch
    from scalability_worker import prior_policy
    torch.set_num_threads(1)
    t = transport()
    name = 'gate' if args.action == 'gate' else f'{args.scenario}.{args.cell}.trace{args.trace}'
    value = dict(action=args.action, completed=False, evaluation_rows=[], scenario=args.scenario,
        cell=args.cell, trace_seed=args.trace)
    try:
        assert transport_runtime.within_dispatch_window(t)
        if args.action == 'gate':
            await_parent_training(t, time.monotonic()+spec.PARENT_WAIT_WALL)
            prior = parent_link(t)
            for cell in spec.TRAIN_CELLS:
                baseline_policy(prior, cell)
            for seed in spec.DEVELOPMENT_SEEDS:
                prior_policy(t, seed)
            value['fixed_learned_controllers_verified'] = 10
        else:
            assert (args.scenario, args.cell, args.trace) in spec.SHARDS
            evaluate(t, args.scenario, args.cell, args.trace, value, time.monotonic()+spec.EVAL_WALL)
            assert len(value['evaluation_rows']) == len(spec.POPULATIONS)
        value['completed'] = True
    except Exception as error:
        value['error'] = dict(type=type(error).__name__, message=str(error))
    t.small(name + '.enc', value, name)
    if value['completed'] and args.action == 'gate':
        matrix = dict(include=[dict(scenario=z, cell=c, trace=s) for z, c, s in spec.SHARDS])
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as out:
            out.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
    raise SystemExit(0 if value['completed'] else 2)


if __name__ == '__main__':
    main()

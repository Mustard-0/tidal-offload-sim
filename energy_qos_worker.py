"""Single finite CPU run, reusing authenticated frozen weights."""

import argparse
from copy import copy
import json
import math
import os
import time

from evaluation_adapters import evaluation_environment_types
import energy_qos_protocol as spec
from energy_qos_telemetry import with_task_telemetry
import optimization2_worker as store
from review_checks import bootstrap, check_metrics
from scalability_protocol import PRIOR, TOTAL_STEPS, settings, split_cell


def open_transport():
    store.BRANCH, store.CAMPAIGN, store.plan = spec.BRANCH, spec.CAMPAIGN, spec.plan
    return store.Transport()


def parent_store(t):
    api = t.api
    run = api.request('GET', store.BASE+f'/actions/runs/{spec.PARENT_RUN}')
    assert run['id'] == spec.PARENT_RUN and run['run_attempt'] == 1 and run['head_sha'] == spec.BASE_COMMIT
    assert run['status'] == 'completed' and run['conclusion'] == 'success'
    rel = api.request('GET', store.BASE+'/releases/tags/'+spec.PARENT_CAMPAIGN)
    assert rel['id'] == spec.PARENT_RELEASE and rel['target_commitish'] == spec.BASE_COMMIT
    parent = copy(t)
    parent.release, parent.plan_hash = rel, spec.PARENT_PLAN
    parent.assets = {r['name']: r for r in api.request('GET', store.BASE+f"/releases/{rel['id']}/assets?per_page=100")}
    assert parent.assets['inputs.enc']['digest'] == 'sha256:'+spec.PARENT_INPUTS
    parent.header = lambda cell, purpose: dict(campaign=spec.PARENT_CAMPAIGN,
        source_commit=spec.BASE_COMMIT, run_id=str(spec.PARENT_RUN), attempt='1',
        inputs_sha256=spec.PARENT_INPUTS, cell=cell, purpose=purpose)
    return parent


def weights_for(t, parent, cell, method, seed):
    from optimization2_core import deserialize
    from scalability_worker import prior_policy

    if seed is None:
        return None, None
    if method == 'TMAPO':
        weights, train_input = prior_policy(t, seed)
        assert train_input == PRIOR[str(seed)][2]
        return weights, None
    receipt = parent.read_small(cell+'.train.enc', cell+'.train')
    name = cell+'.weights.enc'
    assert receipt['cell'] == cell and receipt['completed'] and receipt['env_steps'] == TOTAL_STEPS
    assert receipt['pending_tasks'] == 0 and receipt['training_input_hash'] == PRIOR[str(seed)][2]
    assert receipt['weights']['digest'] == parent.assets[name]['digest']
    assert receipt['weights']['size'] == parent.assets[name]['size']
    return deserialize(parent.get(name, cell, 'scale-up-terminal-weights', 4*1024**2)), receipt['weights_hash']


def close_parent_scene(row, old):
    for key in ('full_input_hash', 'model_hash', 'arrivals', 'completed', 'failed'):
        assert old[key] == row[key], key
    for key in ('delay_s', 'energy_j_per_arrival', 'qos_rate', 'mean_task_cost',
                'transmit_energy_j_per_arrival', 'compute_energy_j_per_arrival', 'report_energy_j_per_arrival'):
        assert math.isclose(old[key], row[key], abs_tol=1e-12, rel_tol=1e-8), key


def execute(t, mode, name, trace, result):
    bootstrap()
    from experiments.lr_screen.runtime import torch, seed_all
    from pressure9.policies import evaluation_policy
    from pilot_worker import module_hash

    torch.set_num_threads(1)
    cell = name[:-len(spec.OPT)] if name.endswith(spec.OPT) else name
    method, seed = split_cell(cell)
    plain, tt, greedy, local = evaluation_environment_types(optimize_greedy=name != cell)
    env_type = with_task_telemetry(tt if method == 'TMAPO' else plain)
    parent = parent_store(t) if seed is not None else None
    weights, expected_model = weights_for(t, parent, cell, method, seed)
    if mode == 'development':
        cases = sorted(spec.DEV_KEYS)
        measure = spec.DEV_MEASURE
        deadline = time.monotonic()+spec.DEV_WALL
        previous = {}
    else:
        frozen = f'{cell}.trace{trace}'
        prev = parent.read_small(frozen+'.enc', frozen)
        assert prev['completed'] and prev['trace_seed'] == trace and prev['cell'] == cell
        previous = {r['users']: r for r in prev['evaluation_rows']}
        cases = [(n, trace, 0.) for n in spec.REPLAY_POP]
        measure = spec.REPLAY_MEASURE
        deadline = time.monotonic()+spec.REPLAY_WALL
    for n, trace, phase in cases:
        if time.monotonic() >= deadline:
            raise TimeoutError('Wall budget before next complete scene')
        cfg, train = settings(method, seed, n)
        seed_all(trace)
        opts = dict(seed=trace, phase_s=phase, measure_start=spec.WARMUP, measure_end=spec.WARMUP+measure)
        env = env_type(cfg, **opts) if method == 'TMAPO' else env_type(cfg, family=method, **opts)
        if seed is None:
            policy, identity = (local() if method == 'Local-Only' else greedy(cfg)), None
        else:
            policy = evaluation_policy(method, train, cfg, weights)
            identity = module_hash(policy)
            if expected_model is None:
                expected_model = identity
            assert identity == expected_model
        stop = round((spec.WARMUP+measure)/cfg['dt_s'])
        with torch.no_grad():
            for i in range(stop+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
                env.step(policy, arrivals=i < stop)
                if env.slot >= stop and not env.tasks:
                    break
                if i % 100 == 99 and time.monotonic() >= deadline:
                    raise TimeoutError(f'Wall budget within N={n}; unfinished scene omitted')
            else:
                raise AssertionError('Natural drainage did not finish')
        assert seed is None or module_hash(policy) == identity
        metrics = check_metrics(env)
        # Queue/utilization telemetry is missing in the inherited plain controller.
        for k in ('mean_queue_pressure', 'high_queue_region_time_fraction', 'remote_compute_utilization',
                  'region_0_high_queue_fraction', 'region_1_high_queue_fraction', 'region_2_high_queue_fraction'):
            metrics.pop(k, None)
        row = dict(method=method, training_seed=seed, users=n, trace_seed=trace,
                   phase_s=phase, measure_s=measure, full_input_hash=env.full_input_hash,
                   model_hash=identity, **metrics, energy_by_task_group=env.accounting(metrics))
        if previous:
            close_parent_scene(row, previous[n])
            row['reproduces_parent'] = True
        result['rows'].append(row)
        print(f'Saved complete {mode} scene, N={n}, trace={trace}, phase={phase}', flush=True)
    assert len(result['rows']) == len(cases)


def gate(t):
    bootstrap()
    from review_checks import rollout
    parents = parent_store(t)
    for cell in spec.TRAIN_CELLS:
        assert cell+'.weights.enc' in parents.assets and cell+'.train.enc' in parents.assets
    signatures = {}
    for optimize in (False, True):
        plain, _, greedy, _ = evaluation_environment_types(optimize_greedy=optimize)
        for method in spec.GREEDY:
            cfg, _ = settings(method, None, 20)
            env = with_task_telemetry(plain)(cfg, family=method, seed=spec.DEV_TRACES[0])
            m = rollout(env, greedy(cfg), admission=30)
            env.accounting(m)
            fingerprint = env.full_input_hash, m['arrivals']
            assert signatures.setdefault(20, fingerprint) == fingerprint
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        f.write('dev='+json.dumps(dict(cell=spec.DEV_CELLS))+'\n')
        f.write('replay='+json.dumps(dict(include=[dict(cell=c, trace=r) for c,r in spec.REPLAY_SHARDS]))+'\n')
    t.small('gate.enc', dict(completed=True, rule_smokes=4, parent_run=spec.PARENT_RUN), 'gate')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('gate','development','select','replay'))
    parser.add_argument('--cell')
    parser.add_argument('--trace', type=int)
    args = parser.parse_args()
    t = open_transport()
    assert store.within_dispatch_window(t)
    if args.mode == 'gate':
        return gate(t)
    if args.mode == 'select':
        cells = {c:t.read_small(c+'.enc', c) for c in spec.DEV_CELLS if c+'.enc' in t.assets and
                 c.removesuffix(spec.OPT) in spec.GREEDY}
        result = spec.choose(cells)
        t.small('selection.enc', result, 'selection')
        return result['complete']
    if args.mode == 'replay':
        assert (args.cell, args.trace) in spec.REPLAY_SHARDS
        name = f'replay.{args.cell}.trace{args.trace}'
    else:
        assert args.cell in spec.DEV_CELLS and args.trace is None
        name = args.cell
    result = dict(completed=False, mode=args.mode, cell=args.cell, rows=[])
    try:
        execute(t, args.mode, args.cell, args.trace, result)
        result['completed'] = True
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
    t.small(name+'.enc', result, name)
    return result['completed']


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)

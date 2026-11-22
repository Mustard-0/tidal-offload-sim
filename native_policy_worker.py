"""Explicit evaluation runner: independent policies, identical exogenous inputs."""

import argparse
import math
import os
import time

import baseline_individual_worker as transport
from energy_qos_worker import parent_store, weights_for
from evaluation_adapters import GREEDY_PAIR, evaluation_environment_types
import lowload_worker as accounting
from native_policy_eval import ActionMeter, QuotedGreedy, with_greedy_quotes
from native_policy_protocol import BRANCHES, stage
import optimization2_worker as store
from paper_load_worker import with_latency_bins
from review_checks import bootstrap, check_metrics


def evaluate(t, spec, method, arm, result):
    bootstrap()
    from experiments.lr_screen.runtime import seed_all, torch
    from pilot_worker import module_hash
    from pressure9.policies import evaluation_policy

    torch.set_num_threads(1)
    plain, tt, _, local = evaluation_environment_types()
    base = tt if method == 'TMAPO' else plain
    if method in GREEDY_PAIR:
        base = with_greedy_quotes(base)
    env_type = with_latency_bins(accounting.with_deadline_sweep(base))
    seed = None if method in spec.RULES else spec.SEED
    cell = method if seed is None else f'{method}_seed{seed}'
    parent = parent_store(t) if seed is not None and method != 'TMAPO' else None
    weights, expected = weights_for(t, parent, cell, method, seed)
    expiry = time.monotonic() + spec.WALL
    for mix, n, trace, phase in spec.CASES:
        assert time.monotonic() < expiry, 'Wall budget before next scene'
        seed_all(trace)
        cfg, train = spec.settings(method, seed, n)
        cfg = spec.settings_for_mix(cfg, mix)
        opts = dict(seed=trace, phase_s=phase, measure_start=spec.WARM,
                    measure_end=spec.WARM+spec.MEASURE)
        env = env_type(cfg, **opts) if method == 'TMAPO' else env_type(cfg, family=method, **opts)
        model = None
        if method == 'Local-Only':
            policy = local()
        elif method in GREEDY_PAIR:
            policy = QuotedGreedy(env)
        else:
            policy = evaluation_policy(method, train, cfg, weights)
            model = module_hash(policy)
            expected = model if expected is None else expected
            assert model == expected
        meter = ActionMeter(policy, env)
        stop = round((spec.WARM + spec.MEASURE) / cfg['dt_s'])
        with torch.no_grad():
            for i in range(stop+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
                env.step(meter, arrivals=i < stop)
                if env.slot >= stop and not env.tasks:
                    break
                if i % 100 == 99 and time.monotonic() >= expiry:
                    raise TimeoutError(f'Wall budget in {mix} N={n}')
            else:
                raise AssertionError('Natural drainage did not finish')
        assert seed is None or module_hash(policy) == model
        m = check_metrics(env)
        result['rows'].append(dict(mix=mix, method=method, arm=arm, training_seed=seed, users=n,
            trace_seed=trace, phase_s=phase, measure_s=spec.MEASURE, model_hash=model,
            full_input_hash=env.full_input_hash, arrival_hash=env.arrival_hash, admission_edits=0,
            energy_by_task_group=env.accounting(m), **meter.metrics(), **m))
        print(f'Completed {method} N={n} phase={phase:g}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('arm')
    args = parser.parse_args()
    [name] = [k for k,b in BRANCHES.items() if os.environ['GITHUB_REF'] == 'refs/heads/'+b]
    spec = stage(name)
    assert (args.method,args.arm) in spec.ARMS
    transport.spec = accounting.spec = spec
    store.CAMPAIGN = spec.CAMPAIGN
    t = transport.Transport()
    cell = args.method+'.'+args.arm
    result = dict(method=args.method, arm=args.arm, completed=False,
                  rows=accounting.Rows(t, cell, args.method, args.arm))
    try:
        assert store.within_dispatch_window(t)
        evaluate(t, spec, args.method, args.arm, result)
        assert len(result['rows']) == len(spec.CASES)
        result['completed'] = True
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
    t.small(cell+'.enc', result, cell)
    return result['completed']


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)

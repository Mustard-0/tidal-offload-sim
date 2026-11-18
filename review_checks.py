"""Real simulator regression checks and bounded gradient smoke tests."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

from review_protocol import ABLATIONS, DEVELOPMENT_SEEDS, DEVELOPMENT_TRACES, LEARNERS, POPULATIONS, RULES, settings


def bootstrap():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    root = Path(__file__).resolve().parent
    if not (root / 'model').exists():
        root = Path(os.environ['REVIEW_SOURCE_ROOT'])
        if os.environ.get('REVIEW_EXTRA_DEPS'):
            sys.path.insert(0, os.environ['REVIEW_EXTRA_DEPS'])
    sys.path[:0] = [str(root), str(root / 'model')]


def check_metrics(e):
    assert not e.tasks and e.arrivals == e.ended
    m = e.metrics()
    n = m['arrivals']
    assert n > 0 and m['completed'] + m['failed'] == n
    assert m['number_queued_pending'] == 0
    assert math.isclose(m['energy_j_per_arrival'], sum(m[k + '_energy_j_per_arrival'] for k in ('transmit', 'compute', 'report')), rel_tol=1e-8, abs_tol=1e-12)
    assert math.isclose(m['qos_rate'], m['failed'] / n + m['late_completed_rate'], abs_tol=1e-12)
    assert 0 <= m['qos_rate'] <= 1 and math.isfinite(m['mean_task_cost'])
    return m


def rollout(e, p, admission=80):
    for slot in range(admission + 203):
        e.step(p, arrivals=slot < admission)
        if e.slot >= admission and not e.tasks:
            return check_metrics(e)
    raise AssertionError('Natural drainage exceeded task lifetime')


def environment_checks():
    bootstrap()
    import numpy as np
    from review_environment import environment_types
    Plain, TM, Greedy, Local = environment_types()
    results = []
    signatures = {}
    for n in POPULATIONS:
        for method in RULES:
            c, _ = settings(method, n=n)
            e = Plain(c, family=method, seed=DEVELOPMENT_TRACES[0], collect=False)
            p = Local() if method == 'Local-Only' else Greedy(c)
            m = rollout(e, p)
            sig = (e.full_input_hash, m['arrivals'])
            assert signatures.setdefault(n, sig) == sig
            assert m['class_gate_wait_actions'] == m['gate_allowed'] == 0
            assert m['maximum_distinct_class_batch'] <= 1
            if method in ('Greedy-NoPrivacy', 'Local-Only'):
                assert m['report_energy_j_per_arrival'] == 0.
            if method == 'Local-Only':
                assert m['published'] == 0 and m['transmit_energy_j_per_arrival'] == 0.
            results.append(dict(method=method, users=n, input_hash=e.full_input_hash, **m))
    # Mutating a TMAPO-only predictor must not alter ANY baseline observations.
    for method in LEARNERS[1:] + RULES:
        c, _ = settings(method)
        e = Plain(c, family=method, seed=DEVELOPMENT_TRACES[0])
        e.prepare(True)
        a = e.observations()
        e.previous_forecast[:] = 12345.
        e.pred_error.append(np.array([3., 4., 5.]))
        b = e.observations()
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert not hasattr(e, 'class_staging') and not hasattr(e, 'wait_information')
        try:
            e.agent_rewards(.7)
        except ValueError:
            pass
        else:
            raise AssertionError('Baseline accepted TMAPO local credit')
    # Removing forecast features and waiting must actually affect their code paths.
    for variant in ABLATIONS:
        c, _ = settings('TMAPO', ablation=variant)
        e = TM(c, seed=DEVELOPMENT_TRACES[0], ablation=variant)
        e.prepare(True)
        j = next(iter(e.tasks.values()))
        e.previous_forecast[:] = 12345.
        e.pred_error.append(np.array([3., 4., 5.]))
        context, glob, *_ = e.observations()
        if variant == 'NoForecastFeatures':
            assert (context[:e.u, 4:6] == 0).all() and (glob[28:34] == 0).all()
            j.report = np.asarray(c['region_centers_m'][j.region], float)
            assert (e.route_features(j)[0][57:62] == 0).all()
        if variant == 'NoWait':
            assert not e.wait_information([j])['allow']
    # Greedy with/without GeoI uses the exact same controller class.
    assert type(Greedy(settings('Greedy-NoPrivacy')[0])) is type(Greedy(settings('Greedy-GeoI')[0]))
    return dict(environment_checks_passed=True, rule_scenes=results,
        all_seven_populations_checked=True, matched_input_groups=len(signatures),
        interpretation='80 admission slots plus full drainage: correctness only, NOT a scale-up performance result')


def gradient_checks():
    bootstrap()
    from experiments.lr_screen.runtime import np, torch, seed_all
    from pressure9.policies import learner, update
    from pressure9.runner import model_hash
    from review_environment import environment_types
    Plain, TM, _, _ = environment_types()
    torch.set_num_threads(1)
    cases = [(m, None) for m in LEARNERS] + [('TMAPO', v) for v in ABLATIONS]
    rows = []
    for method, variant in cases:
        c, t = settings(method, ablation=variant)
        seed_all(DEVELOPMENT_SEEDS[0])
        e = TM(c, seed=DEVELOPMENT_TRACES[0], ablation=variant) if method == 'TMAPO' else Plain(c, family=method, seed=DEVELOPMENT_TRACES[0])
        p = learner(method, t, c)
        def hashes():
            modules = (p.actors, p.critic) if hasattr(p, 'actors') else (p.actor, p.q)
            import hashlib
            return tuple(hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in module.parameters())).hexdigest() for module in modules)
        before = hashes()
        history, update_rows = [], []
        for slot in range(503):
            reward, records, *_ = e.step(p, arrivals=slot < 300)
            done = e.slot >= 300 and not e.tasks
            history.append(float(reward))
            p.remember(e.agent_rewards(t['local_credit_fraction']), records, done)
            if e.slot % 100 == 0 or done:
                context, glob, *_ = e.observations()
                stats = update(p, context, glob)
                assert all(math.isfinite(float(v)) for v in stats.values())
                update_rows.append(stats)
            if done:
                break
        else:
            raise AssertionError('Smoke training did not drain')
        assert 3 <= len(update_rows) <= 6
        after = hashes()
        assert all(a != b for a, b in zip(before, after)), 'Actor AND critic must really update'
        assert math.isclose(math.fsum(history), -e.cost_total / (e.u * c['arrival_per_user_s'] * c['dt_s']), rel_tol=1e-8)
        m = check_metrics(e)
        if method != 'TMAPO':
            assert m['class_gate_wait_actions'] == 0
        if variant == 'NoWait':
            assert m['class_gate_wait_actions'] == 0
        rows.append(dict(method=method, ablation=variant, actor_changed=True, critic_changed=True,
            updates=len(update_rows), env_steps=e.slot, input_hash=e.full_input_hash,
            pending=0, stats=update_rows))
        del p, e
        import gc
        gc.collect()
    return dict(real_update_checks_passed=True, cases=rows,
        interpretation='Short real-update regression, not trained performance or convergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gradients', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    result = environment_checks()
    if args.gradients:
        result.update(gradient_checks())
    result['wall_seconds'] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(environment_passed=True, rule_scenes=len(result['rule_scenes']),
        real_updates=bool(args.gradients), result_path=str(args.out))))


if __name__ == '__main__':
    main()

"""Source-bound training snapshots and independent fixed validation cohorts."""
import copy
import io
import math
from optimization2_protocol import (ADMISSION_STEPS, BLOCK, TOTAL_STEPS, CHECKPOINT_UPDATES,
    VALIDATION_PHASES, VALIDATION_POPULATIONS, VALIDATION_TRACES, VARIANTS, settings)
from review_checks import bootstrap, check_metrics


def runtime():
    bootstrap()
    from experiments.lr_screen.runtime import np, torch, seed_all, rng_state, rng_load, runtime_info
    from pressure9.policies import learner, weights_only, set_progress
    from review_environment import environment_types
    _, tm, _, _ = environment_types()
    torch.set_num_threads(1)
    return np, torch, seed_all, rng_state, rng_load, runtime_info, learner, weights_only, set_progress, tm


def make(variant, seed, n=150):
    np, torch, seed_all, rng_state, rng_load, info, learner, weights, progress, env_type = runtime()
    c, t = settings(variant, seed, n)
    seed_all(seed)
    e = env_type(c, seed=1191900 + seed,
        ablation=None if VARIANTS[variant][0] else 'NoForecastFeatures')
    p = learner('TMAPO', t, c)
    return e, p


def train_block(e, p):
    from experiments.lr_screen.runtime import torch
    from pressure9.policies import set_progress
    update = e.slot // BLOCK + 1
    set_progress(p, update, 1208)
    total = 0.
    for _ in range(BLOCK):
        reward, records, *_ = e.step(p, arrivals=e.slot < ADMISSION_STEPS)
        assert math.isfinite(float(reward))
        total += float(reward)
        p.remember(e.agent_rewards(p.cfg['local_credit_fraction']), records, e.slot == TOTAL_STEPS)
    context, glob, *_ = e.observations()
    stats = p.update(context, glob)
    assert all(math.isfinite(float(v)) for v in stats.values())
    assert all(torch.isfinite(v).all() for net in (p.actors, p.critic) for v in net.parameters())
    return dict(update=update, env_steps=e.slot, block_return=total, **stats)


def snapshot(e, p, identity, rows, validation):
    from experiments.lr_screen.runtime import rng_state
    assert e.slot % BLOCK == 0 and not p.buffer
    return copy.deepcopy(dict(identity=identity, environment=e.__dict__, learner=p.state(),
        rng=rng_state(), training_rows=rows, validation_rows=validation))


def restore(saved, variant, seed, identity, n=150):
    from experiments.lr_screen.runtime import rng_load
    assert saved['identity'] == identity
    e, p = make(variant, seed, n=n)
    e.__dict__.update(copy.deepcopy(saved['environment']))
    p.load(saved['learner'])
    rng_load(saved['rng'])
    assert e.slot % BLOCK == 0 and len(saved['training_rows']) == e.slot // BLOCK
    return e, p, copy.deepcopy(saved['training_rows']), copy.deepcopy(saved['validation_rows'])


def validate(p, variant, seed, update, saved_rows=(), deadline=None):
    import time
    np, torch, seed_all, rng_state, rng_load, info, learner, weights, progress, env_type = runtime()
    before = rng_state()
    result = list(saved_rows)
    done = {(r['users'],r['trace_seed'],r['phase_s']) for r in result}
    try:
        for n in VALIDATION_POPULATIONS:
            c, t = settings(variant, seed, n)
            policy = learner('TMAPO', t, c)
            policy.load(weights(p))
            policy.deterministic = True
            for trace in VALIDATION_TRACES:
                for phase in VALIDATION_PHASES:
                    if (n,trace,phase) in done:
                        continue
                    if deadline is not None and time.monotonic() >= deadline:
                        return result
                    env = env_type(c, seed=trace, phase_s=phase,
                        ablation=None if VARIANTS[variant][0] else 'NoForecastFeatures',
                        measure_start=5., measure_end=15.)
                    with torch.no_grad():
                        for slot in range(1703):
                            env.step(policy, arrivals=slot < 1500)
                            if env.slot >= 1500 and not env.tasks:
                                break
                        else:
                            raise AssertionError('Validation did not drain')
                    m = check_metrics(env)
                    result.append(dict(update=update, users=n, trace_seed=trace, phase_s=phase,
                        warmup_s=5., measure_s=10., full_input_hash=env.full_input_hash, **m))
    finally:
        rng_load(before)
    assert len(result) == 8
    return result


def serialize(value):
    from experiments.lr_screen.runtime import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def deserialize(raw):
    from experiments.lr_screen.runtime import torch
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)


def gate():
    from optimization2_audit import fingerprint
    from review_protocol import DEVELOPMENT_SEEDS
    rows = []
    for v in VARIANTS:
        e, p = make(v, DEVELOPMENT_SEEDS[0], n=20)
        before = fingerprint((p.actors.state_dict(), p.critic.state_dict()))
        first = train_block(e, p)
        state = snapshot(e, p, {'component': v}, [first], [])
        raw = serialize(state)
        second = train_block(e, p)
        expected = fingerprint((e.__dict__, p.state()))
        saved = deserialize(raw)
        clone, policy, _, _ = restore(saved, v, DEVELOPMENT_SEEDS[0], {'component':v}, n=20)
        train_block(clone, policy)
        assert fingerprint((clone.__dict__, policy.state())) == expected
        assert fingerprint((p.actors.state_dict(), p.critic.state_dict())) != before
        from experiments.lr_screen.runtime import rng_state
        import time
        before_validation = fingerprint(rng_state())
        assert validate(policy,v,DEVELOPMENT_SEEDS[0],0,deadline=time.monotonic()-1)==[]
        assert fingerprint(rng_state()) == before_validation
        rows.append(dict(variant=v, real_updates=2, environment_steps=200, exact_resume=True,
            validation_rng_isolation=True))
    return dict(passed=True, component_only=True, cases=rows, numerical_failure=False)


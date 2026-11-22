"""Train once in a hosted VM and evaluate the same policy on separate CPU processes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from multiprocessing import get_context
from types import SimpleNamespace

import fresh_baseline_learners as learners
import fresh_baseline_worker as frozen
from public_baseline_updates import make_controller
import seven_scale_protocol as spec


def controller(method):
    return learners.ClippedPPO(centralized=False) if method == "IPPO" else make_controller(method)


def setup():
    frozen.bootstrap()
    from experiments.lr_screen.runtime import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen.spec = frozen.transport.spec = frozen.accounting.spec = spec
    frozen.store.CAMPAIGN = spec.CAMPAIGN
    learners.controller = controller
    return torch


def initialize_eval(method, seed, model, weights):
    global EVAL
    for name in ("GH_TOKEN", "LR_SCREEN_TRANSPORT_KEY"):
        os.environ.pop(name, None)
    torch = setup()
    policy = None
    if seed:
        policy = controller(method)
        policy.training = False
        for net, state in zip((policy.actor, policy.critic), weights):
            net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
        assert frozen.parameter_hash(policy) == model
    plain, *_ = frozen.evaluation_environment_types()
    base = frozen.rule_environment(plain) if method.startswith("Greedy-") else plain
    env_type = frozen.with_latency_bins(frozen.accounting.with_deadline_sweep(base))
    EVAL = method, seed, policy, model, env_type


def evaluate_case(tier, n, trace, phase, limit):
    assert time.monotonic() < limit, "Evaluation wall budget"
    warm, measure = (spec.PREVIEW_WARM, spec.PREVIEW_MEASURE) if tier == "preview" else (
        spec.FULL_WARM, spec.FULL_MEASURE)
    frozen.spec = SimpleNamespace(settings=spec.settings, WARM=warm, MEASURE=measure)
    start = time.monotonic()
    row = frozen.evaluate(*EVAL, n, trace, phase, limit)
    row["wall_s"] = time.monotonic() - start
    return row


def outcome(method, seed, tier, trace_index, train=None):
    return dict(method=method, seed=seed, tier=tier, trace_index=trace_index,
                completed=False, rows=[], training=train)


def save(t, name, result):
    result["rows"].sort(key=lambda r: (r["users"], r["trace_seed"]))
    t.small(name + ".enc", result, name)


def gather(t, pool, index, tier, training, limit):
    method, seed = spec.CELLS[index]
    is_full = tier == "full"
    traces = spec.FULL_TRACES if is_full else spec.PREVIEW_TRACES
    by_trace = {i: outcome(method, seed, tier, i, training) for i in range(len(traces))} if is_full else {
        None: outcome(method, seed, tier, None, training)}
    population_order = reversed(spec.POPS) if is_full else spec.POPS
    pending = {pool.submit(evaluate_case, tier, n, trace, phase, limit): (i if is_full else None, n, trace)
               for n in population_order for i, (trace, phase) in enumerate(traces)}
    for future in as_completed(pending):
        i, n, trace = pending[future]
        result = by_trace[i]
        try:
            row = future.result()
            assert (row["users"], row["trace_seed"]) == (n, trace)
            result["rows"].append(row)
        except Exception as exc:
            error = dict(users=n, trace_seed=trace, type=type(exc).__name__, message=str(exc))
            result.setdefault("errors", []).append(error)
            result["error"] = error
            print(f"{tier} N={n} trace={trace} {type(exc).__name__}", flush=True)
            continue
        count = len(result["rows"])
        print(f"{tier}: completed N={n} trace={trace}", flush=True)
        if is_full and count in spec.PART_ENDS:
            save(t, spec.receipt_name(index, tier, i, part=count), result)
    for i, result in by_trace.items():
        result["completed"] = not result.get("errors") and len(result["rows"]) == (
            len(spec.POPS) * (1 if is_full else len(traces)))
        save(t, spec.receipt_name(index, tier, i), result)
    return all(result["completed"] for result in by_trace.values())


def run(t, index):
    assert frozen.store.within_dispatch_window(t)
    method, seed = spec.CELLS[index]
    preview_limit = time.monotonic() + spec.PREVIEW_WALL
    policy, training = (None, None)
    if seed:
        try:
            policy, training = frozen.train(method, seed, frozen.evaluation_environment_types()[0], preview_limit)
        except Exception as exc:
            result = outcome(method, seed, "preview", None)
            result["error"] = dict(type=type(exc).__name__, message=str(exc))
            save(t, spec.receipt_name(index, "preview"), result)
            raise
    model = training["model_hash"] if training else None
    weights = tuple({k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
                    for net in (policy.actor, policy.critic)) if policy else None
    with ProcessPoolExecutor(max_workers=spec.EVAL_CPUS, mp_context=get_context("spawn"),
            initializer=initialize_eval, initargs=(method, seed, model, weights)) as pool:
        if not gather(t, pool, index, "preview", training, preview_limit):
            return False
        return gather(t, pool, index, "full", training, time.monotonic() + spec.FULL_WALL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cell", type=int, choices=range(len(spec.CELLS)))
    a = p.parse_args()
    setup()
    raise SystemExit(0 if run(frozen.transport.Transport(), a.cell) else 2)

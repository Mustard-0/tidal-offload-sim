"""Transactional, source/runtime-locked runner. Formal training stays gated.

Component tests are explicitly marked; formal runs require a source-locked
successful preflight for the reviewed transfer integration.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

from .runtime import (np, torch, ROOT, seed_all, rng_state, rng_load, json_write, csv_write,
                      read_csv, runtime_info, source_hashes, checkpoint_write, available_ram_gib)
from .protocol import configuration, plan, digest, LRS
from .readiness import require_transfer
from .algorithms import learner_for
from .restricted_environment import CandidateRestrictedEnvironment


def new_environment(cfg, seed, episode, episode_steps):
    phase = ((episode % 5)+.5)*cfg["tide_period_s"]/5
    episode_seed = int(np.random.SeedSequence([seed, 901, episode]).generate_state(1)[0])
    env = CandidateRestrictedEnvironment(cfg, seed=episode_seed, phase_s=phase)
    env.arrival_stop_slot = episode_steps-math.ceil(cfg["lifetime_s"]/cfg["dt_s"])-1
    return env


def audit_tasks(env, metrics):
    c = env.cfg
    if len(env.rows) != metrics["arrivals"] or len({r["task_id"] for r in env.rows}) != len(env.rows):
        raise AssertionError("Missing or duplicate ended task")
    costs = []
    for r in env.rows:
        energy = sum(r[k] for k in ("transmit_energy_j", "compute_energy_j", "report_energy_j"))
        if not math.isclose(energy, r["energy_j"], abs_tol=1e-9):
            raise AssertionError("Inconsistent energy units/components")
        value = c["weight_time"]*(r["end_s"]-r["arrival_s"])/c["time_ref_s"]
        value += c["weight_energy"]*energy/c["energy_ref_j"]+c["weight_violation"]*r["qos"]
        if r["published"]:
            value += c["weight_quality"]*r["ql_m"]/c["quality_ref_m"]
        costs.append(value)
    if not math.isclose(float(np.mean(costs)), metrics["mean_task_cost"], abs_tol=1e-8):
        raise AssertionError("Raw per-task objective differs from metric")
    if metrics["arrivals"] != metrics["completed"]+metrics["failed"]:
        raise AssertionError("Task/queue conservation failure")
    for batch in env.batch_rows:
        ids = batch["user_ids"].split(";")
        if len(ids) != batch["n_distinct"] or len(set(ids)) != len(ids):
            raise AssertionError("Wrong actual distinct batch n")
    return costs


def evaluate(policy, cfg, seed, phase, warmup, measure):
    env = CandidateRestrictedEnvironment(cfg, seed=seed, phase_s=phase,
        measure_start=warmup*cfg["dt_s"], measure_end=(warmup+measure)*cfg["dt_s"], collect=True)
    env.arrival_stop_slot = warmup+measure
    limit = warmup+measure+math.ceil(cfg["lifetime_s"]/cfg["dt_s"])+3
    for slot in range(limit):
        env.step(policy, arrivals=slot < warmup+measure)
        if slot >= warmup+measure and not env.tasks:
            break
    else:
        raise AssertionError("Evaluation cohort did not drain by lifetime")
    metrics = env.metrics()
    audit_tasks(env, metrics)
    metrics["max_active_wait_s"] = env.max_observed_wait
    metrics["objective_time"] = cfg["weight_time"]*metrics["resolution_time_s_per_arrival"]/cfg["time_ref_s"]
    metrics["objective_energy"] = cfg["weight_energy"]*metrics["energy_j_per_arrival"]/cfg["energy_ref_j"]
    metrics["objective_quality"] = cfg["weight_quality"]*env.sum_ql/metrics["arrivals"]/cfg["quality_ref_m"]
    metrics["objective_qos"] = cfg["weight_violation"]*metrics["qos_rate"]
    return metrics, env


def pooled(rows, key):
    denominator = "completed" if key == "delay_s" else "published" if key in (
        "ql_m", "report_weighted_batch_n", "amplified_publication_ratio") else "arrivals"
    valid = [r for r in rows if r.get(key) is not None]
    n = sum(r[denominator] for r in valid)
    return sum(r[key]*r[denominator] for r in valid)/n if n else None


def validate(method, learner, cfg, training, out, update, device):
    # A new learner AND complete RNG restoration isolate validation completely.
    state = rng_state()
    rows, tasks, batches, waits = [], [], [], []
    try:
        clone = learner_for(method, training, cfg, device)
        clone.load(copy.deepcopy(learner.state(for_evaluation=True)))
        clone.deterministic = True
        for seed in training["validation_seeds"]:
            for phase in training["validation_phases"]:
                m, env = evaluate(clone, cfg, seed, phase,
                    round(training["validation_warmup_s"]/cfg["dt_s"]),
                    round(training["validation_measure_s"]/cfg["dt_s"]))
                tag = dict(update=update, env_steps=update*training["rollout_steps"],
                           method=method, validation_seed=seed, phase_s=phase)
                rows.append(dict(tag, **m))
                tasks.extend(dict(tag, **r) for r in env.rows)
                batches.extend(dict(tag, **r) for r in env.batch_rows)
                waits.extend(dict(tag, **r) for r in env.wait_decision_rows)
        for name, data in (("tasks", tasks), ("batches", batches), ("waits", waits)):
            csv_write(out/"validation_traces"/f"{name}_{update:06d}.csv", data)
        return rows
    finally:
        rng_load(state)


def pid_running(pid):
    if os.name == "nt":
        import ctypes
        api = ctypes.windll.kernel32
        api.OpenProcess.restype = ctypes.c_void_p
        handle = api.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_ulong()
            if not api.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            api.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def train_cell(out, method, lr, device="cpu", resume=False, max_seconds=math.inf,
               stop_after=None, component_settings=None):
    if component_settings is None:
        require_transfer()
    if available_ram_gib() < 1.:
        raise RuntimeError("Less than 1GiB RAM available before worker startup")
    cfg, training = configuration(method, lr)
    if component_settings is not None:
        # Used by explicit tests only; evidence is marked and kept out of runs/.
        training.update(component_settings)
    if training["episode_steps"] <= math.ceil(cfg["lifetime_s"]/cfg["dt_s"])+2:
        raise ValueError("Episode too short to drain fixed-lifetime tasks")
    out = Path(out)
    if (out/"STOP_AFTER_UPDATE").exists() or (out.parent/"STOP_AFTER_UPDATE").exists():
        raise RuntimeError("Explicit stop marker present")
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.parent/("."+out.name+".lock")
    if lock.exists():
        import json
        previous = json.loads(lock.read_text())
        if not resume or pid_running(previous["pid"]):
            raise RuntimeError("Active or unreviewed worker lock")
        os.replace(lock, lock.with_name(lock.name+f".stale.{time.time_ns()}"))
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd, ('{"pid":%d}' % os.getpid()).encode())
        return _train(out, method, lr, device, resume, max_seconds, stop_after, cfg, training,
                      component_settings is not None)
    finally:
        os.close(fd)
        lock.unlink()


def _train(out, method, lr, device, resume, max_seconds, stop_after, cfg, training, component):
    import json
    seed_all(training["seed"])
    identity = dict(schema="lr-cell-20261025", method=method, lr=lr, environment=cfg, training=training,
                    runtime=runtime_info(device), source_hashes=source_hashes(), component_only=component)
    identity_hash = digest(identity)
    if not resume:
        out.mkdir(exist_ok=False)
        sha = subprocess.check_output(["git", "-c", "safe.directory="+ROOT.as_posix(),
                                       "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        json_write(out/"manifest.json", dict(identity, identity_hash=identity_hash, source_commit=sha,
                    test_set_used=False, transfer_integrated=True,
                    status="COMPONENT_ONLY" if component else "formal"))
        for rel in identity["source_hashes"]:
            dest = out/"source"/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/rel).read_bytes())
    else:
        manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        if manifest["identity_hash"] != identity_hash:
            raise ValueError("Resume requires identical source, config and runtime/device")
        if (out/"complete.json").exists():
            raise ValueError("Already complete; refusing duplicate training")
    learner = learner_for(method, training, cfg, device)
    env = new_environment(cfg, training["seed"], 0, training["episode_steps"])
    rows, valrows, episode, completed, selected, best = [], [], 0, 0, None, math.inf
    if resume:
        saved = torch.load(out/"latest.pt", map_location=device, weights_only=False)
        if saved["identity_hash"] != identity_hash:
            raise ValueError("Checkpoint identity mismatch")
        for name, stored in (("train.csv", saved["rows"]), ("validation.csv", saved["valrows"])):
            if (out/name).exists():
                raw = read_csv(out/name)
                # A transaction can replace latest.pt just before CSV flushing;
                # all present prefix rows must still match exactly.
                for actual, reference in zip(raw, stored):
                    if any(actual.get(k, "") != ("" if reference.get(k) is None else str(reference[k]))
                           for k in set(actual) | set(reference)):
                        raise ValueError("Raw CSV was modified; refusing unsafe resume")
        # Preserve any post-checkpoint CSV/failure evidence before deterministic
        # replay. Never silently edit, merge or relabel the abandoned attempt.
        recovery = out/"recovery"/str(time.time_ns())
        recovery.mkdir(parents=True)
        for name in ("train.csv", "validation.csv", "status.json", "failure.json", "paused.json"):
            if (out/name).exists():
                (recovery/name).write_bytes((out/name).read_bytes())
        learner.load(saved["learner"])
        rng_load(saved["rng"])
        env, episode, completed = saved["environment"], saved["episode"], saved["update"]
        rows, valrows = saved["rows"], saved["valrows"]
        selected, best = saved["selected"], saved["best"]
        # best weights are part of the same transaction, not a future CSV tail.
        checkpoint_write(out/"best.pt", saved["best_policy"])
        csv_write(out/"train.csv", rows)
        csv_write(out/"validation.csv", valrows)
    best_policy = copy.deepcopy(learner.state(for_evaluation=True)) if not resume else saved["best_policy"]
    started = time.monotonic()
    previous_wall = rows[-1]["wall_s"] if rows else 0.

    def validation(update):
        nonlocal best, selected, best_policy
        current = validate(method, learner, cfg, training, out, update, device)
        score = pooled(current, "mean_task_cost")
        valrows.extend(current)
        if not math.isfinite(score):
            raise FloatingPointError("Invalid validation objective")
        if score < best:
            best, selected = score, update
            best_policy = copy.deepcopy(learner.state(for_evaluation=True))
            checkpoint_write(out/"best.pt", best_policy)
        csv_write(out/"validation.csv", valrows)
        print("VALIDATION", method, lr, update, score, "best", selected, flush=True)

    def checkpoint(update):
        checkpoint_write(out/"latest.pt", dict(identity_hash=identity_hash, learner=learner.state(),
            environment=env, episode=episode, update=update, rows=rows, valrows=valrows,
            rng=rng_state(), selected=selected, best=best, best_policy=best_policy))
        json_write(out/"checkpoint_status.json", dict(update=update, env_steps=update*training["rollout_steps"]))

    try:
        if not resume:
            validation(0)
            checkpoint(0)
        for update in range(completed+1, training["updates"]+1):
            tick = time.monotonic()
            learner.deterministic = False
            learner.set_progress(update, training["updates"])
            rewards = []
            for _ in range(training["rollout_steps"]):
                if env.slot == training["episode_steps"]:
                    if env.tasks:
                        raise AssertionError("Unresolved task cost at terminal boundary")
                    episode += 1
                    env = new_environment(cfg, training["seed"], episode, training["episode_steps"])
                reward, records, _, _ = env.step(learner, arrivals=env.slot < env.arrival_stop_slot)
                learner.remember(env.agent_rewards(0.), records, env.slot == training["episode_steps"])
                rewards.append(reward)
            ctx, glob, *_ = env.observations()
            stats = learner.update(ctx, glob)
            if not np.isfinite(rewards).all() or not all(np.isfinite(v) for v in stats.values()):
                raise FloatingPointError("Nonfinite return/update diagnostics")
            if any(not torch.isfinite(p).all() for module in (learner.actors, learner.critic) for p in module.parameters()):
                raise FloatingPointError("Nonfinite model parameters")
            rows.append(dict(update=update, env_steps=update*training["rollout_steps"], episode=episode,
                mean_training_reward=float(np.mean(rewards)), raw_training_reward_sum=float(np.sum(rewards)),
                actor_lr=learner.aopt.param_groups[0]["lr"], critic_lr=learner.copt.param_groups[0]["lr"],
                train_trace_hash=env.arrival_hash, arrivals=env.arrivals, ended=env.ended,
                cost_total=env.cost_total, max_active_wait_s=env.max_observed_wait,
                update_wall_s=time.monotonic()-tick, wall_s=previous_wall+time.monotonic()-started, **stats))
            # Persist raw scalars on EVERY update, not only at process exit.
            csv_write(out/"train.csv", rows)
            json_write(out/"status.json", dict(update=update, env_steps=update*training["rollout_steps"],
                state="running", component_only=component, best_update=selected, best_validation_cost=best, pid=os.getpid()))
            if update % training["validation_interval"] == 0 or update == training["updates"]:
                validation(update)
            reason = ("available RAM below .5GiB" if available_ram_gib() < .5
                      else "stop marker" if (out/"STOP_AFTER_UPDATE").exists() or (out.parent/"STOP_AFTER_UPDATE").exists()
                      else "wall budget" if time.monotonic()-started >= max_seconds
                      else "component test boundary" if stop_after is not None and update >= stop_after else None)
            if reason or update % training["checkpoint_interval"] == 0 or update % training["validation_interval"] == 0 or update == training["updates"]:
                checkpoint(update)
            if reason and update < training["updates"]:
                json_write(out/"paused.json", dict(update=update, reason=reason))
                return False
            print("UPDATE", method, lr, update, "env_steps", update*training["rollout_steps"], flush=True)
        json_write(out/"selection.json", dict(update=selected, validation_cost=best,
                    learned_beyond_initialization=selected != 0, test_set_used=False))
        json_write(out/"complete.json", dict(updates=training["updates"],
                    env_steps=training["updates"]*training["rollout_steps"], component_only=component))
        for name in ("paused.json", "failure.json"):
            if (out/name).exists():
                (out/name).unlink()
        return True
    except BaseException as exc:
        # Keep last valid optimizer/RNG/environment transaction, never save a
        # partially updated/possibly nonfinite learner as a valid checkpoint.
        json_write(out/"failure.json", dict(type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(), last_logged_update=rows[-1]["update"] if rows else 0,
            recover_from="latest.pt", component_only=component))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "report"))
    p.add_argument("--out", type=Path, default=ROOT/"runs/lr_screen_20261025")
    p.add_argument("--method", choices=tuple(LRS))
    p.add_argument("--lr", type=int)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=float, default=3600.)
    a = p.parse_args()
    if a.command == "plan":
        if (a.out/"plan.json").exists():
            raise FileExistsError("Keep the existing draft/frozen plan; use a new output path")
        json_write(a.out/"plan.json", plan())
        csv_write(a.out/"status.csv", plan()["cells"])
        print(a.out/"plan.json")
    elif a.command == "report":
        from .report import report
        report(a.out)
    else:
        require_transfer()  # Deliberately BEFORE making any training directory.
        if a.method is None or a.lr is None:
            p.error("run requires --method and --lr")
        train_cell(a.out/f"{a.method}_LR{a.lr}_seed87021", a.method, a.lr,
                   a.device, a.resume, a.max_seconds)


if __name__ == "__main__":
    main()

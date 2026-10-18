"""Study 1: bounded pilots and explicitly authorized SINGLE formal cells.

No GitHub dispatch, test-set tuning, or implicit 18-cell long-training matrix.
Run from the repository root; output directories are immutable per run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import gc
import hashlib
import json
import importlib.metadata
import math
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "experiments")]
from paper_four_studies import (full_config, LEARNING_RATES, TRAIN_SEEDS,
                                VALIDATION_SEEDS, TEST_SEEDS, paper_plan)
from cross_slot_environment import CrossSlotWLOEnvironment
from cross_slot_learners import CrossSlotPPO, CrossSlotTD3
from cross_slot_wlo_train import CrossSlotRule
from atomic_io import replace_file

METHODS = ("TMPPO", "IPPO", "MARATD3-Adapted")
PHASES = (30., 130., 230.)
SOURCE_FILES = tuple(sorted(set(
    list((ROOT / "model").glob("*.py")) +
    list((ROOT / "experiments").glob("*.py")) +
    [ROOT / "config.json", ROOT / "experiments/PAPER_FOUR_STUDIES_PROTOCOL.md"] +
    list((ROOT/"experiments").glob("LONG_PROBE_2000_PROTOCOL.md")) +
    list((ROOT/"experiments").glob("LEARNING_CONTRACT_PROTOCOL.md")))))


def free_ram_gib():
    if os.name != "nt":
        try:
            fields = dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
            return int(fields["MemAvailable"].split()[0])*1024/2**30
        except (OSError,KeyError):
            return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length",ctypes.c_ulong),("load",ctypes.c_ulong)] + [
            (key,ctypes.c_ulonglong) for key in
            ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","extended")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot read available system memory")
    return status.avail_phys/2**30


def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    replace_file(temp, path)


def csv_write(path, rows):
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    replace_file(temp, path)


def source_hashes():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(source_bytes(p)).hexdigest()
            for p in SOURCE_FILES}


def source_bytes(path):
    # Source identity is independent of Git's Windows checkout line endings.
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def learner_for(method, train, cfg):
    if method == "MARATD3-Adapted":
        return CrossSlotTD3(method, train, cfg, device="cpu")
    return CrossSlotPPO("TMAPO" if method == "TMPPO" else "IPPO", train, cfg, device="cpu")


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def rng_load(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def validation_mean(rows, key, objective="equal_scenario_cost"):
    valid = [r for r in rows if r.get(key) not in (None, "")]
    if not valid:
        return None
    if objective == "equal_scenario_cost":
        return float(np.mean([float(r[key]) for r in valid]))
    if objective != "pooled_task_cost":
        raise ValueError("Unknown validation objective")
    denominator = ("completed" if key == "delay_s" else "published" if key in
                   ("ql_m", "report_weighted_batch_n", "amplified_publication_ratio") else "arrivals")
    weights = [int(r[denominator]) for r in valid]
    total = sum(weights)
    return sum(float(r[key])*w for r, w in zip(valid, weights))/total if total else None


def evaluate(policy, cfg, seed, phase, measure_steps, warmup_steps, collect=False,
             deterministic=True):
    env = CrossSlotWLOEnvironment(cfg, seed=seed, phase_s=phase,
                                 measure_start=warmup_steps * cfg["dt_s"],
                                 measure_end=(warmup_steps + measure_steps) * cfg["dt_s"],
                                 collect=collect)
    if getattr(policy, "is_rule_based", False):
        policy.env = env
    else:
        policy.reset_history()
        policy.deterministic = deterministic
        policy.sample_evaluation = not deterministic
    end = warmup_steps + measure_steps
    limit = end + math.ceil(cfg["lifetime_s"] / cfg["dt_s"]) + 5
    for slot in range(limit):
        env.step(policy, arrivals=slot < end)
        if slot >= end and not env.tasks:
            break
    else:
        raise RuntimeError("Evaluation cohort did not drain")
    metrics = env.metrics()
    if metrics["arrivals"] != metrics["completed"] + metrics["failed"]:
        raise AssertionError("Task conservation failure")
    if collect:
        audit_tasks(env, metrics)
        count = len(env.rows)
        if count:
            metrics.update(
                objective_time=sum(cfg["weight_time"] * (r["end_s"]-r["arrival_s"]) /
                                   cfg["time_ref_s"] for r in env.rows) / count,
                objective_energy=cfg["weight_energy"] * metrics["energy_j_per_arrival"] / cfg["energy_ref_j"],
                objective_quality=sum(cfg["weight_quality"] * float(r["ql_m"]) /
                                      cfg["quality_ref_m"] for r in env.rows if r["published"]) / count,
                objective_qos=cfg["weight_violation"] * metrics["qos_rate"])
    metrics["validation_reward"] = -metrics["mean_task_cost"]
    return metrics, env


def audit_tasks(env, metrics):
    rows = env.rows
    if len(rows) != metrics["arrivals"] or len({r["task_id"] for r in rows}) != len(rows):
        raise AssertionError("Missing/duplicate raw task rows")
    c = env.cfg
    costs = []
    for r in rows:
        energy = sum(r[k] for k in ("transmit_energy_j", "compute_energy_j", "report_energy_j"))
        if not math.isclose(energy, r["energy_j"], abs_tol=1e-9):
            raise AssertionError("Energy components do not sum")
        cost = c["weight_time"] * (r["end_s"]-r["arrival_s"]) / c["time_ref_s"]
        cost += c["weight_energy"] * energy / c["energy_ref_j"] + c["weight_violation"] * r["qos"]
        if r["published"]:
            cost += c["weight_quality"] * r["ql_m"] / c["quality_ref_m"]
        costs.append(cost)
    if not math.isclose(float(np.mean(costs)), metrics["mean_task_cost"], abs_tol=1e-8):
        raise AssertionError("Raw task cost differs from reported cost")
    if not math.isclose(sum(r["qos"] for r in rows)/len(rows), metrics["qos_rate"], abs_tol=1e-12):
        raise AssertionError("Raw QoS differs from reported QoS")


def validation_snapshot(method, learner, train, cfg, settings, update, out, baseline=False):
    # New policy objects AND RNG restoration: evaluation cannot alter training
    # buffers, pending transitions, normalizer, recurrent history or randomness.
    state = rng_state()
    rows, tasks, batches = [], [], []
    try:
        clone = learner_for(method, train, cfg)
        clone.load(copy.deepcopy(learner.state(for_evaluation=True)))
        policies = [(method, clone)]
        if baseline:
            policies += [(name, CrossSlotRule(rule, cfg)) for name, rule in
                         (("Immediate-Remote", "immediate_remote"),
                          ("Local-Only", "strict_local_only"),
                          ("Forecast-Release", "forecast_release"))]
        for seed in settings["validation_seeds"]:
            for phase in settings["validation_phases"]:
                hashes = set()
                for label, policy in policies:
                    m, env = evaluate(policy, cfg, seed, phase, settings["validation_steps"],
                                      settings["warmup_steps"], collect=True)
                    tag = dict(update=update, env_steps=update*settings["rollout"],
                               method=label, seed=seed, phase_s=phase)
                    rows.append(dict(tag, **m))
                    tasks.extend(dict(tag, **r) for r in env.rows)
                    batches.extend(dict(tag, **r) for r in env.batch_rows)
                    hashes.add(m["trace_hash"])
                if len(hashes) != 1:
                    raise AssertionError("Unpaired validation arrivals")
        # This diagnostic never participates in checkpoint selection. Reset
        # action RNG for each trace; restore training RNG in the outer finally.
        if settings.get("policy_diagnostics", False) and (
                update in (0, settings["updates"]) or
                (settings.get("training_profile") == "credit_v3" and
                 not settings.get("diagnostics_endpoints_only", False))):
            for seed in settings["validation_seeds"]:
                for phase in settings["validation_phases"]:
                    action_seed = int(np.random.SeedSequence(
                        [seed, 317, int(phase * 1000)]).generate_state(1)[0])
                    random.seed(action_seed)
                    np.random.seed(action_seed)
                    torch.manual_seed(action_seed)
                    m, env = evaluate(clone, cfg, seed, phase, settings["validation_steps"],
                                      settings["warmup_steps"], collect=True, deterministic=False)
                    tag = dict(update=update, env_steps=update*settings["rollout"],
                               method=method + ":sampled-diagnostic", seed=seed, phase_s=phase)
                    rows.append(dict(tag, **m))
                    tasks.extend(dict(tag, **r) for r in env.rows)
                    batches.extend(dict(tag, **r) for r in env.batch_rows)
        trace_dir = out / "validation_traces"
        trace_dir.mkdir(exist_ok=True)
        csv_write(trace_dir / ("tasks_%06d.csv" % update), tasks)
        csv_write(trace_dir / ("batches_%06d.csv" % update), batches)
        return rows
    finally:
        rng_load(state)


def configuration(settings):
    cfg, train = full_config()
    # Objective, tasks, physics and privacy remain as declared in the protocol.
    # Only a common causal policy initialization and training numerics change.
    train.update(actor_cost_prior_scale=2., value_normalization=True, gamma=1., gae_lambda=.98,
                 rollout_steps=settings["rollout"], td3_start_steps=128,
                 td3_replay_slots=4096, td3_batch=16, td3_update_every=8)
    train["actor_lr"], train["critic_lr"] = LEARNING_RATES[settings["lr"]-1]
    profile = settings.get("training_profile", "audit_v1")
    if profile in ("team_v2", "credit_v3", "contract_v4", "shared_value_v5", "role_credit_v6"):
        # Conserving mean local rewards does not preserve the gradient of the
        # cooperative objective when a decision changes other users' queues.
        train.update(local_credit_fraction=0., linear_lr_decay=True,
                     lr_final_fraction=.1, td3_huber=True,
                     td3_discrete_objective="categorical_expectation")
        if profile in ("credit_v3", "contract_v4", "shared_value_v5", "role_credit_v6"):
            train.update(critic_clock_features=2, actor_warmup_steps=2048)
        if profile == "contract_v4":
            variants = {"control": (.98, .01), "trace1": (1., .01),
                        "noentropy": (.98, 0.), "trace1_noentropy": (1., 0.)}
            variant = settings.get("diagnostic_variant")
            if variant not in variants:
                raise ValueError("Unknown predeclared diagnostic variant")
            train["gae_lambda"], train["entropy"] = variants[variant]
            train["record_advantage_diagnostics"] = True
        if profile == "shared_value_v5":
            variant = settings.get("diagnostic_variant")
            if settings["method"] != "TMPPO" or variant not in ("team_value_control", "team_value_shared"):
                raise ValueError("Unregistered shared-team-value diagnostic")
            train.update(gae_lambda=1., entropy=0., record_advantage_diagnostics=True,
                         shared_team_value=variant == "team_value_shared")
        if profile == "role_credit_v6":
            frozen = {"role_joint": [], "role_route_only": ["user"],
                      "role_wlo_only": ["route"]}
            variant = settings.get("diagnostic_variant")
            if settings["method"] != "TMPPO" or variant not in frozen:
                raise ValueError("Unregistered actor-component training diagnostic")
            train.update(gae_lambda=1., entropy=0., record_advantage_diagnostics=True,
                         shared_team_value=True, frozen_actor_stages=frozen[variant])
    elif profile != "audit_v1":
        raise ValueError("Unknown training profile")
    return cfg, train


def new_environment(cfg, seed, episode, phase_count):
    phase = ((episode % phase_count) + .5) * cfg["tide_period_s"] / phase_count
    episode_seed = int(np.random.SeedSequence([seed, 901, episode]).generate_state(1)[0])
    return CrossSlotWLOEnvironment(cfg, seed=episode_seed, phase_s=phase)


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    replace_file(temp, path)


def train_cell(out, settings, resume=False, wall_seconds=600., stop_after=None):
    out = Path(out)
    if (out/"STOP_AFTER_UPDATE").exists() or (out.parent/"STOP_AFTER_UPDATE").exists():
        raise ValueError("Inspect and archive the explicit stop marker before starting a worker")
    if not resume and out.exists():
        raise FileExistsError("Refusing to overwrite an existing cell")
    if resume and not (out/"paused.json").exists():
        raise ValueError("Resume requires a saved paused boundary, not an active/unknown run")
    available = free_ram_gib()
    if available is not None and available < 1.:
        raise ValueError("Less than 1GiB available before worker startup; existing data left untouched")
    out.parent.mkdir(parents=True,exist_ok=True)
    lock = out.parent/("."+out.name+".worker.lock")
    descriptor = os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,json.dumps(dict(pid=os.getpid(),cell=str(out))).encode("utf-8"))
        return _train_cell(out,settings,resume,wall_seconds,stop_after)
    finally:
        os.close(descriptor)
        lock.unlink()


def _train_cell(out, settings, resume=False, wall_seconds=600., stop_after=None):
    out = Path(out)
    cfg, train = configuration(settings)
    if settings["episode_steps"] <= math.ceil(cfg["lifetime_s"]/cfg["dt_s"]) + 2:
        raise ValueError("Episode must leave room to drain the fixed-lifetime task cohort")
    hashes = source_hashes()
    if resume:
        manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        if manifest["settings"] != settings or manifest["source_hashes"] != hashes:
            raise ValueError("Resume settings/source hash mismatch")
        if (out/"complete.json").exists() or (out/"failure.json").exists():
            raise ValueError("Do not resume completed/failed cells")
    else:
        out.mkdir(parents=True, exist_ok=False)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest = dict(schema="paper-convergence-v1", source_commit=sha,
                        source_hashes=hashes, settings=settings, environment=cfg, training=train,
                        runtime=dict(python=sys.version, python_version=platform.python_version(),
                                     torch=torch.__version__,numpy=np.__version__,
                                     scipy=importlib.metadata.version("scipy"),
                                     matplotlib=importlib.metadata.version("matplotlib"),
                                     platform=platform.platform(), processor=platform.processor(),
                                     device="cpu",torch_threads=1),
                        methods=METHODS, main_modified=False, test_seeds_unused=list(TEST_SEEDS),
                        DP_status="exploratory one-batch accounting; no full-transcript DP proof",
                        adaptation=train.get("td3_discrete_objective", "straight_through") +
                                   "; recurrent/attention discrete adaptation, not exact MARATD3",
                        source_hash_format="SHA256 of LF-normalized source bytes",
                        checkpoint_rule="minimum " + settings.get("validation_objective", "equal_scenario_cost") +
                                        ", including update 0",
                        publication_grade=False, started_utc=datetime.now(timezone.utc).isoformat())
        json_write(out/"manifest.json", manifest)
        for rel in hashes:
            dest = out/"source"/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source_bytes(ROOT/rel))
    torch.set_num_threads(1)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    learner = learner_for(settings["method"], train, cfg)
    training, validation = [], []
    episode = 0
    sim = new_environment(cfg, settings["seed"], episode, settings["phase_count"])
    start_update, best, selected = 0, math.inf, None
    if resume:
        saved = torch.load(out/"latest.pt", map_location="cpu", weights_only=False)
        paused = json.loads((out/"paused.json").read_text(encoding="utf-8"))
        csv_tail = read_csv(out/"train.csv") if (out/"train.csv").exists() else []
        if paused["update"] != saved["update"] or (csv_tail and
                int(csv_tail[-1]["update"]) != saved["update"]):
            raise ValueError("Paused marker, raw CSV and latest checkpoint disagree")
        for raw, stored in ((csv_tail, saved["training"]),
                            (read_csv(out/"validation.csv"), saved["validation"])):
            if len(raw) != len(stored) or any(
                    row.get(key, "") != ("" if reference.get(key) is None else str(reference[key]))
                    for row, reference in zip(raw, stored) for key in set(row) | set(reference)):
                raise ValueError("Raw CSV differs from the saved paused checkpoint")
        if manifest["environment"] != cfg or manifest["training"] != train:
            raise ValueError("Expanded environment/training differs from saved manifest")
        learner.load(saved["learner"])
        rng_load(saved["rng"])
        sim, episode = saved["environment"], saved["episode"]
        training, validation = saved["training"], saved["validation"]
        start_update, best, selected = saved["update"], saved["best"], saved["selected"]
    started = time.monotonic()
    historical_wall = training[-1]["wall_s"] if training else 0.
    latest_checkpoint_update = start_update

    def validate(update):
        nonlocal best, selected
        validation_start = time.monotonic()
        rows = validation_snapshot(settings["method"], learner, train, cfg, settings, update, out,
                                   baseline=update==0 and settings.get("baseline_diagnostics", True))
        validation.extend(rows)
        score = validation_mean([r for r in rows if r["method"]==settings["method"]],
                                "mean_task_cost", settings.get("validation_objective", "equal_scenario_cost"))
        if score < best:
            best, selected = score, update
            save_checkpoint(out/"best_checkpoint.pt", copy.deepcopy(learner.state(for_evaluation=True)))
        print("VALIDATE", settings["method"], "LR", settings["lr"], "update", update,
              "cost", round(score, 6), "best", selected, flush=True)
        csv_write(out/"validation.csv", validation)
        with (out/"validation_timing.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(update=update,cohorts=len(rows),
                         wall_s=time.monotonic()-validation_start))+"\n")

    def checkpoint(update, durable=True):
        nonlocal latest_checkpoint_update
        checkpoint_started = time.monotonic()
        if durable:
            save_checkpoint(out/"latest.pt", dict(
                learner=learner.state(), environment=sim, episode=episode, rng=rng_state(),
                update=update, best=best, selected=selected, training=training, validation=validation))
            latest_checkpoint_update = update
        csv_write(out/"train.csv", training)
        json_write(out/"status.json", dict(update=update, env_steps=update*settings["rollout"],
                   latest_checkpoint_update=latest_checkpoint_update,
                   best_update=selected, best_validation_cost=best, pid=os.getpid()))
        with (out/"checkpoint_timing.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(update=update, durable=durable,
                wall_s=time.monotonic()-checkpoint_started,
                latest_bytes=(out/"latest.pt").stat().st_size if durable else None))+"\n")

    if not resume:
        validate(0)
        checkpoint(0)
    def set_clock():
        remaining = settings["episode_steps"] - sim.slot
        arrivals_remaining = max(0, remaining - math.ceil(cfg["lifetime_s"]/cfg["dt_s"]) - 1)
        learner.set_training_clock(remaining, arrivals_remaining, settings["episode_steps"])

    for update in range(start_update+1, settings["updates"]+1):
        tick = time.monotonic()
        rewards = []
        learner.deterministic = False
        learner.set_progress(update, settings.get("schedule_updates", settings["updates"]))
        for _ in range(settings["rollout"]):
            if sim.slot == settings["episode_steps"]:
                if sim.tasks:
                    raise AssertionError("Training episode truncated unresolved task costs")
                episode += 1
                learner.reset_history()
                sim = new_environment(cfg, settings["seed"], episode, settings["phase_count"])
            set_clock()
            arrivals = sim.slot < settings["episode_steps"] - math.ceil(cfg["lifetime_s"]/cfg["dt_s"])-1
            reward, records, _, _ = sim.step(learner, arrivals=arrivals)
            done = sim.slot == settings["episode_steps"]
            learner.remember(sim.agent_rewards(train.get("local_credit_fraction",.7)), records, done)
            rewards.append(reward)
        ctx, glob, *_ = sim.observations()
        set_clock()
        diag = learner.update(ctx, glob)
        if not np.isfinite(rewards).all() or any(not np.isfinite(v) for v in diag.values()):
            raise FloatingPointError("Nonfinite training diagnostics")
        ram = free_ram_gib()
        training.append(dict(update=update, env_steps=update*settings["rollout"], free_ram_gib=ram,
            mean_training_reward=float(np.mean(rewards)), episode=episode, phase_s=sim.phase_s,
            episode_slot=sim.slot, train_trace_hash=sim.arrival_hash,
            arrivals=sim.arrivals, ended=sim.ended, cost_total=sim.cost_total,
            update_wall_s=time.monotonic()-tick, wall_s=historical_wall+time.monotonic()-started,
            **{k:float(v) for k,v in diag.items()}))
        if update % settings["validate_every"] == 0 or update == settings["updates"]:
            validate(update)
        ram = free_ram_gib()
        if update == 1 or update % 8 == 0:
            print("TRAIN", settings["method"], "LR", settings["lr"], "update", update,
                  "steps", update*settings["rollout"], "seconds", round(time.monotonic()-started,1), flush=True)
        reason = ("STOP_AFTER_UPDATE" if (out/"STOP_AFTER_UPDATE").exists() or (out.parent/"STOP_AFTER_UPDATE").exists()
                  else "available RAM below .5GiB" if ram is not None and ram<.5
                  else "wall time budget" if time.monotonic()-started>=wall_seconds
                  else "requested test boundary" if stop_after is not None and update>=stop_after else None)
        # Full replay/environment serialization is much larger than scalar
        # progress. Always persist a safe boundary on pause/validation/finish.
        checkpoint(update, durable=bool(reason) or update == settings["updates"] or
                   update % settings["validate_every"] == 0 or
                   update % settings.get("checkpoint_every", 1) == 0)
        if reason and update < settings["updates"]:
            json_write(out/"paused.json", dict(update=update, reason=reason, free_ram_gib=ram,resumable=True))
            return False
    json_write(out/"selection.json", dict(update=selected, validation_cost=best,
               learned_beyond_initialization=selected!=0, test_used=False))
    json_write(out/"complete.json", dict(updates=settings["updates"],
               env_steps=settings["updates"]*settings["rollout"], best_update=selected,
               wall_s=historical_wall+time.monotonic()-started))
    if (out/"paused.json").exists():
        (out/"paused.json").unlink()
    return True


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    summaries, paired, seed_points, mode_points = [], {}, [], []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    completed = 0
    durations = []
    for cell in sorted(root.glob("*_LR*_seed*")):
        if not (cell/"validation.csv").exists():
            continue
        manifest = json.loads((cell/"manifest.json").read_text(encoding="utf-8"))
        s = manifest["settings"]
        val = read_csv(cell/"validation.csv")
        learned = [r for r in val if r["method"]==s["method"]]
        points = sorted({int(r["env_steps"]) for r in learned})
        objective = s.get("validation_objective", "equal_scenario_cost")
        series = {key:[validation_mean([r for r in learned if int(r["env_steps"])==x], key, objective)
                       for x in points] for key in ("validation_reward", "qos_rate", "mean_task_cost")}
        for index, point in enumerate(points):
            seed_points.append(dict(method=s["method"],lr=s["lr"],train_seed=s["seed"],
                                    env_steps=point,**{key:values[index] for key,values in series.items()}))
        for mode, label in (("argmax", s["method"]),
                            ("sampled", s["method"]+":sampled-diagnostic")):
            selected_mode = [r for r in val if r["method"] == label]
            for step in sorted({int(r["env_steps"]) for r in selected_mode}):
                group = [r for r in selected_mode if int(r["env_steps"]) == step]
                mode_points.append(dict(method=s["method"], lr=s["lr"], train_seed=s["seed"],
                    mode=mode, env_steps=step,
                    **{key:validation_mean(group,key,objective) for key in
                       ("mean_task_cost","qos_rate","objective_time","objective_energy",
                        "objective_quality","publication_ratio")}))
        for r in val:
            if r["method"] == s["method"]:
                key = (r["seed"],r["phase_s"],r["env_steps"])
                paired.setdefault(key,set()).add(r["trace_hash"])
        if (cell/"complete.json").exists():
            completed += 1
            complete = json.loads((cell/"complete.json").read_text())
            durations.append(dict(method=s["method"], wall_s=complete["wall_s"],
                                  steps=complete["env_steps"]))
            selection = json.loads((cell/"selection.json").read_text())
            selected = [r for r in learned if int(r["update"])==selection["update"]]
            row = dict(method=s["method"], lr=s["lr"], train_seed=s["seed"],
                       checkpoint_update=selection["update"], wall_s=complete["wall_s"])
            for key in ("mean_task_cost", "qos_rate", "delay_s", "energy_j_per_arrival", "ql_m",
                        "publication_ratio", "report_weighted_batch_n", "amplified_publication_ratio", "active_wait_s"):
                row[key] = validation_mean(selected, key, objective)
            row["initial_cost"] = series["mean_task_cost"][0]
            row["validation_improvement_percent"] = 100*(row["initial_cost"]-row["mean_task_cost"])/row["initial_cost"]
            row["tail_validation_cost_std"] = float(np.std(series["mean_task_cost"][-3:]))
            threshold = row["initial_cost"] - .95*(row["initial_cost"]-row["mean_task_cost"])
            row["observed_95percent_improvement_steps"] = (
                next((x for x,y in zip(points,series["mean_task_cost"]) if y<=threshold),None)
                if selection["update"] > 0 and row["validation_improvement_percent"] > 0 else None)
            train_rows = read_csv(cell/"train.csv")
            seconds_per_step = sum(float(r["update_wall_s"]) for r in train_rows)/complete["env_steps"]
            row["training_seconds_per_global_step"] = seconds_per_step
            row["formal_256000_training_steps_hours_excludes_validation_io"] = seconds_per_step*256000/3600
            summaries.append(row)
    aggregate = []
    for method in METHODS:
        for lr in (1,2):
            subset = [p for p in seed_points if p["method"]==method and p["lr"]==lr]
            xs = sorted({p["env_steps"] for p in subset})
            records = []
            for x in xs:
                values = [p for p in subset if p["env_steps"]==x]
                record = dict(method=method,lr=lr,env_steps=x,training_seeds=len(values))
                for key in ("validation_reward","qos_rate","mean_task_cost"):
                    record[key] = float(np.mean([p[key] for p in values]))
                    record[key+"_seed_std"] = float(np.std([p[key] for p in values],ddof=1)) if len(values)>1 else None
                records.append(record)
            aggregate.extend(records)
            if records:
                for ax,key in zip(axes,("validation_reward","qos_rate","mean_task_cost")):
                    mean = np.asarray([r[key] for r in records])
                    line, = ax.plot(xs,mean,label="%s LR%d" % (method,lr))
                    if all(r["training_seeds"]>=3 for r in records):
                        std = np.asarray([r[key+"_seed_std"] for r in records])
                        ax.fill_between(xs,mean-std,mean+std,color=line.get_color(),alpha=.15)
    for ax, title in zip(axes, ("Validation reward (higher is better)", "QoS violation fraction", "Validation cost")):
        ax.set(xlabel="Global environment interactions", ylabel=title)
        ax.grid(alpha=.2)
    if summaries:
        axes[0].legend(fontsize=6)
    formal = completed == 18 and all(r.get("training_seeds") == 3 for r in aggregate)
    fig.suptitle("Three-seed training: validation only, final tidal test pending" if formal else
                 "Bounded pilot: validation only, not three-seed convergence evidence")
    fig.tight_layout()
    fig.savefig(root/"convergence_pilot.png", dpi=160)
    if formal:
        fig.savefig(root/"convergence_validation.png", dpi=200)
    plt.close(fig)
    csv_write(root/"summary.csv", summaries)
    csv_write(root/"convergence_by_seed.csv",seed_points)
    csv_write(root/"convergence_aggregate.csv",aggregate)
    csv_write(root/"policy_mode_diagnostics.csv",mode_points)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, method in zip(axes, METHODS):
        for lr in (1,2):
            for mode, style in (("argmax","-"),("sampled","--")):
                records = [r for r in mode_points if r["method"]==method and r["lr"]==lr
                           and r["mode"]==mode]
                xs = sorted({r["env_steps"] for r in records})
                ys = [np.mean([r["mean_task_cost"] for r in records if r["env_steps"]==x]) for x in xs]
                ax.plot(xs,ys,style,color="C%d" % (lr-1),label="LR%d %s" % (lr,mode))
        ax.set(title=method, xlabel="Global environment interactions", ylabel="Validation cost")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Policy-mode diagnostics; sampled curves never select checkpoints")
    fig.tight_layout()
    fig.savefig(root/"policy_mode_diagnostics.png",dpi=160)
    plt.close(fig)
    candidates = {}
    for lr in (1,2):
        rows = [r for r in summaries if r["method"]=="TMPPO" and r["lr"]==lr]
        candidates[str(lr)] = dict(training_seeds=sorted(r["train_seed"] for r in rows),
                                  mean_selected_validation_cost=float(np.mean([r["mean_task_cost"] for r in rows])) if rows else None)
    qualified = all(candidates[str(lr)] is not None and
                    len(candidates[str(lr)]["training_seeds"])==3 and
                    set(candidates[str(lr)]["training_seeds"])==set(TRAIN_SEEDS) for lr in (1,2))
    all_methods_three_seeds = all(
        {r["train_seed"] for r in summaries if r["method"]==method and r["lr"]==lr}==set(TRAIN_SEEDS)
        for method in METHODS for lr in (1,2))
    json_write(root/"learning_rate_selection.json",dict(
        candidates=candidates,final_selection_allowed=qualified,test_used=False,
        selected_lr=min((1,2),key=lambda lr:candidates[str(lr)]["mean_selected_validation_cost"]) if qualified else None,
        reason="three predeclared training seeds per LR required; pilot remains provisional"))
    audit = dict(paired_arrival_hashes=all(len(h)==1 for h in paired.values()),
                 completed_cells=completed, rows=len(summaries), test_seeds_used=False,
                 full_tidal_evaluation=False, three_training_seeds=all_methods_three_seeds,
                 DP_proven=False, raw_task_checks="performed for every validation cohort",
                 approximate_td3_discrete_adapter=True)
    json_write(root/"audit.json", audit)
    if not audit["paired_arrival_hashes"]:
        raise AssertionError("Cross-cell validation trajectories differ")
    return summaries


def settings_from_args(a):
    result = dict(method=a.method, lr=a.lr, seed=a.seed, updates=a.updates, rollout=a.rollout,
                episode_steps=a.episode_steps, phase_count=a.phase_count or (3 if a.updates<=128 else 5),
                validate_every=a.validate_every, validation_steps=a.validation_steps,
                warmup_steps=a.warmup_steps, validation_seeds=list(VALIDATION_SEEDS),
                validation_phases=list(PHASES), training_profile=a.training_profile,
                checkpoint_every=a.checkpoint_every, schedule_updates=a.schedule_updates or a.updates,
                policy_diagnostics=a.policy_diagnostics, baseline_diagnostics=a.baseline_diagnostics,
                validation_objective=a.validation_objective)
    if a.diagnostics_endpoints_only:
        result["diagnostics_endpoints_only"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "pilot", "cell", "report"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--method", choices=METHODS, default="TMPPO")
    parser.add_argument("--lr", type=int, choices=(1,2), default=1)
    parser.add_argument("--seed", type=int, default=TRAIN_SEEDS[0])
    parser.add_argument("--updates", type=int, default=48)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=1024)
    parser.add_argument("--phase-count", type=int)
    parser.add_argument("--validate-every", type=int, default=12)
    parser.add_argument("--validation-steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--wall-seconds", type=float, default=600.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-long-cell", action="store_true")
    parser.add_argument("--training-profile", choices=("audit_v1", "team_v2", "credit_v3"), default="audit_v1")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--schedule-updates", type=int)
    parser.add_argument("--policy-diagnostics", action="store_true")
    parser.add_argument("--diagnostics-endpoints-only", action="store_true")
    parser.add_argument("--baseline-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-objective", choices=("equal_scenario_cost", "pooled_task_cost"),
                        default="equal_scenario_cost")
    a = parser.parse_args()
    if a.command == "plan":
        plan = paper_plan()
        plan["convergence"] = [dict(method=m, lr=lr, seed=seed, updates=2000, rollout=128)
                               for m in METHODS for lr in (1,2) for seed in TRAIN_SEEDS]
        print(json.dumps(plan, indent=2))
        return
    if a.out is None:
        parser.error("Provide a new --out directory (or existing directory for resume/report)")
    if min(a.updates,a.rollout,a.episode_steps,a.validate_every,a.validation_steps) < 1 or a.warmup_steps < 0:
        parser.error("Budgets must be positive")
    if a.command == "report":
        report(a.out)
        return
    if a.command == "pilot":
        if a.updates > 128 or a.resume:
            parser.error("Pilot is bounded to <=128 updates; resume individual saved cells")
        a.out.mkdir(parents=True, exist_ok=False)
        json_write(a.out/"pilot_plan.json", dict(
            seed=a.seed, methods=METHODS, learning_rates=LEARNING_RATES,
            updates=a.updates, rollout=a.rollout, total_cells=6,
            total_interactions=6*a.updates*a.rollout, formal_matrix_launched=False,
            predeclared=True, max_wall_seconds_per_cell=a.wall_seconds))
        awake = os.name=="nt"
        if awake and not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError("Cannot request temporary idle-sleep inhibition")
        try:
            for method in METHODS:
                for lr in (1,2):
                    ram = free_ram_gib()
                    if ram is not None and ram < 1.:
                        raise RuntimeError("Less than 1 GiB available RAM before next CPU pilot cell")
                    cell = a.out/("%s_LR%d_seed%d" % (method,lr,a.seed))
                    command = [sys.executable, str(Path(__file__).resolve()), "cell", "--out", str(cell),
                               "--method", method, "--lr", str(lr), "--seed", str(a.seed)]
                    for flag in ("updates","rollout","episode_steps","validate_every","validation_steps",
                                 "warmup_steps","wall_seconds","phase_count","training_profile",
                                 "checkpoint_every","schedule_updates","validation_objective"):
                        value = getattr(a,flag)
                        if value is not None:
                            command += ["--"+flag.replace("_","-"), str(value)]
                    if a.policy_diagnostics:
                        command.append("--policy-diagnostics")
                    if a.diagnostics_endpoints_only:
                        command.append("--diagnostics-endpoints-only")
                    if not a.baseline_diagnostics:
                        command.append("--no-baseline-diagnostics")
                    with (a.out/(cell.name+".log")).open("w", encoding="utf-8") as stdout, \
                         (a.out/(cell.name+".err.log")).open("w", encoding="utf-8") as stderr:
                        print("START", cell.name, flush=True)
                        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
                    gc.collect()
                    with (a.out/"resource_release.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(dict(cell=cell.name,exit_code=result.returncode,
                                                      free_ram_gib=free_ram_gib()))+"\n")
                    if result.returncode:
                        report(a.out)
                        raise SystemExit(result.returncode)
            report(a.out)
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return
    if a.updates > 128 and not a.allow_long_cell:
        parser.error("Long single cells require explicit --allow-long-cell; no automatic full matrix")
    settings = settings_from_args(a)
    if min(a.updates, a.rollout, a.validate_every, a.checkpoint_every) <= 0:
        parser.error("Update/rollout/checkpoint/validation intervals must be positive")
    if settings["schedule_updates"] < a.updates:
        parser.error("Schedule horizon must cover the requested training budget")
    try:
        complete = train_cell(a.out, settings, a.resume, a.wall_seconds)
    except (ValueError, FileExistsError):
        # A rejected invocation must not mark someone else's existing run failed.
        raise
    except Exception as exc:
        if a.out.exists():
            json_write(a.out/"failure.json", dict(error=repr(exc)))
        raise
    raise SystemExit(0 if complete else 2)


if __name__ == "__main__":
    main()

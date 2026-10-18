"""Finite, resumable study queue. Coordinator uses only the standard library."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

import hybrid_campaign as h

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (87021,)
METHODS = ("TMPPO", "IPPO")
TAG = re.compile(r"paper2500-wave-(\d{8}T\d{6}Z)-(\d{4})\Z")
REPO = "Mustard-0/tidal-offload-sim"
MAX_INTERACTIONS = 360000000


def settings(method, lr, seed, overrides=None):
    steps = 2500*6000
    return dict(method=method, lr=lr, seed=seed, updates=math.ceil(steps/128),
        total_env_steps=steps, rollout=128, episode_steps=6000, phase_count=5,
        validate_every=3750, validation_steps=1000, warmup_steps=300,
        validation_seeds=[814001, 814019], validation_phases=[30., 130., 230.],
        training_profile="paper2500_v8", checkpoint_every=3750,
        schedule_updates=math.ceil(steps/128), policy_diagnostics=True,
        diagnostics_endpoints_only=True, baseline_diagnostics=False,
        validation_objective="pooled_task_cost", record_complete_episodes=True,
        retain_initial_checkpoint=True, append_training_csv=True, status_every=25,
        environment_overrides=overrides or {})


def train_task(stage, method, lr, seed, suffix="", overrides=None):
    return dict(id=f"{stage}_{method}_LR{lr}_seed{seed}{suffix}", kind="train", stage=stage,
                settings=settings(method, lr, seed, overrides))


def convergence_tasks():
    return [train_task("convergence", m, lr, seed) for seed in SEEDS
            for m in METHODS for lr in (1, 2)]


def plan(verification):
    proof = h.read_json(verification)
    hashes = h.code_hashes()
    if not proof.get("passed") or proof["source_hashes"] != hashes:
        raise ValueError("Exact-source verification required")
    if h.command(["git", "status", "--porcelain"]) or h.command(
            ["git", "branch", "--show-current"]) in ("main", "master", ""):
        raise ValueError("Freeze a clean non-main branch first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    value = dict(schema="paper2500-v1", campaign="paper2500-campaign-"+stamp,
        stamp=stamp, commit=h.command(["git", "rev-parse", "HEAD"]), source_hashes=hashes,
        runtime=dict(h.RUNTIME, local_workers=4, github_workers=16, segment_updates=3750,
                     segment_wall_s=10800, maximum_segments_per_task=48),
        stages=["convergence", "auxiliary", "scaling", "privacy_C1", "sensitivity", "sensitivity_evaluation"],
        training_seeds=list(SEEDS), complete_episodes_each=2500,
        maximum_training_cells=24, maximum_primary_training_interactions=MAX_INTERACTIONS,
        authorization="User confirmed2500 complete episodes, free CPU, local/cloud parallel and quota fallback",
        learning_rate_selection="pooled cost at final three validation points; one declared exploratory training seed",
        quality_gate=dict(max_tail_cost_ratio=.99, max_qos_increase=.01, max_tail_qos=.10,
                          max_tail_cost_cv=.05, minimum_learned_seed_count=1),
        initial_tasks=convergence_tasks(), verification=proof,
        frozen_environment="8BS+3LEO, N60, WLO/Shuffle, unchanged workload/reward/privacy",
        maximum_wall_days=7, resource_wait_hours=12,
        caveats=["No guaranteed TMAPO win", "legacy accountant exploratory, not full-trajectory DP",
                 "Recent methods are explicitly architecture-level MARATD3 and MATORA-DQN adaptations, not author reproductions",
                 "Single training seed: not final multi-seed statistical evidence",
                 "C1 is fixed300 users, two mechanisms and five epsilon values, not extra full service grids",
                 "Final evaluation uses frozen models and four reserved test seeds, no test-based retuning"])
    value["plan_hash"] = h.digest(value)
    return value


def load(root):
    value = h.read_json(root/"plan.json")
    if h.digest({k: v for k, v in value.items() if k != "plan_hash"}) != value["plan_hash"]:
        raise ValueError("Plan changed")
    if h.code_hashes() != value["source_hashes"] or h.command(["git", "rev-parse", "HEAD"]) != value["commit"]:
        raise ValueError("Frozen source/commit changed")
    return value


def immutable(path, value):
    if path.exists():
        if h.read_json(path) != value:
            raise ValueError("Conflicting immutable evidence: "+str(path))
    else:
        h.write_json(path, value)


def gh(*args):
    return h.command(["gh", *map(str, args)], timeout=600)


def gh_json(*args):
    return json.loads(gh(*args))


def public_only():
    value = gh_json("repo", "view", REPO, "--json", "isPrivate,nameWithOwner")
    if value["isPrivate"] or value["nameWithOwner"] != REPO:
        raise ValueError("Only the authorized public pure-code repository may run")


def publish_plan(root, p):
    public_only()
    releases = gh_json("api", f"repos/{REPO}/releases?per_page=100")
    if not any(r["tag_name"] == p["campaign"] for r in releases):
        gh("release", "create", p["campaign"], "--repo", REPO, "--target", p["commit"],
           "--title", "Finite 2500 complete-episode simulation", "--notes",
           "Public simulation code/results only. No paper or private source history.")
    metadata = gh_json("api", f"repos/{REPO}/releases/tags/{p['campaign']}")
    found = [a for a in metadata["assets"] if a["name"] == "plan.json"]
    if not found:
        gh("release", "upload", p["campaign"], root/"plan.json", "--repo", REPO)
    elif len(found) != 1 or found[0]["digest"] != "sha256:"+h.file_hash(root/"plan.json"):
        raise ValueError("Published plan differs")


def validate_request(p, request):
    if request["plan_hash"] != p["plan_hash"] or not request["entries"]:
        raise ValueError("Wrong/empty segment request")
    identifiers = [e["task"]["id"] for e in request["entries"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate requested tasks")
    for e in request["entries"]:
        task = e["task"]
        if not 1 <= e["segment"] <= p["runtime"]["maximum_segments_per_task"]:
            raise ValueError("Finite segment allowance exceeded")
        if task["kind"] == "train":
            s = task["settings"]
            if (s["seed"] not in SEEDS or s["lr"] not in (1, 2) or
                    s != settings(s["method"], s["lr"], s["seed"], s["environment_overrides"])):
                raise ValueError("Unregistered training budget/settings")
            if task["stage"] == "convergence":
                if task not in convergence_tasks():
                    raise ValueError("Unregistered convergence task")
            elif task["stage"] == "auxiliary":
                if task not in auxiliary_tasks({"TMPPO": s["lr"]}):
                    raise ValueError("Unregistered auxiliary task")
            elif task["stage"] == "sensitivity":
                if task not in sensitivity_tasks(s["lr"]):
                    raise ValueError("Unregistered sensitivity task")
            else:
                raise ValueError("Unregistered training stage")
        elif task["kind"] == "evaluation":
            from paper2500_evaluation import validate_evaluation_task
            validate_evaluation_task(task)
        else:
            raise ValueError("Unknown task kind")


def receive_asset(root, ref):
    metadata = gh_json("api", f"repos/{REPO}/releases/tags/{ref['release']}")
    matches = [a for a in metadata["assets"] if a["name"] == ref["name"]]
    if len(matches) != 1:
        raise ValueError("Missing or duplicate immutable checkpoint asset")
    asset = matches[0]
    if ref.get("digest") and asset["digest"] != ref["digest"]:
        raise ValueError("Parent checkpoint asset changed")
    target = root/"incoming"/str(asset["id"])
    target.mkdir(parents=True, exist_ok=True)
    archive = target/asset["name"]
    if not archive.exists():
        gh("release", "download", ref["release"], "--repo", REPO,
           "--pattern", asset["name"], "--dir", target)
    if archive.stat().st_size != asset["size"] or "sha256:"+h.file_hash(archive) != asset["digest"]:
        raise ValueError("Checkpoint digest/size mismatch")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Corrupt checkpoint ZIP")
        origin = json.loads(zipped.read("execution_origin.json"))
    destination = target/"data"
    if destination.exists():
        if not (target/"verified.json").exists() or h.read_json(target/"verified.json") != dict(digest=asset["digest"], origin=origin):
            raise ValueError("Unattested previous extraction")
    else:
        h.extract_evidence(archive, destination)
        immutable(target/"verified.json", dict(digest=asset["digest"], origin=origin))
    return destination, origin, dict(release=ref["release"], name=asset["name"],
                                     digest=asset["digest"], size=asset["size"])


def restore(root, p, task, parent):
    path = root/"cells"/task["id"]
    if path.exists():
        return path
    if parent is not None:
        data, origin, _ = receive_asset(root, parent)
        if (origin["plan_hash"] != p["plan_hash"] or origin["task_id"] != task["id"]
                or origin["commit"] != p["commit"] or origin["outcome"] not in ("paused", "complete")):
            raise ValueError("Wrong parent checkpoint lineage")
        shutil.copytree(data/"cell", path)
    return path


def inspect_training(path, task, require_complete=False):
    import torch
    import paper_convergence as pc
    m = h.read_json(path/"manifest.json")
    if m["settings"] != task["settings"] or m["source_hashes"] != pc.source_hashes():
        raise ValueError("Saved settings/source differ")
    cfg, training = pc.configuration(task["settings"])
    if m["environment"] != cfg or m["training"] != training:
        raise ValueError("Expanded frozen configuration differs")
    if (path/"failure.json").exists():
        raise ValueError("Numerical/code failures are not resumable")
    state = torch.load(path/"latest.pt", map_location="cpu", weights_only=False)
    rows = pc.read_csv(path/"train.csv")
    if not rows or int(rows[-1]["update"]) != state["update"]:
        raise ValueError("Raw CSV and latest checkpoint disagree")
    for filename, saved in (("train.csv", state["training"]), ("validation.csv", state["validation"]),
                            ("episodes.csv", state["episodes"])):
        raw = pc.read_csv(path/filename) if (path/filename).exists() else []
        if len(raw) != len(saved) or any(
                r.get(k, "") != ("" if s.get(k) is None else str(s[k]))
                for r, s in zip(raw, saved) for k in set(r) | set(s)):
            raise ValueError("Raw CSV prefix differs from checkpoint: "+filename)
    if (path/"complete.json").exists():
        c = h.read_json(path/"complete.json")
        if c["env_steps"] != task["settings"]["total_env_steps"] or c["completed_episodes"] != 2500:
            raise ValueError("Exit success was not2500 complete episodes")
        episodes = pc.read_csv(path/"episodes.csv")
        if len(episodes) != 2500 or [int(r["env_steps"]) for r in episodes] != [6000*i for i in range(1, 2501)]:
            raise ValueError("Complete episode budget differs")
        return "complete"
    if require_complete:
        raise ValueError("Incomplete training")
    pause = h.read_json(path/"paused.json")
    if pause["update"] != state["update"] or pause["reason"] not in (
            "requested test boundary", "wall time budget", "available RAM below .5GiB", "STOP_AFTER_UPDATE"):
        raise ValueError("Unknown or inconsistent pause")
    return "paused"


def run_cell(root, p, task, parent):
    path = restore(root, p, task, parent)
    if task["kind"] != "train":
        from paper2500_evaluation import run_evaluation_task
        return run_evaluation_task(root, p, task, path)
    import paper_convergence as pc
    resume = path.exists()
    start = 0
    if resume:
        outcome = inspect_training(path, task)
        if outcome == "complete":
            return outcome
        start = h.read_json(path/"paused.json")["update"]
        if (path/"STOP_AFTER_UPDATE").exists() or (path.parent/"STOP_AFTER_UPDATE").exists():
            raise ValueError("Explicit STOP requires human/monitor verification before archival")
    finish = min(task["settings"]["updates"], start+p["runtime"]["segment_updates"])
    done = pc.train_cell(path, task["settings"], resume=resume,
                         wall_seconds=p["runtime"]["segment_wall_s"], stop_after=finish)
    return "complete" if done else "paused"


def bundle(root, p, task, destination, parent, outcome):
    immutable(root/"receipts"/(destination.name+".json"), dict(task_id=task["id"], outcome=outcome,
              parent=parent, plan_hash=p["plan_hash"], commit=p["commit"]))
    if destination.exists():
        raise FileExistsError("Result ZIP already exists")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        origin = dict(plan_hash=p["plan_hash"], commit=p["commit"], task_id=task["id"],
                      parent=parent, outcome=outcome, run_id=os.environ.get("GITHUB_RUN_ID"),
                      attempt=os.environ.get("GITHUB_RUN_ATTEMPT"), platform=sys.platform)
        zipped.writestr("execution_origin.json", json.dumps(origin, indent=2))
        for path in sorted((root/"cells"/task["id"]).rglob("*")):
            if path.is_file() and not path.name.endswith((".tmp", ".worker.lock")):
                zipped.write(path, "cell/"+path.relative_to(root/"cells"/task["id"]).as_posix())
        for path in (root/"logs").glob(task["id"]+"*"):
            if path.is_file():
                zipped.write(path, "logs/"+path.name)
    return destination


def worker(root, p, request, task_id, local=False):
    validate_request(p, request)
    entry = next(e for e in request["entries"] if e["task"]["id"] == task_id)
    task, parent = entry["task"], entry["parent"]
    logs = root/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    number = entry["segment"]
    suffix = f".segment{number:03d}"
    with (logs/(task_id+suffix+".stdout.log")).open("x", encoding="utf-8") as stdout, (
            logs/(task_id+suffix+".stderr.log")).open("x", encoding="utf-8") as stderr:
        args = [sys.executable, "-u", str(Path(__file__)), "cell", "--out", str(root),
                "--request", str(root/"requests"/request["filename"]), "--task", task_id, "--execute"]
        proc = subprocess.Popen(args, cwd=ROOT, stdout=stdout, stderr=stderr)
        code = proc.wait()
    path = root/"cells"/task_id
    immutable(logs/(task_id+suffix+".resource_release.json"), dict(child_pid=proc.pid,
        exit_code=code, child_exited=True, available_ram_gib=h.free_ram_gib(),
        observed_utc=datetime.now(timezone.utc).isoformat()))
    outcome = "complete" if code == 0 and (path/"complete.json").exists() else (
        "paused" if code == 2 and (path/"paused.json").exists() else "failed")
    immutable(root/"outcomes"/(task_id+suffix+".json"), dict(outcome=outcome, exit_code=code,
              segment=number, plan_hash=p["plan_hash"], parent=parent))
    release = request["release"]
    target = root/(task_id+suffix+".zip")
    if path.exists():
        bundle(root, p, task, target, parent, outcome)
        if not local:
            gh("release", "upload", release, target, "--repo", REPO)
    if outcome == "failed":
        raise RuntimeError("Worker failed; preserve all evidence, do not rerun blindly")


def choose_and_gate(root, tasks, p, stage="convergence"):
    import torch
    import paper_convergence as pc
    from collections import defaultdict
    rows = []
    paired = defaultdict(set)
    initial_actors = None
    for task in tasks:
        path = root/"cells"/task["id"]
        inspect_training(path, task, require_complete=True)
        m = h.read_json(path/"manifest.json")
        if m["source_commit"] != p["commit"]:
            raise ValueError("Executed source commit differs")
        for rel, checksum in m["source_hashes"].items():
            if p["source_hashes"].get(rel) != checksum or h.file_hash(path/"source"/rel, True) != checksum:
                raise ValueError("Frozen source snapshot differs")
        rt = m["runtime"]
        if (rt["python_version"] != p["runtime"]["python"] or rt["device"] != "cpu" or
                rt["torch_threads"] != 1 or any(rt[k] != p["runtime"][k] for k in ("torch", "numpy", "scipy", "matplotlib"))):
            raise ValueError("Executed dependency/device contract differs")
        for checkpoint in ("initial_checkpoint.pt", "best_checkpoint.pt", "latest.pt"):
            with zipfile.ZipFile(path/checkpoint) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Checkpoint CRC failed")
        if stage == "convergence":
            initial = torch.load(path/"initial_checkpoint.pt", map_location="cpu", weights_only=False)["actors"]
            if initial_actors is None:
                initial_actors = initial
            elif set(initial) != set(initial_actors) or any(not torch.equal(initial[k], v) for k, v in initial_actors.items()):
                raise ValueError("Actual initial actors differ across four curves")
        cfg = m["environment"]
        for ep in pc.read_csv(path/"episodes.csv"):
            h.close(float(ep["episode_reward_sum"]), float(ep["mean_training_reward"])*6000)
            h.close(float(ep["episode_reward_sum"]), -float(ep["physical_cost_total"])/(
                cfg["users_per_region"]*3*cfg["arrival_per_user_s"]*cfg["dt_s"]))
        validation = [r for r in pc.read_csv(path/"validation.csv") if r["method"] == task["settings"]["method"]]
        updates = sorted({int(r["update"]) for r in validation})
        s = task["settings"]
        expected_updates = sorted({0, s["updates"], *range(s["validate_every"], s["updates"]+1, s["validate_every"])})
        grid = {(u, seed, float(phase)) for u in expected_updates for seed in s["validation_seeds"] for phase in s["validation_phases"]}
        actual = [(int(r["update"]), int(r["seed"]), float(r["phase_s"])) for r in validation]
        if len(actual) != len(grid) or set(actual) != grid:
            raise ValueError("Validation grid changed")
        for u in updates:
            subset = [r for r in validation if int(r["update"]) == u]
            identities = h.audit_validation(path/"validation_traces"/f"tasks_{u:06d}.csv", cfg, subset, s["method"])
            for key, identity in identities.items():
                paired[key].add(identity)
        points = [pc.validation_mean([r for r in validation if int(r["update"]) == u],
                                    "mean_task_cost", "pooled_task_cost") for u in updates]
        tail = [r for r in validation if int(r["update"]) in updates[-3:]]
        qos = pc.validation_mean(tail, "qos_rate", "pooled_task_cost")
        initial_qos = pc.validation_mean([r for r in validation if int(r["update"]) == 0], "qos_rate", "pooled_task_cost")
        selected = h.read_json(path/"selection.json")
        best = min(range(len(points)), key=lambda i: points[i])
        if selected["update"] != updates[best] or selected["test_used"]:
            raise ValueError("Checkpoint selection differs from validation including update0")
        h.close(selected["validation_cost"], points[best])
        rows.append(dict(task=task["id"], method=task["settings"]["method"], lr=task["settings"]["lr"],
            seed=task["settings"]["seed"], initial_cost=points[0],
            tail_cost=pc.validation_mean(tail, "mean_task_cost", "pooled_task_cost"),
            tail_qos=qos, initial_qos=initial_qos, tail_cv=statistics.pstdev(points[-3:])/statistics.mean(points[-3:]),
            selected_update=selected["update"]))
    if not paired or any(len(v) != 1 for v in paired.values()):
        raise ValueError("Validation workloads do not pair")
    methods = sorted({r["method"] for r in rows})
    choices = {}
    for method in methods:
        candidates = sorted({r["lr"] for r in rows if r["method"] == method})
        for lr in candidates:
            if {r["seed"] for r in rows if r["method"] == method and r["lr"] == lr} != set(SEEDS):
                raise ValueError("Never select a preferred training seed")
        choices[method] = min(candidates, key=lambda lr: statistics.mean(
            r["tail_cost"] for r in rows if r["method"] == method and r["lr"] == lr))
    g = p["quality_gate"]
    selected = [r for r in rows if r["method"] == "TMPPO" and r["lr"] == choices.get("TMPPO")]
    passed = stage != "convergence" or (bool(selected) and all(r["tail_cost"] <= r["initial_cost"]*g["max_tail_cost_ratio"] and
        r["tail_qos"] <= min(g["max_tail_qos"], r["initial_qos"]+g["max_qos_increase"]) and
        r["tail_cv"] <= g["max_tail_cost_cv"] for r in selected) and sum(
        r["selected_update"] > 0 for r in selected) >= g["minimum_learned_seed_count"])
    result = dict(passed=passed, integrity_passed=True, paired_raw_tasks=True,
                  rows=rows, selected_lr=choices, heldout_test_used=False,
                  meaning="Predeclared learning/stability gate, not a requirement to beat IPPO")
    immutable(root/(stage+"_gate.json"), result)
    return result


def auxiliary_tasks(choices):
    return [train_task("auxiliary", m, lr, seed) for seed in SEEDS
            for m, lr in (("NoTidal", choices["TMPPO"]), ("MARATD3-Adapted", 1), ("MARATD3-Adapted", 2),
                          ("MATORA-DQN-Adapted", 1), ("MATORA-DQN-Adapted", 2))]


def sensitivity_tasks(lr):
    pairs = (("weight_time", "weight_energy"), ("weight_time", "weight_quality"), ("weight_energy", "weight_quality"))
    result = []
    for pi, pair in enumerate(pairs):
        for wi, (x, y) in enumerate(((.1,.5),(.2,.4),(.3,.3),(.4,.2),(.5,.1))):
            weights = dict(weight_time=.2, weight_energy=.2, weight_quality=.2, weight_violation=1.)
            weights[pair[0]], weights[pair[1]] = x, y
            for seed in SEEDS:
                result.append(train_task("sensitivity", "TMPPO", lr, seed, f"_p{pi}w{wi}", weights))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "coordinate", "prepare", "worker", "cell", "gate", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--task")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.command == "init":
        p = plan(args.verification)
        root.mkdir(parents=True, exist_ok=False)
        h.write_json(root/"plan.json", p)
        print(json.dumps(dict(plan_hash=p["plan_hash"], training_cells=24,
                             maximum_primary_training_interactions=MAX_INTERACTIONS)))
        return
    if args.command == "prepare":
        match = TAG.fullmatch(args.tag or "")
        if not match:
            raise ValueError("Unregistered wave tag")
        root.mkdir(parents=True, exist_ok=False)
        campaign = "paper2500-campaign-"+match.group(1)
        gh("release", "download", campaign, "--repo", REPO, "--pattern", "plan.json", "--dir", root)
        p = load(root)
        release = f"paper2500-{match.group(1)}-w{match.group(2)}"
        destination = root/"requests"
        destination.mkdir()
        gh("release", "download", release, "--repo", REPO, "--pattern", "wave.json", "--dir", destination)
        request = h.read_json(destination/"wave.json")
        if request["plan_hash"] != p["plan_hash"] or request["release"] != release:
            raise ValueError("Wrong wave identity")
        validate_request(p, request)
        immutable(destination/request["filename"], request)
        print(json.dumps(dict(task=[e["task"]["id"] for e in request["entries"]])))
        return
    p = load(root)
    if not args.execute and args.command in ("cell", "worker", "coordinate"):
        raise ValueError("Execution must be explicit")
    if args.command == "coordinate":
        from paper2500_coordinator import coordinate
        coordinate(root, p)
        return
    if args.command == "report":
        from paper2500_evaluation import report
        report(root, p)
        return
    if args.command == "gate":
        state = h.read_json(root/"launcher.json")
        if args.task not in ("convergence", "auxiliary") or state["stage"] != args.task:
            raise ValueError("Wrong stage gate")
        choose_and_gate(root, state["tasks"], p, args.task)
        from paper2500_evaluation import report
        report(root, p)
        return
    request = h.read_json(args.request or root/"requests/wave.json")
    validate_request(p, request)
    if args.command == "worker":
        worker(root, p, request, args.task, local=args.local)
        return
    entry = next(e for e in request["entries"] if e["task"]["id"] == args.task)
    try:
        outcome = run_cell(root, p, entry["task"], entry["parent"])
    except Exception as error:
        cell = root/"cells"/args.task
        cell.mkdir(parents=True, exist_ok=True)
        h.write_json(cell/"failure.json", dict(type=type(error).__name__, error=str(error)))
        raise
    raise SystemExit(0 if outcome == "complete" else 2)


if __name__ == "__main__":
    main()

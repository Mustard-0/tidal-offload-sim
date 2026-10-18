"""Four registered PPO learning-rate curves, with actual complete-episode logs."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import hybrid_campaign as h

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("TMPPO", "IPPO")
LABELS = {"TMPPO": "TMAPO", "IPPO": "IPPO"}
TAG = re.compile(r"four-episode-curves-(\d{8}T\d{6}Z)\Z")
CAMPAIGN = re.compile(r"fourcurves-campaign-\d{8}T\d{6}Z\Z")


def settings(method, lr):
    if method not in METHODS or lr not in (1, 2):
        raise ValueError("Only TMAPO/IPPO with the two registered learning rates")
    return dict(method=method, lr=lr, seed=87021, updates=6000, rollout=128,
        episode_steps=6000, phase_count=5, validate_every=375, validation_steps=1000,
        warmup_steps=300, validation_seeds=[814001, 814019],
        validation_phases=[30., 130., 230.], training_profile="from_scratch_v7",
        checkpoint_every=100, schedule_updates=6000, policy_diagnostics=True,
        diagnostics_endpoints_only=True, baseline_diagnostics=False,
        validation_objective="pooled_task_cost", record_complete_episodes=True,
        retain_initial_checkpoint=True)


def build_plan(campaign, commit, hashes):
    if not CAMPAIGN.fullmatch(campaign):
        raise ValueError("Unregistered four-curve campaign identity")
    cells = [dict(id="%s_LR%d_seed87021" % (m, lr), owner="github", settings=settings(m, lr))
             for m in METHODS for lr in (1, 2)]
    plan = dict(schema="four-episode-curves-v1", campaign=campaign, commit=commit,
        source_hashes=hashes, stages=dict(probe=cells),
        runtime=dict(h.RUNTIME, github_workers=4, local_workers=0, worker_wall_s=15000),
        gate=dict(tail_points=3, max_cost_ratio=1.25, max_qos_increase=.05, max_tail_qos=.1),
        complete_episodes_each=128, primary_interactions_each=768000,
        total_primary_interactions=3072000, smoothing_window_complete_episodes=50,
        smoothing="trailing; min_periods=1 disclosed; no future data or hand edits",
        initialization="Common from-scratch Xavier actors, small output head; no hand-coded cost prior",
        display_names=LABELS, learning_rate_selection="pooled cost over final three independent validation checkpoints",
        zoom="One common 50-episode window around validation-selected TMAPO checkpoint, labelled exploratory",
        authorization="User requested TMAPO two LRs and another RL method two LRs, 50-episode smoothing",
        exploratory=True, publication_grade=False, final_test_executed=False,
        extension_policy="Exactly four cells, one seed; no automatic rerun, extra LR, training seed or formal study")
    plan["plan_hash"] = h.digest(plan)
    return plan


def load_plan(root):
    plan = h.read_json(root / "plan.json")
    assert h.digest({k: v for k, v in plan.items() if k != "plan_hash"}) == plan["plan_hash"]
    expected = build_plan(plan["campaign"], plan["commit"], plan["source_hashes"])
    if any(plan.get(k) != v for k, v in expected.items() if k != "plan_hash"):
        raise ValueError("Budget, seeds or selection changed")
    if not plan.get("verification", {}).get("passed") or plan["verification"]["source_hashes"] != plan["source_hashes"]:
        raise ValueError("Missing exact-source verification")
    if h.code_hashes() != plan["source_hashes"] or h.command(["git", "rev-parse", "HEAD"]) != plan["commit"]:
        raise ValueError("Frozen source changed")
    return plan


def initialize(root, verification):
    proof = h.read_json(verification)
    if not proof.get("passed") or proof["source_hashes"] != h.code_hashes():
        raise ValueError("Verify this source first")
    if h.command(["git", "branch", "--show-current"]) in ("main", "master", "") or h.command(["git", "status", "--porcelain"]):
        raise ValueError("Commit on a clean development branch")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = build_plan("fourcurves-campaign-" + stamp, h.command(["git", "rev-parse", "HEAD"]), h.code_hashes())
    plan["verification"] = proof
    plan["plan_hash"] = h.digest({k: v for k, v in plan.items() if k != "plan_hash"})
    root.mkdir(parents=True, exist_ok=False)
    h.write_json(root / "plan.json", plan)
    return plan


def trailing_mean(values, window=50):
    if window <= 0 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Invalid smoothing input")
    result, total = [], 0.
    for i, value in enumerate(values):
        total += float(value)
        if i >= window:
            total -= float(values[i-window])
        result.append(total / min(i+1, window))
    return result


def progress(plan, root, cell_id, update):
    run_id, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not run_id.isdigit() or not attempt.isdigit() or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return
    cell = root / "cells/probe" / cell_id
    episodes = list(h.rows(cell / "episodes.csv")) if (cell / "episodes.csv").exists() else []
    episodes = episodes[:update*128//6000]
    path = cell / "progress" / ("four_progress_%s_run%s_attempt%s_update%06d.json" % (cell_id, run_id, attempt, update))
    h.write_json(path, dict(plan_hash=plan["plan_hash"], cell=cell_id, run_id=run_id, attempt=attempt,
        validated_update=update, completed_episodes=len(episodes), episodes=episodes,
        observed_utc=datetime.now(timezone.utc).isoformat(),
        note="Live raw episode aggregates, not a durable checkpoint or final audit"))
    try:
        h.command(["gh", "release", "upload", plan["campaign"], path, "--repo", repo])
    except (OSError, subprocess.CalledProcessError) as error:
        print("PROGRESS_UPLOAD_WARNING", type(error).__name__, flush=True)


def worker(root, plan, cell_id):
    cell = next(c for c in plan["stages"]["probe"] if c["id"] == cell_id)
    h.require_runtime(plan)
    if h.free_ram_gib() < plan["runtime"]["start_ram_gib"]:
        raise RuntimeError("Insufficient available RAM")
    out = root / "cells/probe" / cell_id
    if out.exists():
        raise FileExistsError("No duplicate or blind cloud resume")
    logs = root / "logs/probe"
    logs.mkdir(parents=True, exist_ok=True)
    with h.awake_lock(root), (logs/(cell_id+".log")).open("x", encoding="utf-8") as stdout, \
            (logs/(cell_id+".err.log")).open("x", encoding="utf-8") as stderr:
        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "cell",
            "--out", str(root), "--cell", cell_id, "--execute"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8", bufsize=1)
        try:
            for line in proc.stdout:
                stdout.write(line)
                stdout.flush()
                print(line, end="", flush=True)
                match = re.match(r"VALIDATE .+ update (\d+) cost ", line)
                if match:
                    progress(plan, root, cell_id, int(match.group(1)))
            code = proc.wait()
        finally:
            if proc.poll() is None:
                (out.parent/"STOP_AFTER_UPDATE").touch()
                proc.wait()
    with (root/"resource_release.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(cell=cell_id, exit_code=code, free_ram_gib=h.free_ram_gib()))+"\n")
    return code


def report(root, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_convergence as pc
    summaries, series, validation = [], {}, []
    all_complete = all((root/"cells/probe"/c["id"]/"complete.json").exists()
                       for c in plan["stages"]["probe"])
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#24659e", "#cb8526", "#278673", "#ba4858"]
    for color, cell in zip(colors, plan["stages"]["probe"]):
        path = root / "cells/probe" / cell["id"]
        if not (path/"episodes.csv").exists():
            continue
        rows = list(h.rows(path/"episodes.csv"))
        if [int(r["episode"]) for r in rows] != list(range(1, len(rows)+1)):
            raise ValueError("Incomplete or reordered episode rows")
        y = [float(r["mean_training_reward"]) for r in rows]
        x = [int(r["episode"]) for r in rows]
        smooth = trailing_mean(y)
        series[cell["id"]] = dict(x=x, y=y, smooth=smooth, color=color)
        label = "%s  %.0e / %.0e" % (LABELS[cell["settings"]["method"]], *pc.LEARNING_RATES[cell["settings"]["lr"]-1])
        ax.plot(x, y, color=color, alpha=.15, linewidth=.7)
        ax.plot(x, smooth, color=color, label=label, linewidth=2)
        vals = [r for r in h.rows(path/"validation.csv") if r["method"] == cell["settings"]["method"]]
        updates = sorted({int(r["update"]) for r in vals})
        for update in updates:
            group = [r for r in vals if int(r["update"]) == update]
            validation.append(dict(cell=cell["id"], method=cell["settings"]["method"], lr=cell["settings"]["lr"],
                update=update, completed_episodes=update*128/6000,
                validation_reward=-pc.validation_mean(group, "mean_task_cost", "pooled_task_cost")))
        last = [r for r in vals if int(r["update"]) in updates[-3:]]
        selected = h.read_json(path/"selection.json") if (path/"selection.json").exists() else None
        summaries.append(dict(cell=cell["id"], method=cell["settings"]["method"], lr=cell["settings"]["lr"],
            completed_episodes=len(rows), tail_validation_cost=pc.validation_mean(last, "mean_task_cost", "pooled_task_cost"),
            tail_validation_qos=pc.validation_mean(last, "qos_rate", "pooled_task_cost"),
            initial_validation_cost=pc.validation_mean([r for r in vals if int(r["update"]) == 0], "mean_task_cost", "pooled_task_cost"),
            best_checkpoint_update=selected["update"] if selected else None))
    ax.set(xlabel="Complete training episodes (6000 slots each)", ylabel="Mean per-slot training reward",
           title="Four from-scratch curves | trailing 50 episodes | raw traces faint")
    ax.grid(alpha=.2)
    ax.legend(title="Actor / Critic learning rates", fontsize=9)
    fig.tight_layout()
    fig.savefig(root/"convergence_50episode.png", dpi=180)
    if series and all_complete:
        chosen = min((r for r in summaries if r["method"] == "TMPPO"),
                     key=lambda r: r["tail_validation_cost"], default=None)
        if chosen:
            center = max(1, (chosen["best_checkpoint_update"] or 0)*128//6000)
            end = min(min(len(s["x"]) for s in series.values()), max(50, center+25))
            start = max(1, end-49)
            ax.set_xlim(start, end)
            ax.set_title("Exploratory common-window zoom | full curves retained separately")
            fig.savefig(root/"convergence_common_zoom.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    for cell_id, data in series.items():
        ax.plot(data["x"], data["y"], color=data["color"], label=cell_id, linewidth=1.)
    ax.set(xlabel="Complete training episodes (6000 slots each)", ylabel="Mean per-slot training reward",
           title="All raw episode rewards | no smoothing or omitted interval")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(root/"convergence_raw.png", dpi=180)
    plt.close(fig)
    pc.csv_write(root/"summary.csv", summaries)
    pc.csv_write(root/"validation_points.csv", validation)
    fig, ax = plt.subplots(figsize=(12, 6))
    for color, cell in zip(colors, plan["stages"]["probe"]):
        rows = [r for r in validation if r["cell"] == cell["id"]]
        ax.plot([r["completed_episodes"] for r in rows], [r["validation_reward"] for r in rows],
                marker="o", markersize=3, color=color, label=cell["id"])
    ax.set(xlabel="Completed training episodes", ylabel="Independent validation reward (- task cost)",
           title="Independent validation | raw points | no 50-episode smoothing")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(root/"validation_reward.png", dpi=180)
    plt.close(fig)
    best_lr = {m: min((r for r in summaries if r["method"] == m),
                      key=lambda r: r["tail_validation_cost"], default=None) for m in METHODS}
    h.write_json(root/"learning_rate_selection.json", dict(best_by_method=best_lr,
        criterion=plan["learning_rate_selection"], automatic_proof_of_TMAPO_superiority=False,
        preliminary_only=not all_complete,
        heldout_test_used=False, single_training_seed=True,
        smoothing_note="Before episode50 the trailing average uses all available episodes; no cropping of full plot"))


def preview(root, plan, run_id, attempt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snapshots = {}
    for path in (root/"progress_receipts").glob("*.json"):
        item = h.read_json(path)
        if item.get("plan_hash") != plan["plan_hash"] or item.get("run_id") != run_id or item.get("attempt") != attempt:
            raise ValueError("Unmatched live progress origin")
        cell = item["cell"]
        if cell not in {c["id"] for c in plan["stages"]["probe"]}:
            raise ValueError("Unregistered live cell")
        if item["validated_update"] % 375 or len(item["episodes"]) != item["completed_episodes"]:
            raise ValueError("Invalid progress boundary")
        if cell not in snapshots or item["validated_update"] > snapshots[cell]["validated_update"]:
            snapshots[cell] = item
    if len(snapshots) != 4 or min(s["completed_episodes"] for s in snapshots.values()) == 0:
        print("Waiting for all four actual episode prefixes"); return
    count = min(s["completed_episodes"] for s in snapshots.values())
    fig, ax = plt.subplots(figsize=(12, 6))
    for color, cell in zip(("#24659e", "#cb8526", "#278673", "#ba4858"), plan["stages"]["probe"]):
        rows = snapshots[cell["id"]]["episodes"][:count]
        assert [int(r["episode"]) for r in rows] == list(range(1, count+1))
        assert all(int(r["env_steps"]) == int(r["episode"])*6000 for r in rows)
        y = [float(r["mean_training_reward"]) for r in rows]
        ax.plot(range(1, count+1), y, color=color, alpha=.15, linewidth=.7)
        ax.plot(range(1, count+1), trailing_mean(y), color=color, label=cell["id"], linewidth=2)
    ax.set(xlabel="Complete training episodes (common prefix)", ylabel="Mean per-slot training reward",
           title="LIVE, %d common episodes | trailing50, shorter prefix before50 | not final audit" % count)
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(root/"live_convergence_50episode.png", dpi=180)
    plt.close(fig)
    h.write_json(root/"preview_manifest.json", dict(run_id=run_id, attempt=attempt,
        common_complete_episodes=count, final_audit=False, learning_rate_selected=False))


def audit(root, plan):
    import torch
    import paper_convergence as pc
    verdict = h.gate(plan, root, "probe")
    initial = None
    for cell in plan["stages"]["probe"]:
        path = root/"cells/probe"/cell["id"]
        try:
            cfg, train = pc.configuration(cell["settings"])
            manifest = h.read_json(path/"manifest.json")
            if manifest["environment"] != cfg or manifest["training"] != train:
                raise ValueError("Full executed configuration differs")
            episodes = list(h.rows(path/"episodes.csv"))
            if len(episodes) != 128 or any(int(r["env_steps"]) != int(r["episode"])*6000 or int(r["episode_steps"]) != 6000 for r in episodes):
                raise ValueError("Not exactly128 complete episodes")
            for r in episodes:
                h.close(float(r["mean_training_reward"])*6000, float(r["episode_reward_sum"]))
                scale = cfg["users_per_region"]*3*cfg["arrival_per_user_s"]*cfg["dt_s"]
                h.close(float(r["episode_reward_sum"]), -float(r["physical_cost_total"])/scale)
            state = torch.load(path/"initial_checkpoint.pt", map_location="cpu", weights_only=False)
            if state["actor_initialization_contract"]["mode"] != "xavier_small_head_v1":
                raise ValueError("Wrong actor initialization")
            if initial is None:
                initial = state["actors"]
            elif any(not torch.equal(v, state["actors"][k]) for k, v in initial.items()):
                raise ValueError("Initial actors differ across learning-rate/method cells")
        except (OSError, ValueError, KeyError) as error:
            verdict["errors"].append(cell["id"]+": "+str(error))
    verdict["passed"] = verdict["passed"] and not verdict["errors"]
    verdict.update(complete_episode_audit=True, from_scratch_protocol=True, publication_grade=False)
    h.write_json(root/"four_curve_audit.json", verdict)
    return verdict["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "prepare", "worker", "cell", "publish", "audit", "report", "preview"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--tag")
    parser.add_argument("--cell")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.command == "init":
        print(json.dumps(initialize(root, args.verification))); return
    if args.command == "prepare":
        match = TAG.fullmatch(args.tag or "")
        if not match or not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo or ""):
            raise ValueError("Invalid explicit four-curve repo/tag")
        root.mkdir(parents=True, exist_ok=True)
        h.command(["gh", "release", "download", "fourcurves-campaign-"+match.group(1),
                   "--repo", args.repo, "--pattern", "plan.json", "--dir", root])
        load_plan(root); return
    plan = load_plan(root)
    if args.command == "preview":
        preview(root, plan, args.run_id, args.attempt); return
    if args.command == "report":
        report(root, plan); return
    if args.command == "audit":
        raise SystemExit(0 if audit(root, plan) else 1)
    if not args.execute:
        parser.error("Explicit --execute required for training or publication")
    if args.command == "publish":
        h.publish_cloud_output(plan, root, args.repo, "probe", args.cell, args.run_id, args.attempt); return
    if args.command == "worker":
        raise SystemExit(worker(root, plan, args.cell))
    cell = next(c for c in plan["stages"]["probe"] if c["id"] == args.cell)
    import paper_convergence as pc
    path = root/"cells/probe"/cell["id"]
    try:
        ok = pc.train_cell(path, cell["settings"], wall_seconds=plan["runtime"]["worker_wall_s"])
    except Exception as error:
        if path.exists():
            h.write_json(path/"failure.json", dict(type=type(error).__name__, error=str(error)))
        raise
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

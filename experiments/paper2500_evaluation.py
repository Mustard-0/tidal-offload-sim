"""Paired frozen-model studies; no evaluation result changes training settings."""
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import re
import statistics
import time
import zipfile

import hybrid_campaign as h

TEST_SEEDS = (9848031, 9848047, 9848063, 9848089)
SCALE_USERS = (30, 45, 60, 75, 90)


def evaluation_tasks(stage, models):
    result = []
    if stage == "scaling":
        for users in SCALE_USERS:
            for model in models:
                methods = [model["method"]]
                if model["method"] == "TMPPO":
                    methods += ["TMAPO-NoAmp", "TMAPO-NoWait"]
                for method in methods:
                    result.append(dict(kind="evaluation", stage=stage, method=method, users=users,
                                       model=model, test_seeds=list(TEST_SEEDS)))
            for method in ("Greedy", "Fixed-GeoI+Greedy", "Local-Only"):
                result.append(dict(kind="evaluation", stage=stage, method=method, users=users,
                                   model=None, test_seeds=list(TEST_SEEDS)))
    elif stage == "sensitivity_evaluation":
        result = [dict(kind="evaluation", stage=stage, method="TMPPO", users=60,
                       pair=int(re.search(r"_p([012])w[0-4]$", m["task"]["id"]).group(1)),
                       model=m, test_seeds=list(TEST_SEEDS)) for m in models]
    else:
        raise ValueError("Unknown evaluation stage")
    for i, task in enumerate(result):
        task["id"] = f"{stage}_{i:04d}"
    return result


def validate_evaluation_task(task):
    if task["test_seeds"] != list(TEST_SEEDS) or task["users"] not in SCALE_USERS:
        raise ValueError("Unregistered test workload")
    if task["stage"] == "scaling":
        allowed = {"TMPPO", "IPPO", "NoTidal", "MARATD3-Adapted", "MATORA-DQN-Adapted",
                   "TMAPO-NoAmp", "TMAPO-NoWait", "Greedy", "Fixed-GeoI+Greedy", "Local-Only"}
        if task["method"] not in allowed:
            raise ValueError("Unregistered scale comparator")
    elif task["stage"] != "sensitivity_evaluation" or task["users"] != 60 or task["pair"] not in (0, 1, 2):
        raise ValueError("Unregistered evaluation stage")
    if task["model"]:
        from paper2500_campaign import validate_request
        source_task = task["model"]["task"]
        if task["model"]["seed"] != source_task["settings"]["seed"]:
            raise ValueError("Selected training seed identity differs")
        validate_request(dict(plan_hash="validation", runtime=dict(maximum_segments_per_task=48)),
            dict(plan_hash="validation", entries=[dict(task=source_task, segment=1)]))


def model_files(root, p, model):
    import paper2500_campaign as c
    ref = model["asset"]
    release = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{ref['release']}")
    asset = next(a for a in release["assets"] if a["name"] == ref["name"])
    if asset["digest"] != ref["digest"]:
        raise ValueError("Selected model changed")
    out = root/"model_inputs"/str(asset["id"])
    out.mkdir(parents=True, exist_ok=True)
    archive = out/ref["name"]
    if not archive.exists():
        c.gh("release", "download", ref["release"], "--repo", c.REPO,
             "--pattern", ref["name"], "--dir", out)
    if archive.stat().st_size != asset["size"] or "sha256:"+h.file_hash(archive) != ref["digest"]:
        raise ValueError("Model asset digest/size mismatch")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Model asset CRC failure")
        origin = json.loads(z.read("model_origin.json"))
        if origin != dict(plan_hash=p["plan_hash"], task_id=model["task"]["id"], commit=p["commit"], test_used=False):
            raise ValueError("Wrong selected model origin")
    data = out/"data"
    if not data.exists():
        h.extract_evidence(archive, data)
    return data


class MechanismPolicy:
    """No-WAIT ablation, explicit rather than hidden remapping."""
    is_rule_based = True

    def __init__(self, policy, mode):
        self.policy, self.mode, self.env = policy, mode, None
        self.policy.deterministic = True

    def begin(self, *args):
        self.policy.begin(*args)

    def act(self, obs, mask, stage, agents):
        import numpy as np
        masks = mask.copy()
        if self.mode == "TMAPO-NoWait" and stage == "user":
            masks[:, 0] = False
        return self.policy.act(obs, masks, stage, agents)


def run_evaluation_task(root, p, task, path):
    import torch
    import paper_convergence as pc
    import paper2500_campaign as c
    from cross_slot_wlo_train import CrossSlotRule
    from privacy import local_epsilon
    validate_evaluation_task(task)
    path.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_hash=p["plan_hash"], task=task, source_hashes=h.code_hashes(),
                    warmup_steps=3000, measurement_steps=30000, phase_s=0.,
                    checkpoint_frozen_before_test=True, no_test_selection=True)
    c.immutable(path/"evaluation_manifest.json", identity)
    if (path/"failure.json").exists():
        raise ValueError("Do not resume a failed evaluation")
    if (path/"complete.json").exists():
        return "complete"
    torch.set_num_threads(1)
    model = task["model"]
    if model:
        data = model_files(root, p, model)
        s = model["task"]["settings"]
        cfg, train = pc.configuration(s)
        original = h.read_json(data/"manifest.json")
        if original["environment"] != cfg or original["training"] != train:
            raise ValueError("Model configuration changed")
        cfg.update(users_per_region=task["users"]//3,
                   epsilon_req=task.get("epsilon", cfg["epsilon_req"]))
        policy = pc.learner_for(s["method"], train, cfg)
        policy.load(torch.load(data/"best_checkpoint.pt", map_location="cpu", weights_only=False))
        checkpoint_hash = h.file_hash(data/"best_checkpoint.pt")
        if task["method"] == "TMAPO-NoWait":
            policy = MechanismPolicy(policy, task["method"])
    else:
        cfg, train = pc.configuration(c.settings("TMPPO", 1, 87021))
        cfg["users_per_region"] = task["users"]//3
        rule = "strict_local_only" if task["method"] == "Local-Only" else "local_preferred"
        policy = CrossSlotRule(rule, cfg)
        checkpoint_hash = None
    if cfg["project_public_region"]:
        raise ValueError("Raw unprojected QL is required")
    no_amp = task["method"] in ("Fixed-GeoI+Greedy", "TMAPO-NoAmp")
    started = time.monotonic()
    completed = []
    for seed in task["test_seeds"]:
        case = path/f"seed{seed}"
        if (case/"complete.json").exists():
            prior = h.read_json(case/"complete.json")
            if prior["task"] != task or prior["checkpoint_hash"] != checkpoint_hash:
                raise ValueError("Existing test cohort origin differs")
            for name, checksum in prior["raw_sha256"].items():
                if h.file_hash(case/name) != checksum:
                    raise ValueError("Existing test cohort raw data changed")
            completed.append(seed)
            continue
        if time.monotonic()-started > p["runtime"]["segment_wall_s"] or pc.free_ram_gib() < 1.5:
            h.write_json(path/"paused.json", dict(reason="Evaluation task boundary", completed_test_seeds=completed))
            return "paused"
        if case.exists():
            raise ValueError("Unknown interrupted evaluation cohort; preserve it, no blind rerun")
        case.mkdir()
        metrics, env = pc.evaluate(policy, cfg, seed, 0., 30000, 3000, collect=True,
                                  environment_method="TMAPO-NoAmp" if no_amp else "TMAPO")
        if task["method"] == "Local-Only" and (metrics["published"] or any(r["node"] >= env.u for r in env.rows)):
            raise AssertionError("Local-Only executed a remote task")
        for batch in env.batch_rows:
            n = batch["n_distinct"]
            if abs(batch["epsilon_loc_per_m"]-local_epsilon(cfg, n, no_amp)) > 1e-12:
                raise AssertionError("Actual batch/accountant mismatch")
        pc.csv_write(case/"tasks.csv", env.rows)
        pc.csv_write(case/"batches.csv", env.batch_rows)
        h.write_json(case/"metrics.json", metrics)
        h.write_json(case/"complete.json", dict(task=task, test_seed=seed,
            checkpoint_hash=checkpoint_hash, config=cfg, metrics=metrics,
            raw_sha256={f.name: h.file_hash(f) for f in case.glob("*.csv")},
            dp_proven=False, ql_is_attack_error=False, full_300_second_measurement=True))
        completed.append(seed)
    h.write_json(path/"complete.json", dict(test_seeds=completed, task=task, config=cfg,
        checkpoint_hash=checkpoint_hash, evaluation_only=True, test_selected=False))
    if (path/"paused.json").exists():
        (path/"paused.json").unlink()
    return "complete"


def report(root, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import paper_convergence as pc
    from four_episode_curves import trailing_mean
    report_dir = root/"report"
    report_dir.mkdir(exist_ok=True)
    registry = root/"registries/convergence.json"
    if registry.exists():
        fig, ax = plt.subplots(figsize=(12, 6))
        raw_fig, raw_ax = plt.subplots(figsize=(12, 6))
        val_fig, val_ax = plt.subplots(figsize=(12, 6))
        validation_points = []
        for index, (method, lr) in enumerate((m, lr) for m in ("TMPPO", "IPPO") for lr in (1, 2)):
            traces = []
            for task in h.read_json(registry)["tasks"]:
                s = task["settings"]
                path = root/"cells"/task["id"]/"episodes.csv"
                if s["method"] == method and s["lr"] == lr and path.exists():
                    traces.append([float(r["mean_training_reward"]) for r in pc.read_csv(path)])
            if not traces:
                continue
            length = min(map(len, traces))
            data = np.asarray([trailing_mean(t[:length], 50) for t in traces])
            x = np.arange(1, length+1)
            mean = data.mean(0)
            color = ("#24659e", "#cb8526", "#278673", "#ba4858")[index]
            label = ("TMAPO" if method == "TMPPO" else method)+f" LR{lr}, n={len(traces)}"
            raw_ax.plot(x, np.asarray([t[:length] for t in traces]).mean(0), color=color,
                        linewidth=.8, alpha=.8, label=label)
            ax.plot(x[:49], mean[:49], ":", color=color, alpha=.6)
            ax.plot(x[49:], mean[49:], color=color, label=label)
            if len(traces) >= 3:
                std = data.std(0, ddof=1)
                ax.fill_between(x, mean-std, mean+std, color=color, alpha=.12)
            points = defaultdict(list)
            for task in h.read_json(registry)["tasks"]:
                s = task["settings"]
                val_path = root/"cells"/task["id"]/"validation.csv"
                if s["method"] != method or s["lr"] != lr or not val_path.exists():
                    continue
                values = [r for r in pc.read_csv(val_path) if r["method"] == method]
                for update in sorted({int(r["update"]) for r in values}):
                    subset = [r for r in values if int(r["update"]) == update]
                    cost = pc.validation_mean(subset, "mean_task_cost", "pooled_task_cost")
                    episode = pc.interaction_steps(s, update)/6000
                    points[episode].append(-cost)
                    validation_points.append(dict(method=method, lr=lr, training_seed=s["seed"],
                        update=update, complete_episode_equivalent=episode, pooled_validation_cost=cost))
            vx = sorted(points)
            val_ax.plot(vx, [statistics.mean(points[e]) for e in vx], color=color, marker="o",
                        markersize=3, label=label)
        ax.set(xlabel="Complete episodes (6000 slots each)", ylabel="Mean per-slot reward",
               title="50 complete-episode trailing mean | first49 dotted | no cropped intervals")
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(report_dir/"convergence_50episode.png", dpi=180)
        plt.close(fig)
        for other_fig, other_ax, name, ylabel, title in (
                (raw_fig, raw_ax, "convergence_raw.png", "Raw episode mean reward", "All complete episodes; unsmoothed"),
                (val_fig, val_ax, "validation_reward.png", "Negative pooled validation task cost", "Independent validation; no training smoothing")):
            other_ax.set(xlabel="Complete episodes (6000 slots each)", ylabel=ylabel, title=title)
            other_ax.grid(alpha=.2)
            if other_ax.get_legend_handles_labels()[0]:
                other_ax.legend(fontsize=9)
            other_fig.tight_layout()
            other_fig.savefig(report_dir/name, dpi=180)
            plt.close(other_fig)
        pc.csv_write(report_dir/"validation_points.csv", validation_points)
    metrics_rows = []
    paired = defaultdict(set)
    for cell in (root/"cells").glob("*"):
        if ".cloud_segment" in cell.name or not (cell/"evaluation_manifest.json").exists():
            continue
        task = h.read_json(cell/"evaluation_manifest.json")["task"]
        for case in cell.glob("seed*/complete.json"):
            result = h.read_json(case)
            for name, checksum in result["raw_sha256"].items():
                if h.file_hash(case.parent/name) != checksum:
                    raise ValueError("Raw evaluation evidence changed")
            m = result["metrics"]
            training_seed = task["model"]["seed"] if task["model"] else None
            row = dict(stage=task["stage"], method=task["method"], users=task["users"],
                pair=task.get("pair"),
                epsilon=task.get("epsilon", result["config"]["epsilon_req"]),
                training_seed=training_seed, test_seed=result["test_seed"],
                time_weight=result["config"]["weight_time"], energy_weight=result["config"]["weight_energy"],
                quality_weight=result["config"]["weight_quality"], **m)
            metrics_rows.append(row)
            paired[(task["users"], result["test_seed"])].add(m["trace_hash"])
    pc.csv_write(report_dir/"evaluation_cohorts.csv", metrics_rows)
    grouped = defaultdict(list)
    for row in metrics_rows:
        key = tuple(row[k] for k in ("stage", "method", "users", "epsilon", "pair", "time_weight", "energy_weight", "quality_weight"))
        grouped[key].append(row)
    aggregate = []
    for key, rows in grouped.items():
        item = dict(zip(("stage", "method", "users", "epsilon", "pair", "time_weight", "energy_weight", "quality_weight"), key))
        for metric in ("delay_s", "energy_j_per_arrival", "qos_rate", "ql_m", "publication_ratio", "report_weighted_batch_n", "amplified_publication_ratio"):
            item[metric] = pc.validation_mean(rows, metric, "pooled_task_cost")
        item.update(test_trajectories=len({r["test_seed"] for r in rows}),
                    training_seeds=len({r["training_seed"] for r in rows if r["training_seed"] is not None}))
        aggregate.append(item)
    pc.csv_write(report_dir/"summary.csv", aggregate)
    h.write_json(report_dir/"evaluation_audit.json", dict(paired_workloads=all(len(v) == 1 for v in paired.values()),
        complete_cohorts=len(metrics_rows), privacy_study="Separate lightweight C1 mechanism comparison",
        dp_proven=False, ql_is_attack_error=False, no_test_selection=True))
    for stage, horizontal in (("scaling", "users"),):
        data = [r for r in aggregate if r["stage"] == stage]
        if not data:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for method in sorted({r["method"] for r in data}):
            scopes = [None]
            for users in scopes:
                subset = [r for r in data if r["method"] == method and (users is None or r["users"] == users)]
                xs = sorted({r[horizontal] for r in subset})
                for ax, key in zip(axes, ("delay_s", "energy_j_per_arrival", "qos_rate")):
                    ys = [statistics.mean(float(r[key]) for r in subset if r[horizontal] == x and r[key] is not None) for x in xs]
                    ax.plot(xs, ys, marker="o", markersize=3, label=method+(f" N{users}" if users else ""))
                    ax.set(xlabel=horizontal, ylabel=key)
                    ax.grid(alpha=.2)
        axes[0].legend(fontsize=5, ncol=2)
        fig.suptitle("Paired frozen-model evaluation; adapted methods are not author reproductions")
        fig.tight_layout()
        fig.savefig(report_dir/(stage+".png"), dpi=180)
        plt.close(fig)
    if metrics_rows and not all(len(v) == 1 for v in paired.values()):
        raise ValueError("Evaluation workloads do not pair")
    pairs = (("time_weight", "energy_weight", "delay_s", "energy_j_per_arrival", .2, .12),
             ("time_weight", "quality_weight", "delay_s", "ql_m", .2, 10000.),
             ("energy_weight", "quality_weight", "energy_j_per_arrival", "ql_m", .12, 10000.))
    data = [r for r in aggregate if r["stage"] == "sensitivity_evaluation"]
    if data:
        from paper_four_studies import full_config
        cfg, _ = full_config()
        references = dict(delay_s=cfg["time_ref_s"], energy_j_per_arrival=cfg["energy_ref_j"], ql_m=cfg["quality_ref_m"])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, (first, second, a, b, _, _) in enumerate(pairs):
            rows = sorted((r for r in data if r["pair"] == i), key=lambda r: r[first])
            for metric, color in ((a, "#24659e"), (b, "#bd543f")):
                valid = [r for r in rows if r[metric] is not None]
                axes[i].plot([r[first] for r in valid], [r[metric]/references[metric] for r in valid],
                             marker="o", color=color, label=metric+" / fixed reference")
            axes[i].set(xlabel=first+" (other weight = 0.6 - x)", ylabel="Normalized physical metric",
                        title=first+" vs "+second)
            axes[i].grid(alpha=.2)
            axes[i].legend(fontsize=7)
        fig.suptitle("Separately trained objective-weight trade-offs; intersections are not imposed")
        fig.tight_layout()
        fig.savefig(report_dir/"sensitivity.png", dpi=180)
        plt.close(fig)

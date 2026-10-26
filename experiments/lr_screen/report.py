"""Plot ONLY recorded formal raw CSV. No synthetic or empty result curves."""
import json
from pathlib import Path
from .runtime import np, csv_write, json_write, read_csv

KEYS = ("mean_task_cost", "delay_s", "energy_j_per_arrival", "qos_rate")
LABELS = ("Validation cost / arrival", "Delay / completed task (s)",
          "Energy / arrival (J)", "QoS violations / arrivals")


def points(rows):
    result = []
    for x in sorted({int(r["env_steps"]) for r in rows}):
        group = [r for r in rows if int(r["env_steps"]) == x]
        row = dict(env_steps=x)
        for key in KEYS:
            denominator = "completed" if key == "delay_s" else "arrivals"
            valid = [r for r in group if r.get(key) not in ("", None)]
            n = sum(int(r[denominator]) for r in valid)
            row[key] = sum(float(r[key])*int(r[denominator]) for r in valid)/n if n else None
        result.append(row)
    return result


def stability(curve, train, rules):
    tail = curve[-rules["window"]:]
    y = np.asarray([r["mean_task_cost"] for r in tail])
    x = np.asarray([r["env_steps"] for r in tail], dtype=float)
    mean, sd = float(y.mean()), float(y.std(ddof=1)) if len(y)>1 else 0.
    cv = sd/max(abs(mean), 1e-12)
    drift = float(np.polyfit(x, y, 1)[0]*(x[-1]-x[0])/max(abs(mean), 1e-12)) if len(x)>1 else 0.
    qos = float(np.mean([r["qos_rate"] for r in tail]))
    finite = all(np.isfinite(float(r[k])) for r in train for k in ("kl", "entropy", "value_loss", "mean_training_reward"))
    stable = (len(curve) >= rules["minimum_validation_points"] and finite and cv <= rules["max_tail_cv"]
        and abs(drift) <= rules["max_relative_linear_drift"]
        and mean <= rules["max_tail_cost_over_initial"]*curve[0]["mean_task_cost"]
        and qos <= curve[0]["qos_rate"]+rules["max_tail_qos_increase"])
    return dict(tail_mean=mean, tail_sd=sd, tail_cv=cv, tail_relative_drift=drift,
                tail_qos=qos, finite_diagnostics=finite, stable_screen=bool(stable), lr_score=mean+sd,
                convergence_statement="末段通过低波动筛查，仍不能证明收敛" if stable else "尚未观察到稳定收敛")


def report(root):
    root = Path(root)
    p = json.loads((root/"plan.json").read_text(encoding="utf-8"))
    statuses, results, curves, training = [], [], {}, {}
    signatures = set()
    paired = {}
    for cell in p["cells"]:
        out = root/cell["id"]
        status = dict(cell)
        status["state"] = cell.get("state", "not_started")
        for marker, state in (("status.json", "started"), ("paused.json", "paused"),
                               ("failure.json", "failed"), ("complete.json", "completed")):
            if (out/marker).exists():
                status["state"] = state
        status["started"] = (out/"manifest.json").exists()
        status["complete"] = (out/"complete.json").exists()
        statuses.append(status)
        if not (out/"validation.csv").exists():
            continue
        m = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        if m["component_only"] or not m["transfer_integrated"]:
            raise ValueError("Component diagnostics are NOT formal LR evidence")
        signatures.add(json.dumps(m["source_hashes"], sort_keys=True))
        if m["environment"] != p["environment"]:
            raise ValueError("Physical/reward configuration differs from frozen plan")
        if m["training"]["updates"]*m["training"]["rollout_steps"] != cell["env_steps"]:
            raise ValueError("Cell uses a different interaction budget")
        if status["complete"]:
            done = json.loads((out/"complete.json").read_text(encoding="utf-8"))
            if done["env_steps"] != cell["env_steps"]:
                raise ValueError("Incomplete budget mislabeled as completed")
        val = read_csv(out/"validation.csv")
        train = read_csv(out/"train.csv") if (out/"train.csv").exists() else []
        curve = points(val)
        curves[cell["id"]], training[cell["id"]] = curve, train
        for row in val:
            key = (row["env_steps"], row["validation_seed"], row["phase_s"])
            paired.setdefault(key, set()).add(row["trace_hash"])
        summary = dict(cell, **stability(curve, train, p["stability"]))
        best = min(curve, key=lambda r: r["mean_task_cost"])
        summary.update(best_env_steps=best["env_steps"], best_validation_cost=best["mean_task_cost"],
                       initial_validation_cost=curve[0]["mean_task_cost"], actual_status=status["state"])
        results.append(summary)
        csv_write(root/"curves"/(cell["id"]+".csv"), curve)
    csv_write(root/"status.csv", statuses)
    if not curves:
        # Keep blocked/not-started distinction from the draft rather than
        # producing a blank chart that looks like experimental output.
        json_write(root/"report_status.json", dict(state="no_formal_training_data", curves_generated=False,
                   missing="formal training results", recommendations=None))
        return
    if len(signatures) > 1 or any(len(v)>1 for v in paired.values()):
        raise ValueError("Mixed source revisions or unpaired external task traces")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for method in ("TMAPO", "IPPO", "PPO"):
        if not any(cell["id"] in curves for cell in p["cells"] if cell["algorithm"] == method):
            continue
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for cell in p["cells"]:
            if cell["algorithm"] != method or cell["id"] not in curves:
                continue
            curve = curves[cell["id"]]
            label = f"A={cell['actor_lr']:g}, C={cell['critic_lr']:g}"
            for key, ax in zip(KEYS, axes):
                ax.plot([r["env_steps"] for r in curve], [r[key] for r in curve], marker=".", label=label)
        for ax, label in zip(axes, LABELS):
            ax.set(xlabel="Cumulative training environment slots", ylabel=label)
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(method+" | raw validation curves, one training seed")
        fig.tight_layout()
        fig.savefig(root/f"{method}_learning_rates.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        for cell in p['cells']:
            if cell['algorithm'] != method or cell['id'] not in training:
                continue
            raw = training[cell['id']]
            ax.plot([int(r['env_steps']) for r in raw],
                    [float(r['mean_training_reward']) for r in raw],
                    linewidth=.8, label=f"A={cell['actor_lr']:g}, C={cell['critic_lr']:g}")
        ax.set(xlabel='Cumulative training environment slots', ylabel='Mean accrued team reward / slot',
               title=method+' | raw rollout means (128 slots), no smoothing')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root/f'{method}_training_returns.png', dpi=180)
        plt.close(fig)
    selections = []
    for method in ("TMAPO", "IPPO", "PPO"):
        complete = [r for r in results if r["algorithm"] == method and r["actual_status"] == "completed"]
        expected = sum(r["algorithm"] == method for r in p["cells"])
        if len(complete) != expected:
            continue
        stable = [r for r in complete if r["stable_screen"]]
        selected = min(stable or complete, key=lambda r: (r["lr_score"], r["lr_index"]))
        selections.append(dict(selected, selection_provisional=not bool(stable)))
    csv_write(root/"summary.csv", results)
    json_write(root/"selected.json", selections)
    if len(selections) == 3:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for selection in selections:
            curve = curves[selection["id"]]
            for key, ax in zip(KEYS, axes):
                ax.plot([r["env_steps"] for r in curve], [r[key] for r in curve], marker=".", label=selection["id"])
        for ax, label in zip(axes, LABELS):
            ax.set(xlabel="Cumulative training environment slots", ylabel=label)
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.suptitle("Validation-selected learning rates | raw data incl. initial policy | one seed")
        fig.tight_layout()
        fig.savefig(root/"selected_convergence.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9,4))
        for chosen in selections:
            raw = training[chosen['id']]
            ax.plot([int(r['env_steps']) for r in raw], [float(r['mean_training_reward']) for r in raw],
                    linewidth=.8, label=chosen['id'])
        ax.set(xlabel='Cumulative training environment slots', ylabel='Mean accrued team reward / slot',
               title='Validation-selected configurations | raw rollout returns, no smoothing')
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(root/'selected_training_returns.png', dpi=180)
        plt.close(fig)
    json_write(root/"report_status.json", dict(formal_cells_seen=len(curves), expected=11,
               selections_available=len(selections), all_complete=all(r["state"] == "completed" for r in statuses),
               smoothing="none", test_set_used=False, multi_seed_evidence=False))

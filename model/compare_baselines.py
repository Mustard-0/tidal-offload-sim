"""Audit and compare one-seed baseline pilots on paired held-out traces."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import read_csv, read_json, save_json, write_csv
from plots import COLORS, LABELS, style


MAIN_METHODS = ("TMAPO", "IPPO", "MARATD3-Adapted", "DPFL-Adapted", "Greedy", "Local-Only",
                "TMAPO-NoPred", "TMAPO-NoWait")
METRICS = ("delay_s", "energy_j_per_arrival", "qos_rate", "completion_rate",
           "mean_task_cost", "ql_m", "publication_ratio",
           "report_weighted_batch_n", "amplified_publication_ratio")


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def load_job(run, method, users, updates):
    plan = read_json(run / "plan.json")
    candidates = [j for j in plan["jobs"] if j["method"] == method
                  and j["users_per_region"] == users and j["epsilon_req"] == 1]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {method} job at M={users} in {run}")
    job = candidates[0]
    path = run / job["id"]
    if read_json(path / "manifest.json")["source_hashes"] != read_json(run / "source_hashes.json"):
        raise ValueError(f"Job source snapshot differs from its run: {path}")
    complete = read_json(path / "complete.json")
    if complete["job"] != job or complete["updates"] != (0 if method in ("Greedy", "Local-Only") else updates):
        raise ValueError(f"Incomplete or mismatched job: {path}")
    if complete["evaluation"] != "short pilot, not final paper evidence":
        raise ValueError(f"Unexpected evaluation protocol: {path}")
    if complete["selected_update"]:
        train = read_csv(path / "train_raw.csv")
        if [int(r["update"]) for r in train] != list(range(1, updates + 1)):
            raise ValueError(f"Missing training updates: {path}")
        val = read_csv(path / "validation_raw.csv")
        by_update = defaultdict(list)
        for row in val:
            by_update[int(row["update"])].append(number(row, "mean_task_cost"))
        expected_seeds = len(read_json(run / "config.json")["training"]["validation_seeds"])
        if any(len(costs) != expected_seeds for costs in by_update.values()):
            raise ValueError(f"Incomplete validation checkpoint: {path}")
        best = min(by_update, key=lambda u: float(np.mean(by_update[u])))
        if best != complete["selected_update"]:
            raise ValueError(f"Selected checkpoint was not the validation minimum: {path}")
    elif method not in ("Greedy", "Local-Only"):
        raise ValueError(f"Learned method has no selected checkpoint: {path}")
    tests = read_csv(path / "test_raw.csv")
    if len(tests) != 5 or len({int(r["trace_seed"]) for r in tests}) != 5:
        raise ValueError(f"Expected five independent held-out traces: {path}")
    tasks = read_csv(path / "test_tasks.csv")
    by_seed = defaultdict(list)
    for row in tasks:
        by_seed[int(row["trace_seed"])].append(row)
    for row in tests:
        seed = int(row["trace_seed"])
        cohort = by_seed[seed]
        arrivals = int(row["arrivals"])
        if len(cohort) != arrivals or int(row["completed"]) + int(row["failed"]) != arrivals:
            raise ValueError(f"Arrival or terminal-task count mismatch: {path}, trace {seed}")
        checks = {
            "completion_rate": sum(int(x["completed"]) for x in cohort) / arrivals,
            "qos_rate": sum(int(x["qos"]) for x in cohort) / arrivals,
            "energy_j_per_arrival": sum(float(x["energy_j"]) for x in cohort) / arrivals,
        }
        for key, calculated in checks.items():
            if not np.isclose(calculated, number(row, key), rtol=1e-8, atol=1e-10):
                raise ValueError(f"Task-level {key} does not match summary: {path}, trace {seed}")
        if method == "Local-Only" and (int(row["published"]) != 0 or number(row, "ql_m") is not None):
            raise ValueError("Local-Only unexpectedly published location data")
    return job, {int(row["trace_seed"]): row for row in tests}, complete


def compare(main_run, noamp_run=None, output=None, users=100, updates=100):
    main_run = Path(main_run)
    noamp_run = Path(noamp_run) if noamp_run else None
    output = Path(output) if output else main_run / "baseline_audit"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    main_config = read_json(main_run / "config.json")
    if users != main_config["environment"]["users_per_region"]:
        raise ValueError("Requested user count differs from saved run configuration")
    run_sources = read_json(main_run / "source_hashes.json")
    if noamp_run:
        if read_json(noamp_run / "config.json") != main_config:
            raise ValueError("NoAmp and main runs have different saved configurations")
        if read_json(noamp_run / "source_hashes.json") != run_sources:
            raise ValueError("NoAmp and main runs have different source snapshots")
    method_runs = [(m, main_run) for m in MAIN_METHODS]
    if noamp_run:
        method_runs.append(("TMAPO-NoAmp", noamp_run))
    loaded = {method: load_job(run, method, users, updates) for method, run in method_runs}
    reference = loaded["TMAPO"][1]
    expected_seeds = set(main_config["training"]["test_seeds"])
    if set(reference) != expected_seeds:
        raise ValueError("Unexpected held-out test seeds")
    for method, (_, tests, _) in loaded.items():
        if set(tests) != expected_seeds:
            raise ValueError(f"Unpaired test seeds for {method}")
        for seed in expected_seeds:
            first, other = reference[seed], tests[seed]
            for key in ("trace_hash", "phase_s", "window_s", "arrivals"):
                if first[key] != other[key]:
                    raise ValueError(f"Unpaired {key}: {method}, trace {seed}")
    output.mkdir(parents=True)
    summary, pairs = [], []
    for method, (_, tests, complete) in loaded.items():
        row = {"method": method, "users_per_region": users, "training_seed": loaded[method][0]["seed"],
               "selected_update": complete["selected_update"], "test_traces": len(tests),
               "total_arrivals": sum(int(tests[s]["arrivals"]) for s in expected_seeds)}
        for metric in METRICS:
            vals = [number(tests[s], metric) for s in sorted(expected_seeds)]
            finite = [v for v in vals if v is not None]
            row[metric] = float(np.mean(finite)) if finite else None
            row[metric + "_trace_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
            diffs = [number(tests[s], metric) - number(reference[s], metric)
                     for s in sorted(expected_seeds)
                     if number(tests[s], metric) is not None and number(reference[s], metric) is not None]
            pairs.append({"method": method, "metric": metric, "n_paired_traces": len(diffs),
                          "mean_minus_tmppo": float(np.mean(diffs)) if diffs else None,
                          "paired_trace_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None})
        summary.append(row)
    write_csv(output / "comparison.csv", summary)
    write_csv(output / "paired_differences.csv", pairs)
    save_json(output / "audit.json", {
        "passed": True, "main_run": str(main_run.resolve()),
        "noamp_run": str(noamp_run.resolve()) if noamp_run else None,
        "source_hashes": run_sources, "methods": list(loaded),
        "training_seed_count": 1, "held_out_trace_count": len(expected_seeds),
        "evaluation_window_s": float(next(iter(reference.values()))["window_s"]),
        "scope": "Single-training-seed short-window pilot, not full-tide or multi-seed evidence",
        "uncertainty": "Trace SD and paired trace SD do not quantify training-seed uncertainty",
    })
    plot_main(summary, output)
    return output


def plot_main(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style()
    data = {r["method"]: r for r in summary}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    specs = (("delay_s", "Completed-task delay (s)", 1),
             ("energy_j_per_arrival", "System energy (J/arrival)", 1),
             ("qos_rate", "QoS violation (%)", 100))
    for ax, (metric, title, scale) in zip(axes, specs):
        for i, method in enumerate(MAIN_METHODS):
            value = data[method][metric]
            ax.bar(i, value * scale if value is not None else np.nan,
                   color=COLORS[method], width=0.72)
        ax.set_xticks(range(len(MAIN_METHODS)), [LABELS[m] for m in MAIN_METHODS], rotation=38, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Baseline comparison at 100 users per region (single-seed pilot)")
    fig.tight_layout()
    fig.savefig(output / "baseline_M100.png", dpi=220)
    fig.savefig(output / "baseline_M100.svg")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-run", type=Path, required=True)
    p.add_argument("--noamp-run", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--updates", type=int, default=100)
    a = p.parse_args()
    print(compare(a.main_run, a.noamp_run, a.out, a.users, a.updates))


if __name__ == "__main__":
    main()

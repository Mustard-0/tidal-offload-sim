"""Verify a completed per-region scaling run and report paired raw metrics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import read_csv, read_json, save_json, write_csv


METRICS = ("delay_s", "energy_j_per_arrival", "qos_rate", "completion_rate", "mean_task_cost")
NONTRAINED = {"Greedy", "Local-Only"}


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def audit(run, output=None, allow_subset=False):
    run = Path(run)
    plan = read_json(run / "plan.json")
    config = read_json(run / "config.json")
    if plan["kind"] != "main":
        raise ValueError("Expected a user-scale main run")
    methods = config["experiments"]["main_methods"]
    registered_scales = config["experiments"]["users_per_region"]
    actual = {(j["method"], j["users_per_region"]) for j in plan["jobs"]}
    scales = sorted({n for _, n in actual}) if allow_subset else registered_scales
    if not set(scales).issubset(registered_scales):
        raise ValueError("An unregistered user scale was used")
    expected = {(m, n) for m in methods for n in scales}
    if actual != expected or len(plan["jobs"]) != len(expected):
        raise ValueError("Incomplete or duplicated method-by-scale plan")
    source_hashes = read_json(run / "source_hashes.json")
    eval_label = "full-cycle" if plan["formal_eval"] else "short pilot, not final paper evidence"
    test_seeds = set(config["training"]["test_seeds"])
    validation_seeds = set(config["training"]["validation_seeds"])
    rows = []
    for job in plan["jobs"]:
        path = run / job["id"]
        if read_json(path / "manifest.json")["source_hashes"] != source_hashes:
            raise ValueError(f"Source mismatch: {path}")
        complete = read_json(path / "complete.json")
        if complete["job"] != job or complete["evaluation"] != eval_label:
            raise ValueError(f"Job or evaluation mismatch: {path}")
        required_updates = 0 if job["method"] in NONTRAINED else config["training"]["updates"]
        if complete["updates"] != required_updates:
            raise ValueError(f"Training budget mismatch: {path}")
        if job["method"] not in NONTRAINED:
            train = read_csv(path / "train_raw.csv")
            expected_updates = required_updates
            if [int(r["update"]) for r in train] != list(range(1, expected_updates + 1)):
                raise ValueError(f"Training updates missing: {path}")
            val = defaultdict(dict)
            for item in read_csv(path / "validation_raw.csv"):
                val[int(item["update"])][int(item["trace_seed"])] = number(item, "mean_task_cost")
            if any(set(values) != validation_seeds for values in val.values()):
                raise ValueError(f"Validation traces missing: {path}")
            best = min(val, key=lambda u: np.mean(list(val[u].values())))
            if complete["selected_update"] != best:
                raise ValueError(f"Test-selected or incorrect checkpoint: {path}")
        elif complete["selected_update"] != 0:
            raise ValueError(f"Untrained policy has a training checkpoint: {path}")
        tests = {int(r["trace_seed"]): r for r in read_csv(path / "test_raw.csv")}
        if set(tests) != test_seeds:
            raise ValueError(f"Held-out test traces missing: {path}")
        tasks = defaultdict(list)
        for task in read_csv(path / "test_tasks.csv"):
            tasks[int(task["trace_seed"])].append(task)
        for seed, test in tests.items():
            cohort = tasks[seed]
            arrivals = int(test["arrivals"])
            if len(cohort) != arrivals or int(test["completed"]) + int(test["failed"]) != arrivals:
                raise ValueError(f"Task accounting failed: {path}, seed {seed}")
            measured = {
                "completion_rate": sum(int(t["completed"]) for t in cohort) / arrivals,
                "qos_rate": sum(int(t["qos"]) for t in cohort) / arrivals,
                "energy_j_per_arrival": sum(float(t["energy_j"]) for t in cohort) / arrivals,
            }
            completed = [t for t in cohort if int(t["completed"])]
            measured["delay_s"] = (np.mean([float(t["end_s"]) - float(t["arrival_s"]) for t in completed])
                                   if completed else None)
            for metric, value in measured.items():
                reported = number(test, metric)
                if value is None:
                    if reported is not None:
                        raise ValueError(f"Undefined {metric} was filled in: {path}")
                elif not np.isclose(value, reported, rtol=1e-7, atol=1e-9):
                    raise ValueError(f"Task-level {metric} mismatch: {path}, seed {seed}")
            if job["method"] == "Local-Only" and (int(test["published"]) != 0 or number(test, "ql_m") is not None):
                raise ValueError("Local-only unexpectedly publishes location")
            rows.append({"method": job["method"], "users_per_region": job["users_per_region"],
                         "training_seed": job["seed"], "selected_update": complete["selected_update"], **test})
    grouped = {(m, n): {int(r["trace_seed"]): r for r in rows if r["method"] == m and r["users_per_region"] == n}
               for m, n in expected}
    for n in scales:
        reference = grouped[("TMAPO", n)]
        for method in methods:
            for seed in test_seeds:
                a, b = reference[seed], grouped[(method, n)][seed]
                for field in ("trace_hash", "phase_s", "window_s", "arrivals"):
                    if a[field] != b[field]:
                        raise ValueError(f"Unpaired {field}: {method}, M={n}, seed={seed}")
    summary, paired = [], []
    for n in scales:
        for method in methods:
            tests = grouped[(method, n)]
            row = {"method": method, "users_per_region": n, "training_seed_count": 1,
                   "held_out_trace_count": len(tests)}
            for metric in METRICS:
                values = [number(tests[s], metric) for s in sorted(test_seeds)]
                finite = [v for v in values if v is not None]
                row[metric] = float(np.mean(finite)) if finite else None
                row[metric + "_trace_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
                for seed in sorted(test_seeds):
                    a = number(tests[seed], metric)
                    b = number(grouped[("TMAPO", n)][seed], metric)
                    paired.append({"method": method, "users_per_region": n, "trace_seed": seed,
                                   "metric": metric, "minus_tmapo": None if a is None or b is None else a - b})
            summary.append(row)
    output = Path(output) if output else run / "scaling_audit"
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_trace_differences.csv", paired)
    save_json(output / "audit.json", {"passed": True, "methods": methods, "users_per_region": scales,
              "complete_registered_grid": scales == registered_scales,
              "source_hashes": source_hashes, "evaluation": eval_label,
              "training_seeds": 1, "test_traces": len(test_seeds),
              "caution": "Trace variability is not training-seed uncertainty; DPFL gradient noise is not a formal privacy guarantee."})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-subset", action="store_true", help="Audit a complete method grid at selected scales only")
    args = parser.parse_args()
    print(audit(args.run, args.out, args.allow_subset))

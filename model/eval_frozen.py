"""Evaluate a validation-selected v74 checkpoint using its own saved configuration."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from common import read_json, save_json, source_hashes, write_csv, seed_all
from environment import GreedyPolicy
from evaluation import evaluate
from learners import make_policy

CORE = ("environment.py", "physics.py", "privacy.py", "workload.py", "learners.py", "evaluation.py")


def checkpoint_protocol(checkpoint):
    manifest = read_json(checkpoint.parent / "manifest.json")
    current = source_hashes()
    for name in CORE:
        if manifest["source_hashes"].get(name) != current[name]:
            raise ValueError(f"Checkpoint belongs to another implementation: {name}; use its saved source")
    return manifest["config"], manifest["job"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--formal-eval", action="store_true")
    parser.add_argument("--compare-greedy", action="store_true")
    args = parser.parse_args()
    config, job = checkpoint_protocol(args.checkpoint)
    seed_all(job["seed"], config["training"]["torch_threads"])
    policy = make_policy(job["method"], config, args.device)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy.load(saved["policy"])
    args.out.mkdir(parents=True, exist_ok=False)
    seeds = config["training"]["test_seeds"]
    policies = [(job["method"], policy)]
    if args.compare_greedy:
        policies.append(("Greedy", GreedyPolicy(config["environment"])))
    hashes = {}
    for method, actor in policies:
        rows, tasks, batches = evaluate(actor, config, method, seeds, args.formal_eval, collect=True)
        write_csv(args.out / f"{method}_test_raw.csv", rows)
        if tasks:
            write_csv(args.out / f"{method}_test_tasks.csv", tasks)
        if batches:
            write_csv(args.out / f"{method}_test_batches.csv", batches)
        for row in rows:
            hashes.setdefault(row["trace_seed"], set()).add(row["trace_hash"])
    if any(len(values) != 1 for values in hashes.values()):
        raise RuntimeError("Frozen methods used different external workloads")
    save_json(args.out / "complete.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "selected_update": saved["update"], "config": config, "job": job,
        "source_hashes": source_hashes(), "test_seeds": seeds,
        "evaluation": "full-cycle" if args.formal_eval else "phase-stratified pilot",
        "training_updates_performed": 0,
    })
    print(args.out.resolve())


if __name__ == "__main__":
    main()

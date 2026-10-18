"""Stage-gated follow-up experiments; dry-run unless --execute is supplied."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT, read_json


STAGES = ("value-screen", "convergence-screen", "baseline-tuning", "privacy-screen", "formal-main")
CORE_FILES = ("environment.py", "physics.py", "privacy.py", "workload.py", "evaluation.py", "learners.py", "run.py")


def verify_pilot(pilot):
    pilot = Path(pilot).resolve()
    plan = read_json(pilot / "plan.json")
    audit_path = pilot / "scaling_audit" / "audit.json"
    if not audit_path.is_file():
        raise ValueError(f"Scaling pilot has not passed raw-data audit yet: {audit_path}")
    audit = read_json(audit_path)
    if plan["kind"] != "main" or not audit["passed"] or not audit["complete_registered_grid"]:
        raise ValueError("The complete 40-job scaling pilot must pass its raw-data audit first")
    if (pilot / "failure.json").exists():
        raise ValueError("Scaling pilot has a failure marker")
    if len(plan["jobs"]) != 40 or any(not (pilot / j["id"] / "complete.json").exists() for j in plan["jobs"]):
        raise ValueError("Scaling pilot is incomplete")
    for name in CORE_FILES:
        if (pilot / "source" / name).read_bytes() != (ROOT / name).read_bytes():
            raise ValueError(f"Core model changed since pilot: {name}")
    pilot_config = read_json(pilot / "config.json")
    follow_config = read_json(ROOT / "config.json")
    for section in ("environment", "experiments"):
        if pilot_config[section] != follow_config[section]:
            raise ValueError(f"Protocol changed since pilot: {section}")
    for key, value in follow_config["training"].items():
        if key != "updates" and pilot_config["training"][key] != value:
            raise ValueError(f"Training protocol changed since pilot: {key}")
    return pilot


def verify_value(run):
    run = Path(run)
    check = read_json(run / "check.json")
    scope = read_json(run / "scope.json")
    if not check["passed"] or not (run / "value_table.csv").is_file() or scope["states"] < 1:
        raise ValueError("Location-value screen did not complete")


def verify_convergence(run):
    run = Path(run)
    selection = read_json(run / "selection.json")
    if not {"TMAPO", "IPPO"}.issubset(selection):
        raise ValueError("Both learned methods require validation-selected learning rates")
    plan = read_json(run / "plan.json")
    expected = 8 if plan["kind"] == "baseline-tuning" else 4
    if plan["kind"] not in ("convergence", "baseline-tuning") or len(plan["jobs"]) != expected:
        raise ValueError("Expected four algorithm-by-learning-rate configurations")
    if any(not (run / j["id"] / "complete.json").exists() for j in plan["jobs"]):
        raise ValueError("Convergence run has missing configurations")
    return run / "selection.json"


def command_for(args, output):
    command = [sys.executable, str(ROOT / "run.py")]
    if args.stage == "value-screen":
        command += ["value", "--states", str(args.states), "--repeats", str(args.repeats),
                    "--samples", str(args.samples)]
    else:
        kind = {"convergence-screen": "convergence", "baseline-tuning": "baseline-tuning", "privacy-screen": "privacy",
                "formal-main": "main"}[args.stage]
        command += [kind, "--updates", str(args.updates), "--device", args.device]
        if args.stage in ("privacy-screen", "formal-main"):
            if args.convergence_run is None:
                raise ValueError("Privacy/main experiments require --convergence-run")
            command += ["--selection", str(Path(args.convergence_run) / "selection.json")]
            if getattr(args, "baseline_run", None):
                command += ["--baseline-selection", str(Path(args.baseline_run) / "selection.json")]
        if args.stage == "formal-main":
            command += ["--formal-eval"]
        if args.max_seconds:
            command += ["--max-seconds", str(args.max_seconds)]
    command += ["--seed", str(args.seed)]
    command += ["--resume", str(output)] if args.resume else ["--out", str(output)]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--value-run", type=Path)
    parser.add_argument("--convergence-run", type=Path)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    if min(args.updates, args.states, args.repeats, args.samples) <= 0:
        parser.error("updates, states, repeats and samples must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    if args.resume and args.out is None:
        parser.error("--resume requires the existing --out directory")
    if args.stage == "convergence-screen" and args.value_run is None:
        parser.error("Convergence is scheduled after --value-run")
    if args.out:
        output = args.out.resolve()
    else:
        output = ROOT / "runs" / (args.stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if args.execute:
        verify_pilot(args.pilot_run)
        if args.stage == "convergence-screen":
            verify_value(args.value_run)
        if args.stage in ("privacy-screen", "formal-main"):
            verify_convergence(args.convergence_run)
            if args.baseline_run:
                verify_convergence(args.baseline_run)
        if output.exists() != args.resume:
            raise FileExistsError("Use a new --out directory, or --resume on an existing run")
    command = command_for(args, output)
    print(json.dumps({"stage": args.stage, "execute": args.execute, "out": str(output),
                      "command": command}, ensure_ascii=False, indent=2), flush=True)
    if args.execute:
        env = os.environ.copy()
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        print(output, flush=True)


if __name__ == "__main__":
    main()

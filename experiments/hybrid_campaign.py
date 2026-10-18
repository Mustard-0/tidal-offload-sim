"""Prepare and coordinate disjoint local/GitHub convergence cells.

Standard library only in the coordinator. Training is opt-in (--execute).
The tag-triggered workflow works on a development commit without editing main.
Code verification is required before freezing any training campaign.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from atomic_io import replace_file

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("TMPPO", "IPPO", "MARATD3-Adapted")
SEEDS = (87021, 87035, 87049)
CAMPAIGN = re.compile(r"paper-campaign-\d{8}T\d{6}Z\Z")
STAGE_TAG = re.compile(r"paper-(screen|formal)-(\d{8}T\d{6}Z)\Z")
RUNTIME = dict(local_workers=2, github_workers=8, torch_threads=1,
               python="3.9.25", torch="2.5.1+cpu", numpy="2.0.2",
               scipy="1.13.1", matplotlib="3.9.2",
               start_ram_gib=1.5, pause_ram_gib=.5,
               wall_seconds_screen=3000, wall_seconds_formal=15000)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace_file(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path, text=False):
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def code_hashes():
    paths = list((ROOT/"model").glob("*.py")) + list((ROOT/"experiments").glob("*.py"))
    paths += [ROOT/p for p in ("config.json", "requirements.txt",
                              "experiments/PAPER_FOUR_STUDIES_PROTOCOL.md",
                              ".github/workflows/paper_hybrid.yml")]
    paths += list((ROOT/".github/workflows").glob("*long_probe*.yml"))
    paths += list((ROOT/".github/workflows").glob("*learning_contract*.yml"))
    if (ROOT/"experiments/LONG_PROBE_2000_PROTOCOL.md").exists():
        paths.append(ROOT/"experiments/LONG_PROBE_2000_PROTOCOL.md")
    if (ROOT/"experiments/LEARNING_CONTRACT_PROTOCOL.md").exists():
        paths.append(ROOT/"experiments/LEARNING_CONTRACT_PROTOCOL.md")
    if (ROOT/"experiments/FOUR_CURVE_PROTOCOL.md").exists():
        paths.append(ROOT/"experiments/FOUR_CURVE_PROTOCOL.md")
    return {p.relative_to(ROOT).as_posix(): file_hash(p, True) for p in sorted(paths)}


def command(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, cwd=ROOT,
                                   encoding="utf-8", **kwargs).strip()


def settings(stage, method, lr, seed):
    return dict(method=method, lr=lr, seed=seed, updates=64 if stage == "screen" else 2000,
                rollout=128, episode_steps=2048 if stage == "screen" else 6000,
                phase_count=5, validate_every=16 if stage == "screen" else 100,
                validation_steps=500 if stage == "screen" else 1000, warmup_steps=300,
                validation_seeds=[814001, 814019], validation_phases=[30., 130., 230.],
                training_profile="credit_v3", checkpoint_every=25, schedule_updates=2000,
                policy_diagnostics=True,
                validation_objective="pooled_task_cost",
                baseline_diagnostics=method == "TMPPO" and lr == 1 and seed == SEEDS[0])


def build_plan(campaign, commit, hashes):
    if not CAMPAIGN.fullmatch(campaign):
        raise ValueError("Expected paper-campaign-YYYYMMDDTHHMMSSZ")
    stages = {}
    for stage in ("screen", "formal"):
        stages[stage] = [dict(id="%s_LR%d_seed%d" % (method, lr, seed),
                              owner="local" if method == "TMPPO" else "github",
                              settings=settings(stage, method, lr, seed))
                         for method in METHODS for lr in (1, 2)
                         for seed in (SEEDS[:1] if stage == "screen" else SEEDS)]
    plan = dict(schema="hybrid-convergence-v1", campaign=campaign, commit=commit,
                source_hashes=hashes, stages=stages,
                runtime=dict(RUNTIME),
                gate=dict(tail_points=3, max_cost_ratio=1.25, max_qos_increase=.05,
                          max_tail_qos=.10,
                          require_all_cells=True, require_tmppo_win=False),
                heldout_test_seeds=[9848031, 9848047, 9848063, 9848089],
                final_test_executed=False, claim="exploratory; no complete-transcript DP proof",
                subsequent_studies="not scheduled; ATO-SLA and matched-publication privacy audit pending")
    plan["plan_hash"] = digest(plan)
    return plan


def load_plan(path):
    plan = read_json(path)
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if digest(payload) != plan["plan_hash"] or not CAMPAIGN.fullmatch(plan["campaign"]):
        raise ValueError("Invalid campaign identity/hash")
    canonical = build_plan(plan["campaign"], plan["commit"], plan["source_hashes"])
    if any(plan.get(key) != value for key, value in canonical.items() if key != "plan_hash"):
        raise ValueError("Plan budgets, ownership or criteria differ from the reviewed implementation")
    verification = plan.get("verification", {})
    if not verification.get("passed") or verification.get("source_hashes") != plan["source_hashes"]:
        raise ValueError("Missing matching code verification")
    if code_hashes() != plan["source_hashes"]:
        raise ValueError("Frozen source differs; create a new reviewed campaign")
    if command(["git", "rev-parse", "HEAD"]) != plan["commit"]:
        raise ValueError("Checkout commit differs from frozen campaign")
    return plan


def require_runtime(plan):
    import importlib.metadata
    if platform.python_version() != plan["runtime"]["python"]:
        raise ValueError("Use the frozen Python version: " + plan["runtime"]["python"])
    for package in ("torch", "numpy", "scipy", "matplotlib"):
        if importlib.metadata.version(package) != plan["runtime"][package]:
            raise ValueError("Runtime package differs from frozen plan: " + package)


def initialize(out, verification):
    verified = read_json(verification)
    if not verified.get("passed") or verified.get("source_hashes") != code_hashes():
        raise ValueError("Run verify-code on this exact source before freezing a training campaign")
    changed = command(["git", "status", "--porcelain", "--", "model", "experiments",
                       "config.json", "requirements.txt", ".github/workflows/paper_hybrid.yml"])
    if changed:
        raise ValueError("Commit reviewed source on a development branch before freezing the campaign")
    branch = command(["git", "branch", "--show-current"])
    if branch in ("main", "master") or not branch:
        raise ValueError("Freeze on a named development branch")
    plan = build_plan("paper-campaign-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                      command(["git", "rev-parse", "HEAD"]), code_hashes())
    plan["verification"] = verified
    plan["plan_hash"] = digest({k: v for k, v in plan.items() if k != "plan_hash"})
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"campaign.json", plan)
    return plan


def verify_code(out):
    require_runtime(dict(runtime=RUNTIME))
    out.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, source_hashes=code_hashes(), checks=[],
                  scope="unit, syntax and physical preflight only; no formal cells")
    commands = [
        [sys.executable, "-m", "py_compile", "experiments/hybrid_campaign.py",
         "experiments/paper_convergence.py", "model/cross_slot_learners.py"],
        [sys.executable, "-m", "unittest", "discover", "-s", "model", "-p", "test_*.py"],
        [sys.executable, "-m", "unittest", "discover", "-s", "experiments", "-p", "test_*.py"],
        [sys.executable, "experiments/paper_four_studies.py", "preflight"]]
    try:
        for index, args in enumerate(commands):
            path = out/("check_%02d.log" % index)
            with path.open("w", encoding="utf-8") as log:
                code = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=600).returncode
            result["checks"].append(dict(command=args[1:], exit_code=code, log_sha256=file_hash(path)))
            if code:
                raise RuntimeError("Verification failed; inspect " + str(path))
        result["passed"] = code_hashes() == result["source_hashes"]
        if not result["passed"]:
            raise ValueError("Source changed while verification was running")
    finally:
        write_json(out/"verification.json", result)


def free_ram_gib():
    if os.name != "nt":
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(fields["MemAvailable"].split()[0]) / 1024**2
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "free", "page_total", "page_free", "virtual_total", "virtual_free", "extended")]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot measure RAM")
    return status.free / 2**30


@contextmanager
def awake_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root/"coordinator.lock"
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    awake = False
    try:
        os.write(fd, json.dumps(dict(pid=os.getpid())).encode())
        if os.name == "nt":
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            if not awake:
                raise OSError("Cannot prevent idle sleep")
        yield
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        os.close(fd)
        lock.unlink()


def worker_command(cell, out, plan, stage):
    args = [sys.executable, "-u", str(ROOT/"experiments/paper_convergence.py"), "cell",
            "--out", str(out), "--wall-seconds", str(plan["runtime"]["wall_seconds_" + stage])]
    for key, value in cell["settings"].items():
        if key in ("validation_seeds", "validation_phases"):
            continue
        if isinstance(value, bool):
            if value:
                args.append("--" + key.replace("_", "-"))
            elif key == "baseline_diagnostics":
                args.append("--no-baseline-diagnostics")
        else:
            args += ["--" + key.replace("_", "-"), str(value)]
    if stage == "formal":
        args.append("--allow-long-cell")
    if out.exists():
        if not (out/"paused.json").exists() or (out/"failure.json").exists():
            raise ValueError("Existing cell is not a clean pause: " + str(out))
        args.append("--resume")
    return args


def run_owned(plan, root, stage, owner, only=None):
    require_runtime(plan)
    cells = [c for c in plan["stages"][stage] if c["owner"] == owner and
             (only is None or c["id"] == only)]
    if not cells:
        raise ValueError("No cells for requested owner/id")
    if owner == "github" and len(cells) != 1:
        raise ValueError("A GitHub job runs exactly one cell")
    directory = root/"cells"/stage
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"STOP_AFTER_UPDATE").exists():
        raise ValueError("Inspect and archive the stage stop marker before resuming")
    pending = [c for c in cells if not (directory/c["id"]/"complete.json").exists()]
    running = []
    limit = plan["runtime"]["local_workers"] if owner == "local" else 1
    stopped = False
    logdir = root/"logs"/stage
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    try:
        while pending or running:
            if (directory/"STOP_AFTER_UPDATE").exists():
                stopped = True
                pending.clear()
            while pending and len(running) < limit and not stopped:
                if free_ram_gib() < plan["runtime"]["start_ram_gib"]:
                    if not running:
                        write_json(root/(stage+"_resource_wait.json"), dict(free_ram_gib=free_ram_gib()))
                        return False
                    break
                cell = pending.pop(0)
                out = directory/cell["id"]
                args = worker_command(cell, out, plan, stage)
                stdout = (logdir/(cell["id"]+".log")).open("a", encoding="utf-8")
                stderr = (logdir/(cell["id"]+".err.log")).open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(args, cwd=ROOT, stdout=stdout, stderr=stderr, env=env)
                except BaseException:
                    stdout.close()
                    stderr.close()
                    raise
                running.append((cell, process, stdout, stderr))
                print("START", stage, cell["id"], "pid", process.pid, flush=True)
            for item in list(running):
                cell, process, stdout, stderr = item
                status = process.poll()
                if status is None:
                    continue
                stdout.close()
                stderr.close()
                running.remove(item)
                with (root/"resource_release.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(stage=stage, cell=cell["id"], exit_code=status,
                                                 free_ram_gib=free_ram_gib())) + "\n")
                if status:
                    stopped = True
                    pending.clear()
                    (directory/"STOP_AFTER_UPDATE").touch(exist_ok=True)
            if running:
                time.sleep(2)
        return not stopped
    finally:
        if running:
            (directory/"STOP_AFTER_UPDATE").touch(exist_ok=True)
            for _, process, stdout, stderr in running:
                process.wait()
                stdout.close()
                stderr.close()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError("Raw metric mismatch: %r vs %r" % (actual, expected))


def audit_validation(path, cfg, validation, method):
    """Stream measured task rows; no final test data or model unpickling."""
    stats = defaultdict(lambda: dict(n=0, completed=0, qos=0, energy=0., cost=0.,
                                    published=0, quality=0., delay=0., ids=set(), inputs=[]))
    for row in rows(path):
        if row["method"] != method:
            continue
        key = (int(row["seed"]), float(row["phase_s"]))
        s = stats[key]
        task_id = int(row["task_id"])
        if task_id in s["ids"]:
            raise ValueError("Duplicate task")
        s["ids"].add(task_id)
        s["n"] += 1
        complete = int(row["completed"])
        published = int(row["published"])
        qos = int(row["qos"])
        delay = float(row["end_s"]) - float(row["arrival_s"])
        energy = float(row["energy_j"])
        close(energy, sum(float(row[k]) for k in ("transmit_energy_j", "compute_energy_j", "report_energy_j")))
        quality = float(row["ql_m"]) if published else 0.
        if (min(delay, energy, quality) < 0 or complete not in (0, 1) or
                qos not in (0, 1) or published not in (0, 1)):
            raise ValueError("Invalid physical task metric")
        s["completed"] += complete
        s["qos"] += qos
        s["published"] += published
        s["energy"] += energy
        s["quality"] += quality
        s["delay"] += delay * complete
        s["cost"] += (cfg["weight_time"]*delay/cfg["time_ref_s"] +
                      cfg["weight_energy"]*energy/cfg["energy_ref_j"] +
                      cfg["weight_quality"]*quality/cfg["quality_ref_m"] + cfg["weight_violation"]*qos)
        slot = round(float(row["arrival_s"])/cfg["dt_s"])
        close(float(row["arrival_s"]), slot*cfg["dt_s"])
        s["inputs"].append((task_id, int(row["user"]), int(row["region"]), slot,
                            row["task_profile"], float(row["task_bits"]), float(row["task_cycles"]),
                            float(row["deadline_s"]), round(float(row["arrival_tide_factor"]), 12)))
    identities = {}
    for row in validation:
        key = (int(row["seed"]), float(row["phase_s"]))
        s = stats[key]
        n = s["n"]
        if not n or n != int(row["arrivals"]) or s["completed"] != int(row["completed"]):
            raise ValueError("Task cohort incomplete")
        close(n, int(row["completed"]) + int(row["failed"]))
        for field, value in (("mean_task_cost", s["cost"]/n), ("qos_rate", s["qos"]/n),
                             ("energy_j_per_arrival", s["energy"]/n),
                             ("publication_ratio", s["published"]/n)):
            close(float(row[field]), value)
        if s["completed"]:
            close(float(row["delay_s"]), s["delay"]/s["completed"])
        if s["published"]:
            close(float(row["ql_m"]), s["quality"]/s["published"])
        identities[str(key)] = digest(sorted(s["inputs"]))
    return identities


def gate(plan, root, stage="screen"):
    verdict = dict(plan_hash=plan["plan_hash"], stage=stage, passed=False,
                   errors=[], cells=[], evidence={}, test_used=False, dp_proven=False)
    paired = defaultdict(set)
    environments = set()
    trainings = set()
    expected_ids = {c["id"] for c in plan["stages"][stage]}
    extra = [p.name for p in (root/"cells"/stage).glob("*")
             if p.is_dir() and p.name not in expected_ids]
    if extra:
        verdict["errors"].append("Unregistered cells: " + ", ".join(sorted(extra)))
    for cell in plan["stages"][stage]:
        path = root/"cells"/stage/cell["id"]
        try:
            manifest = read_json(path/"manifest.json")
            complete = read_json(path/"complete.json")
            expected = cell["settings"]
            if manifest["settings"] != expected or complete["updates"] != expected["updates"]:
                raise ValueError("Wrong settings or incomplete budget")
            if complete["env_steps"] != expected["updates"]*expected["rollout"]:
                raise ValueError("Completed interaction budget differs")
            for checkpoint in ("latest.pt", "best_checkpoint.pt"):
                with zipfile.ZipFile(path/checkpoint) as archive:
                    if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                        raise ValueError("Missing serialized checkpoint payload")
                    if archive.testzip() is not None:
                        raise ValueError("Corrupt checkpoint archive")
            if manifest["source_commit"] != plan["commit"]:
                raise ValueError("Worker ran a different Git commit")
            runtime = manifest["runtime"]
            if runtime.get("python_version") != plan["runtime"]["python"] or any(
                    runtime.get(k) != plan["runtime"][k] for k in ("torch", "numpy", "scipy", "matplotlib")):
                raise ValueError("Worker dependency versions differ")
            if runtime.get("device") != "cpu" or runtime.get("torch_threads") != 1:
                raise ValueError("Unregistered device/thread setting")
            if (path/"failure.json").exists() or (path/"paused.json").exists():
                raise ValueError("Cell failed/paused")
            expected_sources = {k: v for k, v in plan["source_hashes"].items()
                                if k.startswith(("model/", "experiments/")) or k == "config.json"}
            if manifest["source_hashes"] != expected_sources:
                raise ValueError("Manifest source differs from campaign")
            for name, checksum in expected_sources.items():
                if file_hash(path/"source"/name, True) != checksum:
                    raise ValueError("Frozen snapshot modified")
            environments.add(digest(manifest["environment"]))
            train_config = dict(manifest["training"])
            actor, critic = ((1e-4, 3e-4), (3e-4, 1e-3))[expected["lr"]-1]
            if train_config.pop("actor_lr") != actor or train_config.pop("critic_lr") != critic:
                raise ValueError("Learning rate differs from registered candidate")
            if train_config.get("local_credit_fraction") != 0. or train_config.get(
                    "td3_discrete_objective") != "categorical_expectation":
                raise ValueError("Worker did not use the declared team_v2 profile")
            if expected.get("training_profile") == "credit_v3" and (
                    train_config.get("critic_clock_features") != 2 or
                    train_config.get("actor_warmup_steps") != 2048):
                raise ValueError("Worker omitted the finite-horizon critic or actor burn-in")
            trainings.add(digest(train_config))
            train = list(rows(path/"train.csv"))
            if [int(r["update"]) for r in train] != list(range(1, expected["updates"]+1)):
                raise ValueError("Missing or duplicate training updates")
            for row in train:
                for value in row.values():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(numeric):
                        raise ValueError("Nonfinite training data")
                if int(row["env_steps"]) != int(row["update"])*expected["rollout"]:
                    raise ValueError("Interaction budget mismatch")
            val = [r for r in rows(path/"validation.csv") if r["method"] == expected["method"]]
            updates = [0] + list(range(expected["validate_every"], expected["updates"]+1,
                                      expected["validate_every"]))
            if updates[-1] != expected["updates"]:
                updates.append(expected["updates"])
            grid = {(u, seed, phase) for u in updates for seed in expected["validation_seeds"]
                    for phase in expected["validation_phases"]}
            observed = [(int(r["update"]), int(r["seed"]), float(r["phase_s"])) for r in val]
            if len(observed) != len(grid) or set(observed) != grid:
                raise ValueError("Validation grid differs")
            costs, qos = [], []
            for update in updates:
                subset = [r for r in val if int(r["update"]) == update]
                identities = audit_validation(path/"validation_traces"/("tasks_%06d.csv" % update),
                                              manifest["environment"], subset, expected["method"])
                for key, identity in identities.items():
                    paired[key].add(identity)
                if expected.get("validation_objective") == "pooled_task_cost":
                    count = sum(int(r["arrivals"]) for r in subset)
                    costs.append(sum(float(r["mean_task_cost"])*int(r["arrivals"]) for r in subset)/count)
                    qos.append(sum(float(r["qos_rate"])*int(r["arrivals"]) for r in subset)/count)
                else:
                    costs.append(statistics.mean(float(r["mean_task_cost"]) for r in subset))
                    qos.append(statistics.mean(float(r["qos_rate"]) for r in subset))
            selected = read_json(path/"selection.json")
            best_index = min(range(len(costs)), key=lambda i: costs[i])
            if selected["update"] != updates[best_index] or selected["test_used"]:
                raise ValueError("Checkpoint was not selected by the declared validation rule")
            close(float(selected["validation_cost"]), costs[best_index])
            tail = plan["gate"]["tail_points"]
            ratio = statistics.mean(costs[-tail:])/max(costs[0], 1e-12)
            tail_qos = statistics.mean(qos[-tail:])
            growth = tail_qos - qos[0]
            stable = (ratio <= plan["gate"]["max_cost_ratio"] and
                      growth <= plan["gate"]["max_qos_increase"] and
                      tail_qos <= plan["gate"]["max_tail_qos"])
            verdict["cells"].append(dict(id=cell["id"], tail_cost_ratio=ratio,
                tail_qos=tail_qos, tail_qos_increase=growth, stable=stable, best_update=selected["update"],
                improves_initialization=selected["update"] > 0))
            if stage == "screen" and not stable:
                verdict["errors"].append(cell["id"] + ": predeclared instability threshold exceeded")
            for artifact in path.rglob("*"):
                if artifact.is_file() and not artifact.name.endswith(".tmp"):
                    verdict["evidence"][artifact.relative_to(root).as_posix()] = file_hash(artifact)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
            verdict["errors"].append(cell["id"] + ": " + str(exc))
    if len(environments) != 1 or len(trainings) != 1 or not paired or any(len(v) != 1 for v in paired.values()):
        verdict["errors"].append("Physical configs or exogenous validation workloads differ")
    verdict["passed"] = not verdict["errors"] and len(verdict["cells"]) == len(plan["stages"][stage])
    return verdict


def bundle_screen(plan, root):
    verdict = gate(plan, root)
    write_json(root/"screen_gate.json", verdict)
    if not verdict["passed"]:
        raise ValueError("Screen gate failed; preserve evidence, no automatic extension")
    target = root/"screen_evidence.zip"
    if target.exists():
        raise FileExistsError("Evidence bundle is immutable")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["campaign.json", "screen_gate.json"] + sorted(verdict["evidence"]):
            archive.write(root/name, name)
    return target


def extract_evidence(archive, destination):
    destination = destination.resolve()
    # Windows' normal path limit can be exceeded by nested source snapshots.
    filesystem_destination = destination
    if os.name == "nt" and not str(destination).startswith("\\\\?\\"):
        absolute = str(destination)
        filesystem_destination = Path("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                                      else "\\\\?\\" + absolute)
    filesystem_destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        seen = set()
        for member in zipped.infolist():
            name = member.filename.replace("\\", "/")
            target = (destination/name).resolve()
            if (not target.is_relative_to(destination) or target in seen or ":" in name or
                    (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe or duplicate evidence archive member")
            seen.add(target)
        zipped.extractall(filesystem_destination)


def collect(plan, root, incoming, stage):
    for cell in plan["stages"][stage]:
        if cell["owner"] != "github":
            continue
        candidates = list(incoming.rglob(cell["id"]+"/manifest.json"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one cloud output for " + cell["id"])
        source = candidates[0].parent
        destination = root/"cells"/stage/cell["id"]
        if destination.exists():
            left = {p.relative_to(source).as_posix(): file_hash(p) for p in source.rglob("*") if p.is_file()}
            right = {p.relative_to(destination).as_posix(): file_hash(p) for p in destination.rglob("*") if p.is_file()}
            if left != right:
                raise ValueError("Conflicting duplicate cloud output")
        else:
            shutil.copytree(source, destination)


def publish_cloud_output(plan, root, repo, stage, cell, run_id, attempt):
    if not re.fullmatch(r"\d+", str(run_id)) or not re.fullmatch(r"\d+", str(attempt)):
        raise ValueError("Expected numeric GitHub run identity")
    registered = {c["id"] for c in plan["stages"][stage] if c["owner"] == "github"}
    if cell != "preparation" and cell not in registered:
        raise ValueError("Cannot publish unregistered cloud cell")
    paths = []
    if cell == "preparation":
        checks = root.parent/"cloud_checks"
        paths += [(p, "checks/"+p.name) for p in checks.glob("*") if p.is_file()]
    else:
        directory = root/"cells"/stage/cell
        paths += [(p, p.relative_to(root).as_posix()) for p in directory.rglob("*") if p.is_file()]
        for suffix in (".log", ".err.log"):
            log = root/"logs"/stage/(cell+suffix)
            if log.exists():
                paths.append((log, log.relative_to(root).as_posix()))
    release_log = root/"resource_release.jsonl"
    if release_log.exists():
        paths.append((release_log, release_log.name))
    out = root/"uploads"
    out.mkdir(exist_ok=True)
    archive = out/("%s_%s_run%s_attempt%s.zip" % (stage, cell, run_id, attempt))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for source, relative in sorted(paths):
            zipped.write(source, relative)
        zipped.writestr("execution_origin.json", json.dumps(dict(
            plan_hash=plan["plan_hash"], stage=stage, cell=cell, run_id=run_id, attempt=attempt)))
    if archive.stat().st_size >= 2*1024**3:
        raise ValueError("Release asset exceeds 2GiB; preserve local output for split transfer")
    command(["gh", "release", "upload", plan["campaign"], archive, "--repo", repo])


def gh_json(args, repo):
    return json.loads(command(["gh"] + args + (["--repo", repo] if repo else [])))


def stage_tag(plan, stage):
    return plan["campaign"].replace("paper-campaign-", "paper-" + stage + "-")


def ensure_release(plan, root, repo):
    info = gh_json(["repo", "view", repo, "--json", "isPrivate,nameWithOwner"], None)
    if info["isPrivate"]:
        raise ValueError("Free-only cloud mode requires a reviewed public code repository; no visibility change is automatic")
    remote = command(["git", "remote", "get-url", "origin"])
    if not (remote.endswith(repo+".git") or remote.endswith(repo)):
        raise ValueError("origin must be the explicitly selected experiment repository")
    remote_commit = gh_json(["api", "repos/"+repo+"/commits/"+plan["commit"]], repo=None)
    if remote_commit["sha"] != plan["commit"]:
        raise ValueError("Push the reviewed development commit before cloud preparation")
    existing = subprocess.run(["gh", "release", "view", plan["campaign"], "--repo", repo],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if existing.returncode == 0:
        downloaded = root/"release_identity"
        downloaded.mkdir(exist_ok=True)
        command(["gh", "release", "download", plan["campaign"], "--repo", repo, "--pattern",
                 "campaign.json", "--dir", downloaded, "--clobber"])
        if read_json(downloaded/"campaign.json") != plan:
            raise ValueError("Existing release belongs to a different plan")
    else:
        command(["gh", "release", "create", plan["campaign"], root/"campaign.json", "--repo", repo,
                 "--target", plan["commit"], "--title", plan["campaign"],
                 "--notes", "Frozen simulation campaign. Validation screen precedes formal training. No manuscript or personal data."])


def launch_cloud(plan, repo, stage):
    tag = stage_tag(plan, stage)
    refs = command(["git", "ls-remote", "origin", "refs/tags/"+tag])
    if refs:
        if refs.split()[0] != plan["commit"]:
            raise ValueError("Existing stage tag points to another commit")
    else:
        # An explicit tag push works without modifying the default branch.
        command(["git", "push", "origin", plan["commit"]+":refs/tags/"+tag])


def wait_cloud(plan, root, repo, stage):
    tag = stage_tag(plan, stage)
    deadline = time.monotonic() + 24*3600
    while time.monotonic() < deadline:
        runs = gh_json(["run", "list", "--limit", "100", "--json",
                        "databaseId,displayTitle,status,conclusion,headSha"], repo)
        found = [r for r in runs if r["displayTitle"] == tag and r["headSha"] == plan["commit"]]
        if len(found) > 1:
            raise ValueError("Multiple cloud attempts found; inspect before collecting")
        if found and found[0]["status"] == "completed":
            run = found[0]
            write_json(root/(stage+"_cloud_run.json"), run)
            details = gh_json(["api", "repos/"+repo+"/actions/runs/"+str(run["databaseId"])], None)
            attempt = details["run_attempt"]
            incoming = root/"incoming"/("%s%s_a%s" % (stage[0], run["databaseId"], attempt))
            if (incoming/"download.json").exists():
                data = incoming/read_json(incoming/"download.json")["data_directory"]
                if not data.resolve().is_relative_to(incoming.resolve()):
                    raise ValueError("Invalid saved transfer directory")
            else:
                data = incoming/("x"+str(time.time_ns()))
                archives = data/"archives"
                archives.mkdir(parents=True)
                command(["gh", "release", "download", plan["campaign"], "--repo", repo,
                         "--pattern", "%s_*_run%s_attempt%s.zip" % (stage, run["databaseId"], attempt),
                         "--dir", archives])
                for index, archive in enumerate(sorted(archives.glob("*.zip"))):
                    extract_evidence(archive, data/("d%d" % index))
                write_json(incoming/"download.json", dict(data_directory=data.name))
            if run["conclusion"] != "success":
                raise RuntimeError("Cloud stage failed/paused; downloaded all available artifacts for inspection")
            collect(plan, root, data, stage)
            return
        time.sleep(30)
    raise TimeoutError("No completed cloud stage within 24h; do not duplicate-dispatch")


def coordinate(plan, root, repo, auto_formal=False):
    if auto_formal:
        raise ValueError("credit_v3 is a bounded repair pilot; formal dispatch requires a new reviewed authorization")
    with awake_lock(root):
        ensure_release(plan, root, repo)
        for stage in ("screen", "formal"):
            if stage == "formal":
                if not auto_formal:
                    return
                evidence = root/"screen_evidence.zip"
                if not evidence.exists():
                    evidence = bundle_screen(plan, root)
                if not gate(plan, root)["passed"]:
                    raise ValueError("Local screen evidence no longer passes")
                # Never --clobber a frozen scientific evidence asset.
                assets = gh_json(["release", "view", plan["campaign"], "--json", "assets"], repo)["assets"]
                if not any(a["name"] == evidence.name for a in assets):
                    command(["gh", "release", "upload", plan["campaign"], evidence, "--repo", repo])
            launch_cloud(plan, repo, stage)
            local_ok = run_owned(plan, root, stage, "local")
            wait_cloud(plan, root, repo, stage)
            verdict = gate(plan, root, stage)
            write_json(root/(stage+"_gate.json"), verdict)
            if not local_ok or not verdict["passed"]:
                raise ValueError("Stage did not pass; no next-stage dispatch")
            command([sys.executable, "experiments/paper_convergence.py", "report", "--out",
                     root/"cells"/stage])
        write_json(root/"training_complete.json", dict(plan_hash=plan["plan_hash"],
                   formal_cells=18, full_tidal_test_done=False, publication_grade=False))


def cloud_prepare(root, repo, tag):
    match = STAGE_TAG.fullmatch(tag or "")
    if not match or not repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Invalid explicit stage tag/repository")
    stage, identity = match.groups()
    campaign = "paper-campaign-" + identity
    root.mkdir(parents=True, exist_ok=True)
    command(["gh", "release", "download", campaign, "--repo", repo,
             "--pattern", "campaign.json", "--dir", root])
    plan = load_plan(root/"campaign.json")
    if plan["campaign"] != campaign:
        raise ValueError("Release campaign does not match stage tag")
    if stage == "formal":
        command(["gh", "release", "download", campaign, "--repo", repo,
                 "--pattern", "screen_evidence.zip", "--dir", root])
        extracted = root/"evidence"
        extract_evidence(root/"screen_evidence.zip", extracted)
        if read_json(extracted/"campaign.json") != plan:
            raise ValueError("Evidence bundle belongs to a different campaign")
        recomputed = gate(plan, extracted)
        if not recomputed["passed"] or recomputed != read_json(extracted/"screen_gate.json"):
            raise ValueError("Uploaded screen evidence does not pass recomputation")
        write_json(root/"verified_gate.json", recomputed)
    return plan, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "local", "worker", "matrix", "gate", "collect",
                                            "bundle", "coordinate", "verify-bundle", "verify-code", "cloud-prepare", "publish-cloud"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), default="screen")
    parser.add_argument("--cell")
    parser.add_argument("--incoming", type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--tag")
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--auto-formal", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.command == "init":
        if args.verification is None:
            parser.error("--verification path/to/verification.json required")
        print(json.dumps(initialize(root, args.verification), indent=2))
        return
    if args.command == "verify-code":
        if not args.execute:
            parser.error("Verification is opt-in; supply --execute when ready to run tests")
        verify_code(root)
        return
    if args.command == "cloud-prepare":
        plan, stage = cloud_prepare(root, args.repo, args.tag)
        matrix = json.dumps(dict(include=[dict(cell=c["id"]) for c in plan["stages"][stage]
                                          if c["owner"] == "github"]))
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
                output.write("stage="+stage+"\n")
                output.write("matrix="+matrix+"\n")
                output.write("max_parallel="+str(plan["runtime"]["github_workers"])+"\n")
        print(matrix)
        return
    plan = load_plan(root/"campaign.json")
    if args.command == "publish-cloud":
        if not args.execute or not args.repo or not args.cell:
            parser.error("Publishing requires --execute, --repo and --cell")
        publish_cloud_output(plan, root, args.repo, args.stage, args.cell, args.run_id, args.attempt)
    elif args.command == "matrix":
        print(json.dumps(dict(include=[dict(cell=c["id"]) for c in plan["stages"][args.stage]
                                       if c["owner"] == "github"])))
    elif args.command == "gate":
        verdict = gate(plan, root, args.stage)
        write_json(root/(args.stage+"_gate.json"), verdict)
        raise SystemExit(0 if verdict["passed"] else 1)
    elif args.command == "collect":
        if args.incoming is None:
            parser.error("--incoming required")
        collect(plan, root, args.incoming, args.stage)
    elif args.command == "bundle":
        print(bundle_screen(plan, root))
    elif args.command == "verify-bundle":
        extracted = root/"evidence"
        extract_evidence(root/"screen_evidence.zip", extracted)
        if read_json(extracted/"campaign.json") != plan:
            raise ValueError("Evidence belongs to another campaign")
        recomputed = gate(plan, extracted)
        if not recomputed["passed"] or recomputed != read_json(extracted/"screen_gate.json"):
            raise ValueError("Screen evidence failed independent recomputation")
        write_json(root/"verified_gate.json", recomputed)
    else:
        if not args.execute:
            parser.error("Training/upload/tag pushes require --execute; preparation alone runs nothing")
        if args.command == "coordinate":
            if not args.repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
                parser.error("Supply an explicit --repo owner/name")
            coordinate(plan, root, args.repo, args.auto_formal)
        else:
            if args.stage == "formal":
                evidence_root = root/"evidence" if (root/"verified_gate.json").exists() else root
                if not gate(plan, evidence_root)["passed"]:
                    raise ValueError("All six screen cells must pass before formal training")
            owner = "github" if args.command == "worker" else "local"
            with awake_lock(root):
                ok = run_owned(plan, root, args.stage, owner, args.cell)
            raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

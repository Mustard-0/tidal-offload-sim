"""Orchestration-only recovery; execute the unchanged, separately checked out model."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile

MODEL_COMMIT = "4a17b77d98970fe090531420479020c607f4dfb9"
PLAN_HASH = "5a0555696eb44bcd2ee9c1a1ed3348acc1bc5b0de4782c687e9fe9b4fff901e8"
POLICY_NAME = "recovery_policy_20261019.json"
TAG = re.compile(r"recovery2500-wave-(\d{8}T\d{6}Z)-(\d{4})\Z")
RUNTIME_ROOT = Path(__file__).resolve().parents[1]
h = c = co = None
POLICY = ROOT = None


def bind(source):
    global h, c, co
    source = Path(source).resolve()
    os.chdir(source)
    sys.path.insert(0, str(source / "experiments"))
    import hybrid_campaign
    import paper2500_campaign
    import paper2500_coordinator
    h, c, co = hybrid_campaign, paper2500_campaign, paper2500_coordinator


def execution_plan(plan, policy):
    unsigned = {k: v for k, v in policy.items() if k != "policy_hash"}
    if policy.get("policy_hash") != h.digest(unsigned):
        raise ValueError("Recovery policy changed")
    if plan["commit"] != MODEL_COMMIT or plan["plan_hash"] != PLAN_HASH:
        raise ValueError("Wrong scientific plan")
    if (policy["original_plan_hash"] != PLAN_HASH or
            policy["maximum_total_segments_per_task"] != 96 or
            policy["prior_segments"] != dict(zip(
                [t["id"] for t in c.convergence_tasks()], [48, 1, 1, 1])) or
            policy["free_runner"] != "ubuntu-22.04" or
            policy["maximum_wall_days"] != 7 or policy["maximum_training_cells"] != 24 or
            policy["maximum_primary_training_interactions"] != 360000000):
        raise ValueError("Unapproved execution/budget change")
    value = copy.deepcopy(plan)
    value["runtime"].update(maximum_segments_per_task=96, local_workers=0)
    return value


def validate_wave(plan, policy, wave):
    p = execution_plan(plan, policy)
    c.validate_request(p, wave)
    if (wave["recovery_policy_hash"] != policy["policy_hash"] or
            wave["runtime_commit"] != policy["runtime_commit"] or not wave["cloud"]):
        raise ValueError("Wrong recovery wave")
    for e in wave["entries"]:
        previous = e["previous_segment"]
        if e["segment"] != previous + 1 or previous < policy["prior_segments"].get(e["task"]["id"], 0):
            raise ValueError("Historical recovery counter reset or skipped")
        if e["parent"] is None:
            if e["segment"] != 1:
                raise ValueError("Cannot reset a resumed task")
        elif e["segment"] <= policy["prior_segments"].get(e["task"]["id"], 0):
            raise ValueError("Recovery counters must include historical segments")
    if len({t["id"] for t in wave["stage_tasks"]}) != len(wave["stage_tasks"]):
        raise ValueError("Duplicate stage task")
    for task in wave["stage_tasks"]:
        c.validate_request(p, dict(plan_hash=PLAN_HASH, entries=[dict(task=task, segment=1)]))
    allowed = {t["id"]: t for t in wave["stage_tasks"]}
    if any(allowed.get(e["task"]["id"]) != e["task"] for e in wave["entries"]):
        raise ValueError("Task not in stage registry")
    return p


def publish_file(release, path):
    metadata = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{release}")
    found = [a for a in metadata["assets"] if a["name"] == path.name]
    digest = "sha256:" + h.file_hash(path)
    if not found:
        c.gh("release", "upload", release, path, "--repo", c.REPO)
        metadata = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{release}")
        found = [a for a in metadata["assets"] if a["name"] == path.name]
    if len(found) != 1 or found[0]["digest"] != digest or found[0]["size"] != path.stat().st_size:
        raise ValueError("Immutable asset conflicts")
    return dict(release=release, name=path.name, digest=digest, size=path.stat().st_size)


def make_request(p, entries, number, cloud):
    if not cloud:
        raise ValueError("This authorization migrates execution to free cloud only")
    state = h.read_json(ROOT / "launcher.json")
    entries = [dict(e, previous_segment=e["segment"] - 1) for e in entries]
    return dict(plan_hash=PLAN_HASH, release=f"paper2500-{p['stamp']}-w{number:04d}",
                filename=f"wave{number:04d}.json", entries=entries, cloud=True, number=number,
                recovery_policy_hash=POLICY["policy_hash"], runtime_commit=POLICY["runtime_commit"],
                stage=state["stage"], stage_tasks=state["tasks"],
                completed={k: v["parent"] for k, v in state["task_status"].items()
                           if v["status"] == "complete"})


def publish_request(root, p, req):
    validate_wave(c.load(root), POLICY, req)
    c.public_only()
    c.immutable(root / "requests" / req["filename"], req)
    releases = c.gh_json("api", f"repos/{c.REPO}/releases?per_page=100")
    if not any(r["tag_name"] == req["release"] for r in releases):
        c.gh("release", "create", req["release"], "--repo", c.REPO,
             "--target", POLICY["runtime_commit"], "--title", "Frozen-model bounded recovery wave",
             "--notes", "Free standard CPU only. Historical counters and all raw prefixes retained.")
    upload = root / "wave_upload" / f"w{req['number']:04d}"
    upload.mkdir(parents=True, exist_ok=True)
    wavefile = upload / "wave.json"
    c.immutable(wavefile, req)
    publish_file(req["release"], wavefile)
    tag = f"recovery2500-wave-{p['stamp']}-{req['number']:04d}"
    def git(*args):
        return subprocess.check_output(["git", "-C", str(RUNTIME_ROOT), *args],
                                       text=True, encoding="utf-8").strip()
    prior = git("tag", "--list", tag)
    if prior:
        if git("rev-parse", tag) != POLICY["runtime_commit"]:
            raise ValueError("Recovery tag changed")
    else:
        git("tag", tag, POLICY["runtime_commit"])
    git("-c", "credential.helper=!gh auth git-credential", "push",
        "https://github.com/" + c.REPO + ".git", "refs/tags/" + tag)
    return tag


def prepare(root, tag):
    match = TAG.fullmatch(tag)
    if not match:
        raise ValueError("Unregistered recovery tag")
    campaign = "paper2500-campaign-" + match[1]
    release = f"paper2500-{match[1]}-w{match[2]}"
    root.mkdir(parents=True, exist_ok=False)
    for pattern in ("plan.json", POLICY_NAME):
        c.gh("release", "download", campaign, "--repo", c.REPO, "--pattern", pattern, "--dir", root)
    reqdir = root / "requests"
    reqdir.mkdir()
    c.gh("release", "download", release, "--repo", c.REPO, "--pattern", "wave.json", "--dir", reqdir)
    policy = h.read_json(root / POLICY_NAME)
    request = h.read_json(reqdir / "wave.json")
    p = c.load(root)
    validate_wave(p, policy, request)
    if (request["release"] != release or os.environ.get("GITHUB_SHA") != policy["runtime_commit"] or
            os.environ.get("GITHUB_RUN_ATTEMPT") != "1"):
        raise ValueError("Workflow provenance differs")
    c.public_only()
    return p, policy, request


def verify_runtime(p):
    import torch
    import numpy
    import scipy
    import matplotlib
    actual = dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
                  scipy=scipy.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
                  torch_threads=torch.get_num_threads(), device="cpu")
    for k in ("python", "torch", "numpy", "scipy", "matplotlib", "torch_threads"):
        if actual[k] != p["runtime"][k]:
            raise ValueError("Pinned execution dependencies differ: " + k)
    return actual


def check_parent(root, plan, task, parent):
    path = c.restore(root, plan, task, parent)
    if path.exists() and task["kind"] == "train":
        c.inspect_training(path, task)
    return path


def cell(root, p, policy, entry):
    task = entry["task"]
    path = root / "cells" / task["id"]
    try:
        if time.time() > policy["original_started_epoch"] + 7 * 86400:
            raise ValueError("Original seven-day campaign deadline elapsed")
        verify_runtime(p)
        check_parent(root, p, task, entry["parent"])
        return c.run_cell(root, p, task, entry["parent"])
    except Exception as error:
        path.mkdir(parents=True, exist_ok=True)
        h.write_json(path / "failure.json", dict(type=type(error).__name__, error=str(error)))
        raise


def worker(root, p, policy, req, task_id):
    c.public_only()
    entry = next(e for e in req["entries"] if e["task"]["id"] == task_id)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    base = f"{task_id}.segment{entry['segment']:03d}"
    args = [sys.executable, "-u", str(Path(__file__).resolve()), "cell", "--source", str(c.ROOT),
            "--out", str(root), "--task", task_id, "--execute"]
    with (logs / (base + ".stdout.log")).open("x", encoding="utf-8") as log, (
            logs / (base + ".stderr.log")).open("x", encoding="utf-8") as err:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err,
                                text=True, encoding="utf-8", bufsize=1, cwd=c.ROOT)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        proc.stdout.close()
        code = proc.wait()
    path = root / "cells" / task_id
    outcome = "complete" if code == 0 and (path / "complete.json").exists() else (
        "paused" if code == 2 and (path / "paused.json").exists() else "failed")
    receipt = dict(task_id=task_id, outcome=outcome, segment=entry["segment"],
                   policy_hash=policy["policy_hash"], runtime_commit=policy["runtime_commit"],
                   model_commit=MODEL_COMMIT, run_id=os.environ["GITHUB_RUN_ID"],
                   attempt=os.environ["GITHUB_RUN_ATTEMPT"], runtime=verify_runtime(p),
                   parent=entry["parent"], cell_exit_code=code, child_pid=proc.pid,
                   child_exited=True, available_ram_gib=h.free_ram_gib())
    c.immutable(logs / (base + ".resource_release.json"), receipt)
    if path.exists():
        destination = root / (base + ".zip")
        c.bundle(root, p, entry["task"], destination, entry["parent"], outcome)
        with zipfile.ZipFile(destination, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("recovery_execution.json", json.dumps(receipt, indent=2))
        archive_ref = publish_file(req["release"], destination)
        summary = root / (base + ".outcome.json")
        c.immutable(summary, dict(receipt, archive=archive_ref))
        publish_file(req["release"], summary)
    if outcome == "failed":
        raise RuntimeError("Unknown/code/numerical failure: evidence retained, no automatic rerun")


def verify_cloud_origin(origin, execution, task, parent, segment, run_id, policy):
    if (origin["plan_hash"] != PLAN_HASH or origin["commit"] != MODEL_COMMIT or
            origin["task_id"] != task["id"] or origin["parent"] != parent or
            origin["run_id"] != str(run_id) or origin["attempt"] != "1" or
            execution["policy_hash"] != policy["policy_hash"] or
            execution["runtime_commit"] != policy["runtime_commit"] or
            execution["segment"] != segment or execution["task_id"] != task["id"] or
            execution["run_id"] != str(run_id) or execution["attempt"] != "1" or
            execution["parent"] != parent or execution["outcome"] != origin["outcome"]):
        raise ValueError("Cloud provenance/lineage differs")


def finalize(root, p, policy, req):
    """Gate on a roomy cloud runner, never on the low-memory desktop."""
    refs = dict(req["completed"])
    metadata = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{req['release']}")
    assets = {a["name"]: a for a in metadata["assets"]}
    for e in req["entries"]:
        name = f"{e['task']['id']}.segment{e['segment']:03d}.zip"
        if name not in assets:
            return
        ref = dict(release=req["release"], name=name, digest=assets[name]["digest"])
        data, origin, _ = c.receive_asset(root, ref)
        execution = h.read_json(data / "recovery_execution.json")
        verify_cloud_origin(origin, execution, e["task"], e["parent"], e["segment"],
                            os.environ["GITHUB_RUN_ID"], policy)
        if origin["outcome"] != "complete":
            return
        refs[e["task"]["id"]] = ref
    if set(refs) != {t["id"] for t in req["stage_tasks"]}:
        return
    for task in req["stage_tasks"]:
        data, origin, _ = c.receive_asset(root, refs[task["id"]])
        if origin["task_id"] != task["id"] or origin["outcome"] != "complete" or origin["commit"] != MODEL_COMMIT:
            raise ValueError("Final stage checkpoint changed")
        destination = root / "cells" / task["id"]
        destination.parent.mkdir(exist_ok=True)
        shutil.copytree(data / "cell", destination, copy_function=os.link)
    if req["stage_tasks"][0]["kind"] == "train":
        c.choose_and_gate(root, req["stage_tasks"], p, req["stage"])
    c.immutable(root / "registries" / (req["stage"] + ".json"),
                dict(plan_hash=PLAN_HASH, tasks=req["stage_tasks"]))
    from paper2500_evaluation import report
    report(root, p)
    name = "gate_" + req["stage"] + ".zip"
    destination = root / name
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("gate_origin.json", json.dumps(dict(
            plan_hash=PLAN_HASH, policy_hash=policy["policy_hash"], runtime_commit=policy["runtime_commit"],
            model_commit=MODEL_COMMIT, stage=req["stage"], tasks=req["stage_tasks"], parents=refs,
            run_id=os.environ["GITHUB_RUN_ID"], attempt=os.environ["GITHUB_RUN_ATTEMPT"])))
        gate = root / (req["stage"] + "_gate.json")
        if gate.exists():
            archive.write(gate, gate.name)
        for report in (root / "report").glob("*"):
            if report.is_file():
                archive.write(report, "report/" + report.name)
    publish_file(req["release"], destination)


def ready_to_finalize(root, policy, req):
    assets = {a["name"]: a for a in c.gh_json(
        "api", f"repos/{c.REPO}/releases/tags/{req['release']}")["assets"]}
    complete = set(req["completed"])
    folder = root / "summaries"
    folder.mkdir(exist_ok=True)
    for e in req["entries"]:
        name = f"{e['task']['id']}.segment{e['segment']:03d}.outcome.json"
        if name not in assets:
            return False
        c.gh("release", "download", req["release"], "--repo", c.REPO, "--pattern", name, "--dir", folder)
        path = folder / name
        if "sha256:" + h.file_hash(path) != assets[name]["digest"]:
            raise ValueError("Outcome summary digest mismatch")
        value = h.read_json(path)
        if (value["policy_hash"] != policy["policy_hash"] or
                value["runtime_commit"] != policy["runtime_commit"] or
                value["run_id"] != os.environ["GITHUB_RUN_ID"] or value["attempt"] != "1" or
                value["segment"] != e["segment"] or value["task_id"] != e["task"]["id"]):
            raise ValueError("Wrong outcome summary")
        if value["outcome"] != "complete":
            return False
        complete.add(e["task"]["id"])
    return complete == {t["id"] for t in req["stage_tasks"]}


def receive_wave(root, p, wave, state):
    run = c.gh_json("run", "view", wave["run_id"], "--repo", c.REPO,
                    "--json", "status,conclusion,headSha,attempt,jobs")
    if run["headSha"] != POLICY["runtime_commit"] or run["attempt"] != 1:
        raise ValueError("Wrong recovery run/attempt")
    wave["last_state"] = run
    if run["status"] != "completed":
        return False
    req = wave["request"]
    assets = {a["name"]: a for a in c.gh_json(
        "api", f"repos/{c.REPO}/releases/tags/{req['release']}")["assets"]}
    for e in req["entries"]:
        task, segment = e["task"], e["segment"]
        if task["id"] in wave.get("received_tasks", []):
            continue
        name = f"{task['id']}.segment{segment:03d}.zip"
        if name not in assets:
            state.update(status="needs_review", reason="Unpublished cloud outcome; never rerun or assume quota")
            continue
        ref = dict(release=req["release"], name=name, digest=assets[name]["digest"])
        data, origin, parent = c.receive_asset(root, ref)
        execution = h.read_json(data / "recovery_execution.json")
        verify_cloud_origin(origin, execution, task, e["parent"], segment, wave["run_id"], POLICY)
        destination = root / "cells" / task["id"]
        if destination.exists():
            raise ValueError("Local/cloud ownership collision")
        destination.parent.mkdir(exist_ok=True)
        shutil.copytree(data / "cell", destination)
        outcome = origin["outcome"]
        state["task_status"][task["id"]].update(status=outcome, segments=segment, owner="github", parent=parent)
        if outcome == "paused":
            destination.rename(destination.with_name(task["id"] + f".cloud_segment{segment:03d}"))
        elif outcome != "complete":
            state.update(status="needs_review", reason="Cloud code/numerical failure; no expansion")
        wave.setdefault("received_tasks", []).append(task["id"])
    if state["status"] != "needs_review" and all(v["status"] == "complete" for v in state["task_status"].values()):
        name = "gate_" + state["stage"] + ".zip"
        if name not in assets:
            state.update(status="needs_review", reason="Missing complete-stage cloud integrity gate")
        else:
            data, _, _ = receive_gate(root, req["release"], assets[name], state, wave["run_id"])
            gate = data / (state["stage"] + "_gate.json")
            if gate.exists():
                c.immutable(root / gate.name, h.read_json(gate))
            state["last_verified_gate"] = dict(stage=state["stage"], run_id=wave["run_id"],
                                                digest=assets[name]["digest"])
    if run["conclusion"] != "success" and state["status"] != "needs_review":
        state.update(status="needs_review", reason="Cloud workflow failed; retain published results without rerun")
    h.write_json(root / "launcher.json", state)
    return True


def receive_gate(root, release, asset, state, run_id):
    folder = root / "gate_assets" / str(asset["id"])
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / asset["name"]
    if not archive.exists():
        c.gh("release", "download", release, "--repo", c.REPO, "--pattern", asset["name"], "--dir", folder)
    if archive.stat().st_size != asset["size"] or "sha256:" + h.file_hash(archive) != asset["digest"]:
        raise ValueError("Gate digest/size mismatch")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Gate ZIP CRC")
        origin = json.loads(zipped.read("gate_origin.json"))
    parents = {k: v["parent"] for k, v in state["task_status"].items()}
    # Size is downloaded metadata, not part of the original request's parent identity.
    clean = lambda refs: {k: {x: v[x] for x in ("release", "name", "digest")} for k, v in refs.items()}
    if (origin["plan_hash"] != PLAN_HASH or origin["policy_hash"] != POLICY["policy_hash"] or
            origin["runtime_commit"] != POLICY["runtime_commit"] or origin["model_commit"] != MODEL_COMMIT or
            origin["run_id"] != str(run_id) or origin["attempt"] != "1" or
            origin["stage"] != state["stage"] or origin["tasks"] != state["tasks"] or
            clean(origin["parents"]) != clean(parents)):
        raise ValueError("Cloud stage gate provenance mismatch")
    data = folder / "data"
    if not data.exists():
        h.extract_evidence(archive, data)
    return data, origin, asset


def next_stage(root, p, state):
    stage, tasks = state["stage"], state["tasks"]
    if state.get("last_verified_gate", {}).get("stage") != stage:
        raise ValueError("Require a cloud raw-evidence audit before advancing")
    if stage in ("convergence", "auxiliary"):
        gate = h.read_json(root / (stage + "_gate.json"))
        if not gate["passed"]:
            state.update(status="needs_review", reason="Predeclared learning gate failed; no retuning")
            return
        if stage == "convergence":
            state["choices"] = gate["selected_lr"]
            state["convergence_tasks"] = tasks
            new, stage = c.auxiliary_tasks(state["choices"]), "auxiliary"
        else:
            state["choices"].update(gate["selected_lr"])
            state["auxiliary_tasks"] = tasks
            models = []
            for task in state["convergence_tasks"] + tasks:
                s = task["settings"]
                if s["lr"] == state["choices"][s["method"]]:
                    models.append(dict(method=s["method"], seed=s["seed"], task=task,
                                       asset=co.attach_model(root, p, task)))
            state["models"] = models
            from paper2500_evaluation import evaluation_tasks
            new, stage = evaluation_tasks("scaling", models), "scaling"
    elif stage == "scaling":
        if not (root / "privacy_C1/complete.json").exists():
            raise ValueError("Retain/reuse original C1, never silently regenerate it")
        new, stage = c.sensitivity_tasks(state["choices"]["TMPPO"]), "sensitivity"
    elif stage == "sensitivity":
        gate = h.read_json(root / "sensitivity_gate.json")
        if not gate["integrity_passed"]:
            raise ValueError("Sensitivity raw integrity failed")
        models = [dict(method="TMPPO", seed=t["settings"]["seed"], task=t,
                       asset=co.attach_model(root, p, t)) for t in tasks]
        from paper2500_evaluation import evaluation_tasks
        new, stage = evaluation_tasks("sensitivity_evaluation", models), "sensitivity_evaluation"
    else:
        state.update(status="complete_with_disclosed_limits", reason="Finite queue finished, no automatic extra trials")
        return
    c.immutable(root / "registries" / (stage + ".json"), dict(plan_hash=PLAN_HASH, tasks=new))
    registered = [t for f in (root / "registries").glob("*.json")
                  for t in h.read_json(f)["tasks"] if t["kind"] == "train"]
    if len(registered) > 24 or sum(t["settings"]["total_env_steps"] for t in registered) > 360000000:
        raise ValueError("Total scientific budget exceeded")
    state.update(stage=stage, tasks=new, task_status={t["id"]: dict(status="pending", segments=0, owner="github") for t in new})


def migrate(oldroot, root, policy):
    p = c.load(oldroot)
    execution_plan(p, policy)
    old = h.read_json(oldroot / "launcher.json")
    if old["status"] != "needs_review" or old["worker_pids"] or old["waves"]:
        raise ValueError("Require fully exited, audited original queue")
    if policy["original_started_epoch"] != old["started_epoch"]:
        raise ValueError("The original campaign deadline cannot be reset")
    if root.exists():
        raise FileExistsError("Migration already initialized; never reset recovery counters")
    for task in old["tasks"]:
        if c.inspect_training(oldroot / "cells" / task["id"], task) != "paused":
            raise ValueError("Only a verified paused prefix may migrate")
    root.mkdir()
    shutil.copy2(oldroot / "plan.json", root / "plan.json")
    c.immutable(root / POLICY_NAME, policy)
    c.immutable(root / "original_launcher.json", old)
    c.immutable(root / "migration_decision.json", dict(
        authorization="Human approved safe free-cloud migration and a one-time finite recovery allowance",
        original_plan_hash=PLAN_HASH, original_parent=old["pid"], original_run=str(oldroot),
        old_limit=48, new_total_limit=96, reset_training=False, reset_prior_segments=False,
        runtime_commit=policy["runtime_commit"], policy_hash=policy["policy_hash"]))
    shutil.copytree(oldroot / "registries", root / "registries")
    shutil.copytree(oldroot / "privacy_C1", root / "privacy_C1")
    c.public_only()
    publish_file(p["campaign"], root / POLICY_NAME)
    state = copy.deepcopy(old)
    state.update(status="running", pid=None, worker_pids=[], waves=[], parallel_limit=0,
                 recovery_policy_hash=policy["policy_hash"], runtime_commit=policy["runtime_commit"])
    state.pop("reason", None)
    for task in state["tasks"]:
        st = state["task_status"][task["id"]]
        segment = st["segments"]
        if segment != policy["prior_segments"][task["id"]]:
            raise ValueError("Historical counter differs")
        asset = oldroot / f"{task['id']}.segment{segment:03d}.zip"
        with zipfile.ZipFile(asset) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Local checkpoint ZIP CRC failed")
            origin = json.loads(zipped.read("execution_origin.json"))
            if (origin["plan_hash"] != PLAN_HASH or origin["commit"] != MODEL_COMMIT or
                    origin["task_id"] != task["id"] or origin["outcome"] != "paused" or
                    hashlib.sha256(zipped.read("cell/latest.pt")).hexdigest() !=
                    h.file_hash(oldroot / "cells" / task["id"] / "latest.pt")):
                raise ValueError("Migration bundle/prefix differs")
        ref = publish_file(p["campaign"], asset)
        st.update(status="paused", owner="github", parent=ref)
        st.pop("worker_pid", None)
    c.immutable(root / "launcher.json", state)
    print(json.dumps(dict(root=str(root), policy_hash=policy["policy_hash"], parent_segments=policy["prior_segments"])))


def main():
    global POLICY, ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "migrate", "coordinate", "worker", "cell", "finalize", "ready"))
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--task")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    bind(args.source)
    if args.command == "prepare":
        p, policy, req = prepare(root, args.tag)
        print(json.dumps(dict(task=[e["task"]["id"] for e in req["entries"]])))
        return
    if not args.execute:
        raise ValueError("Explicit execution authorization required")
    if args.command == "migrate":
        migrate(args.old_root.resolve(), root, h.read_json(args.policy))
        return
    plan = c.load(root)
    POLICY = h.read_json(root / POLICY_NAME)
    ROOT = root
    p = execution_plan(plan, POLICY)
    if args.command == "coordinate":
        if subprocess.check_output(["git", "-C", str(RUNTIME_ROOT), "rev-parse", "HEAD"], text=True).strip() != POLICY["runtime_commit"]:
            raise ValueError("Unfrozen recovery controller")
        co.request, co.publish_request = make_request, publish_request
        co.receive_wave, co.next_stage = receive_wave, next_stage
        co.coordinate(root, p)
        return
    req = h.read_json(root / "requests/wave.json")
    validate_wave(plan, POLICY, req)
    if args.command == "ready":
        print("true" if ready_to_finalize(root, POLICY, req) else "false")
    elif args.command == "worker":
        worker(root, p, POLICY, req, args.task)
    elif args.command == "finalize":
        finalize(root, p, POLICY, req)
    else:
        entry = next(e for e in req["entries"] if e["task"]["id"] == args.task)
        result = cell(root, p, POLICY, entry)
        raise SystemExit(0 if result == "complete" else 2)


if __name__ == "__main__":
    main()

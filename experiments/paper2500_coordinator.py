"""Standard-library queue, disjoint ownership and explicit cloud-limit fallback."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import hybrid_campaign as h
import paper2500_campaign as c


def request(p, entries, number, cloud):
    release = f"paper2500-{p['stamp']}-w{number:04d}"
    return dict(plan_hash=p["plan_hash"], release=release, filename=f"wave{number:04d}.json",
                entries=entries, cloud=cloud, number=number)


def entry(task, status):
    return dict(task=task, parent=status.get("parent"), segment=status.get("segments", 0)+1)


def publish_request(root, p, req):
    c.public_only()
    c.immutable(root/"requests"/req["filename"], req)
    releases = c.gh_json("api", f"repos/{c.REPO}/releases?per_page=100")
    existing = next((r for r in releases if r["tag_name"] == req["release"]), None)
    if existing is None:
        c.gh("release", "create", req["release"], "--repo", c.REPO, "--target", p["commit"],
             "--title", f"Bounded simulation wave{req['number']:04d}", "--notes",
             "Frozen simulation checkpoint lineage. Free standard CPU only; no artifacts/cache.")
    upload = root/"wave_upload"/f"w{req['number']:04d}"
    upload.mkdir(parents=True, exist_ok=True)
    (upload/"wave.json").write_bytes((root/"requests"/req["filename"]).read_bytes())
    metadata = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{req['release']}")
    asset = next((a for a in metadata["assets"] if a["name"] == "wave.json"), None)
    if asset is None:
        c.gh("release", "upload", req["release"], upload/"wave.json", "--repo", c.REPO)
    elif asset["digest"] != "sha256:"+h.file_hash(upload/"wave.json"):
        raise ValueError("Conflicting published request")
    if req["cloud"]:
        tag = f"paper2500-wave-{p['stamp']}-{req['number']:04d}"
        prior = h.command(["git", "tag", "--list", tag])
        if prior:
            if h.command(["git", "rev-parse", tag]) != p["commit"]:
                raise ValueError("Existing wave tag points elsewhere")
        else:
            h.command(["git", "tag", tag, p["commit"]])
        h.command(["git", "push", "origin", "refs/tags/"+tag])
        return tag


def attach_model(root, p, task):
    identifier = task["id"]
    path = root/"model_assets"/(identifier+".zip")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        cell = root/"cells"/identifier
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for name in ("manifest.json", "selection.json", "best_checkpoint.pt"):
                z.write(cell/name, name)
            z.writestr("model_origin.json", json.dumps(dict(plan_hash=p["plan_hash"],
                       task_id=identifier, commit=p["commit"], test_used=False)))
    release = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{p['campaign']}")
    matches = [a for a in release["assets"] if a["name"] == path.name]
    if not matches:
        c.gh("release", "upload", p["campaign"], path, "--repo", c.REPO)
        release = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{p['campaign']}")
        matches = [a for a in release["assets"] if a["name"] == path.name]
    if len(matches) != 1 or matches[0]["digest"] != "sha256:"+h.file_hash(path):
        raise ValueError("Selected model publication mismatch")
    return dict(release=p["campaign"], name=path.name, digest=matches[0]["digest"])


def next_stage(root, p, state):
    stage = state["stage"]
    tasks = state["tasks"]
    if stage == "convergence":
        h.command([sys.executable, "experiments/paper2500_campaign.py", "gate", "--out", str(root), "--task", stage])
        gate = h.read_json(root/"convergence_gate.json")
        if not gate["passed"]:
            state.update(status="needs_review", reason="Predeclared convergence quality gate failed; no extra tuning")
            return
        state["choices"] = gate["selected_lr"]
        state["convergence_tasks"] = tasks
        new = c.auxiliary_tasks(state["choices"])
        stage = "auxiliary"
    elif stage == "auxiliary":
        h.command([sys.executable, "experiments/paper2500_campaign.py", "gate", "--out", str(root), "--task", stage])
        state["choices"].update(h.read_json(root/"auxiliary_gate.json")["selected_lr"])
        state["auxiliary_tasks"] = tasks
        state["models"] = []
        for task in state["convergence_tasks"]+tasks:
            s = task["settings"]
            if s["lr"] == state["choices"][s["method"]]:
                state["models"].append(dict(method=s["method"], seed=s["seed"], task=task,
                                            asset=attach_model(root, p, task)))
        from paper2500_evaluation import evaluation_tasks
        new = evaluation_tasks("scaling", state["models"])
        stage = "scaling"
    elif stage == "scaling":
        h.command([sys.executable, "experiments/privacy_quality_simple.py", "--out", str(root/"privacy_C1")])
        new = c.sensitivity_tasks(state["choices"]["TMPPO"])
        stage = "sensitivity"
    elif stage == "sensitivity":
        models = [dict(method="TMPPO", seed=t["settings"]["seed"], task=t,
                       asset=attach_model(root, p, t)) for t in tasks]
        from paper2500_evaluation import evaluation_tasks
        new = evaluation_tasks("sensitivity_evaluation", models)
        stage = "sensitivity_evaluation"
    else:
        h.command([sys.executable, "experiments/paper2500_campaign.py", "report", "--out", str(root)])
        state.update(status="complete_with_disclosed_limits", reason="Finite queue finished; no automatic repeats")
        return
    c.immutable(root/"registries"/(stage+".json"), dict(plan_hash=p["plan_hash"], tasks=new))
    registered = [t for f in (root/"registries").glob("*.json") for t in h.read_json(f)["tasks"] if t["kind"] == "train"]
    if (len(registered) > p["maximum_training_cells"] or sum(t["settings"]["total_env_steps"] for t in registered)
            > p["maximum_primary_training_interactions"]):
        raise ValueError("Finite registered training allowance exceeded")
    state.update(stage=stage, tasks=new, task_status={t["id"]: dict(status="pending", segments=0) for t in new})


def receive_wave(root, p, wave, state):
    run = c.gh_json("run", "view", wave["run_id"], "--repo", c.REPO,
                    "--json", "status,conclusion,headSha,attempt,jobs")
    if run["headSha"] != p["commit"] or run["attempt"] != 1:
        raise ValueError("Wrong cloud run/attempt; no blind rerun")
    wave["last_state"] = run
    if run["status"] != "completed":
        return False
    release = c.gh_json("api", f"repos/{c.REPO}/releases/tags/{wave['request']['release']}")
    assets = {a["name"]: a for a in release["assets"]}
    missing = []
    for e in wave["request"]["entries"]:
        task, segment = e["task"], e["segment"]
        if task["id"] in wave.get("received_tasks", []):
            continue
        name = f"{task['id']}.segment{segment:03d}.zip"
        if name not in assets:
            missing.append(e)
            continue
        ref = dict(release=release["tag_name"], name=name, digest=assets[name]["digest"])
        data, origin, parent = c.receive_asset(root, ref)
        if (origin["plan_hash"] != p["plan_hash"] or origin["task_id"] != task["id"] or
                origin["commit"] != p["commit"] or origin["parent"] != e["parent"] or
                origin.get("run_id") != str(wave["run_id"]) or origin.get("attempt") != "1"):
            raise ValueError("Cloud result/checkpoint lineage mismatch")
        target = root/"cells"/task["id"]
        if target.exists():
            raise ValueError("Cloud/local ownership collision")
        import shutil
        # No merged or overwritten raw results: each cloud segment is retained separately.
        shutil.copytree(data/"cell", target)
        outcome = origin["outcome"]
        st = state["task_status"][task["id"]]
        st.update(status=outcome, segments=segment, parent=parent, owner="github")
        if outcome == "failed":
            state.update(status="needs_review", reason="Cloud numerical/code failure")
        elif outcome == "paused":
            # Retain the whole imported prefix under its immutable segment directory.
            target.rename(root/"cells"/(task["id"]+f".cloud_segment{segment:03d}"))
        wave.setdefault("received_tasks", []).append(task["id"])
        h.write_json(root/"launcher.json", state)
    if missing:
        annotations = []
        for job in run["jobs"]:
            details = c.gh_json("api", f"repos/{c.REPO}/actions/jobs/{job['databaseId']}")
            check_id = details["check_run_url"].rsplit("/", 1)[-1]
            annotations.extend(c.gh_json("api", f"repos/{c.REPO}/check-runs/{check_id}/annotations"))
        text = json.dumps(annotations).lower()
        explicit_quota = any(x in text for x in ("billing", "spending limit", "payment", "usage limit"))
        for e in missing:
            st = state["task_status"][e["task"]["id"]]
            job = next((j for j in run["jobs"] if e["task"]["id"] in j["name"]), None)
            ran_training = job and any(s["name"] == "Bounded segment" and s.get("startedAt") not in (None, "0001-01-01T00:00:00Z")
                                       for s in job.get("steps", []))
            if explicit_quota and not ran_training:
                st.update(status="paused" if st.get("parent") else "pending", owner="local")
                state["cloud_disabled_reason"] = dict(reason="Explicit GitHub account/billing limit; free-only local fallback",
                                                        annotations=annotations, run_id=wave["run_id"])
            else:
                state.update(status="needs_review", reason="Unpublished/unknown cloud outcome; do not duplicate unknown training")
    return True


def coordinate(root, p):
    c.publish_plan(root, p)
    state_path = root/"launcher.json"
    if state_path.exists():
        state = h.read_json(state_path)
        if state["status"] not in ("running", "waiting_resources"):
            raise ValueError("Do not resume failed/completed campaigns")
        for st in state["task_status"].values():
            if st["status"] == "running_local":
                raise ValueError("Inspect previous real child PIDs/paused evidence before resuming coordinator")
    else:
        state = dict(plan_hash=p["plan_hash"], status="running", stage="convergence", tasks=p["initial_tasks"],
            task_status={t["id"]: dict(status="pending", segments=0) for t in p["initial_tasks"]},
            waves=[], wave_counter=0, started_epoch=time.time(), pid=os.getpid(), worker_pids=[])
        c.immutable(root/"registries/convergence.json", dict(plan_hash=p["plan_hash"], tasks=state["tasks"]))
    local = {}
    cloud_poll = 0.
    with h.awake_lock(root):
      try:
        while True:
            load = c.load(root)
            if state["plan_hash"] != load["plan_hash"]:
                raise ValueError("Coordinator plan identity changed")
            now = time.time()
            if now-state["started_epoch"] > p["maximum_wall_days"]*86400:
                h.write_json(root/"cells/STOP_AFTER_UPDATE", dict(reason="Finite campaign wall limit"))
                state.update(status="needs_review", reason="Campaign wall limit reached; boundary pause requested")
            for identifier, item in list(local.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["stdout"].close()
                item["stderr"].close()
                del local[identifier]
                st = state["task_status"][identifier]
                outcome = root/"outcomes"/f"{identifier}.segment{st['segments']:03d}.json"
                if code or not outcome.exists():
                    state.update(status="needs_review", reason=f"Local worker failure: {identifier}")
                    st["status"] = "failed"
                    continue
                st["status"] = h.read_json(outcome)["outcome"]
                if st["status"] == "paused":
                    pause = h.read_json(root/"cells"/identifier/"paused.json")
                    if pause["reason"] == "available RAM below .5GiB":
                        state["parallel_limit"] = max(1, state.get("parallel_limit", p["runtime"]["local_workers"])-1)
                h.write_json(root/"last_resource_release.json", dict(task=identifier, pid=item["process"].pid,
                    exit_code=code, available_ram_gib=h.free_ram_gib(), observed_utc=datetime.now(timezone.utc).isoformat()))
                if st["status"] == "failed":
                    state.update(status="needs_review", reason=f"Failed local cell: {identifier}")
            if now-cloud_poll >= 120:
              try:
                for wave in state["waves"]:
                    if wave.get("finished"):
                        continue
                    if wave.get("dispatch_pending"):
                        wave["tag"] = publish_request(root, p, wave["request"])
                        wave["dispatch_pending"] = False
                    if not wave.get("run_id"):
                        runs = c.gh_json("run", "list", "--repo", c.REPO, "--branch", wave["tag"],
                                         "--json", "databaseId,headSha", "--limit", "5")
                        if len(runs) > 1:
                            raise ValueError("Duplicate cloud dispatch")
                        if runs:
                            wave["run_id"] = runs[0]["databaseId"]
                    if wave.get("run_id"):
                        wave["finished"] = receive_wave(root, p, wave, state)
                cloud_poll = now
                state.pop("network_error", None)
              except (subprocess.SubprocessError, OSError) as error:
                # A lost API response is not authorization to rerun training.
                state["network_error"] = dict(message=str(error)[-2000:], observed_epoch=now)
                cloud_poll = now
            terminal = state["status"] == "needs_review"
            active_cloud = any(not w.get("finished") for w in state["waves"])
            if terminal and not local and not active_cloud:
                h.write_json(state_path, state)
                return
            if not terminal and all(st["status"] == "complete" for st in state["task_status"].values()):
                next_stage(root, p, state)
                if state["status"].startswith("complete") or state["status"] == "needs_review":
                    h.write_json(state_path, state)
                    return
            if not terminal:
                ready = [t for t in state["tasks"] if state["task_status"][t["id"]]["status"] in ("pending", "paused")]
                launched_local = False
                for task in list(ready):
                    st = state["task_status"][task["id"]]
                    if st["segments"] >= p["runtime"]["maximum_segments_per_task"]:
                        state.update(status="needs_review", reason="Finite segment recovery allowance exhausted")
                        break
                    if len(local) >= state.get("parallel_limit", p["runtime"]["local_workers"]) or h.free_ram_gib() < 1.5:
                        break
                    if st.get("owner") == "github" and not state.get("cloud_disabled_reason"):
                        continue
                    state["wave_counter"] += 1
                    req = request(p, [entry(task, st)], state["wave_counter"], False)
                    c.immutable(root/"requests"/req["filename"], req)
                    st.update(status="running_local", owner="local", segments=st["segments"]+1)
                    logs = root/"coordinator_logs"
                    logs.mkdir(exist_ok=True)
                    stdout = (logs/(task["id"]+f".{st['segments']:03d}.out.log")).open("x", encoding="utf-8")
                    stderr = (logs/(task["id"]+f".{st['segments']:03d}.err.log")).open("x", encoding="utf-8")
                    args = [sys.executable, "-u", str(c.ROOT/"experiments/paper2500_campaign.py"), "worker",
                            "--out", str(root), "--request", str(root/"requests"/req["filename"]),
                            "--task", task["id"], "--local", "--execute"]
                    proc = subprocess.Popen(args, cwd=c.ROOT, stdout=stdout, stderr=stderr)
                    st["worker_pid"] = proc.pid
                    local[task["id"]] = dict(process=proc, stdout=stdout, stderr=stderr)
                    launched_local = True
                    # Observe RAM after imports before starting another local child.
                    break
                ready = [t for t in state["tasks"] if state["task_status"][t["id"]]["status"] in ("pending", "paused")
                         and state["task_status"][t["id"]].get("owner") != "local"]
                if ready and not active_cloud and not state.get("cloud_disabled_reason") and not launched_local:
                    state["wave_counter"] += 1
                    selected = ready[:p["runtime"]["github_workers"]]
                    req = request(p, [entry(t, state["task_status"][t["id"]]) for t in selected], state["wave_counter"], True)
                    state["waves"].append(dict(request=req, dispatched_epoch=now, dispatch_pending=True))
                    for task in selected:
                        state["task_status"][task["id"]].update(status="running_cloud", owner="github")
                    h.write_json(state_path, state)
                    cloud_poll = 0.
            state["pid"] = os.getpid()
            if not local and not active_cloud and h.free_ram_gib() < 1.5:
                state.setdefault("resource_wait_started", now)
                if now-state["resource_wait_started"] > p["resource_wait_hours"]*3600:
                    state.update(status="needs_review", reason="Finite12-hour resource wait elapsed")
            else:
                state.pop("resource_wait_started", None)
            state["worker_pids"] = [x["process"].pid for x in local.values()]
            state["observed_utc"] = datetime.now(timezone.utc).isoformat()
            h.write_json(state_path, state)
            time.sleep(15)
      except BaseException as error:
        state.update(status="needs_review", reason=f"Coordinator {type(error).__name__}: {error}")
        h.write_json(root/"cells/STOP_AFTER_UPDATE", dict(reason=state["reason"]))
        h.write_json(state_path, state)
        # Do not leave a live local child without its resource supervisor.
        for item in local.values():
            item["process"].wait()
            item["stdout"].close()
            item["stderr"].close()
        raise

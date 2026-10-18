"""Bounded diagnostic ablation; no cloud dispatch or implicit formal training."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import hybrid_campaign as h

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("control", "trace1", "noentropy", "trace1_noentropy")
STUDIES = {"lambda_entropy": VARIANTS,
           "shared_value": ("team_value_control", "team_value_shared"),
           "role_credit": ("role_joint", "role_route_only", "role_wlo_only")}
ALL_VARIANTS = VARIANTS + STUDIES["shared_value"] + STUDIES["role_credit"]


def registered_settings(variant, seed=87021):
    if variant not in ALL_VARIANTS or seed not in (87021, 87035):
        raise ValueError("Unregistered ablation or confirmation seed")
    return dict(method="TMPPO", lr=1, seed=seed, updates=128, rollout=128,
                episode_steps=6000, phase_count=5, validate_every=32,
                validation_steps=1000, warmup_steps=300,
                validation_seeds=[814001, 814019], validation_phases=[30., 130., 230.],
                training_profile=("contract_v4" if variant in VARIANTS else
                                  "shared_value_v5" if variant in STUDIES["shared_value"] else "role_credit_v6"),
                diagnostic_variant=variant,
                checkpoint_every=25, schedule_updates=2000, policy_diagnostics=True,
                diagnostics_endpoints_only=True, baseline_diagnostics=False,
                validation_objective="pooled_task_cost")


def initialize(root, verification, study="lambda_entropy"):
    proof = h.read_json(verification)
    if not proof.get("passed") or proof["source_hashes"] != h.code_hashes():
        raise ValueError("Verify this exact source first")
    if h.command(["git", "status", "--porcelain"]):
        raise ValueError("Commit the reviewed source first")
    if study not in STUDIES:
        raise ValueError("Unknown bounded diagnostic study")
    cases=[dict(variant=v, settings=registered_settings(v)) for v in STUDIES[study]]
    plan = dict(schema="learning-contract-diagnostic-v1", study=study, commit=h.command(["git", "rev-parse", "HEAD"]),
                source_hashes=h.code_hashes(), verification=proof,
                execution="github-only-standard-cpu-jobs",
                cases=cases,
                runtime=dict(max_workers=len(cases), local_workers=0, start_ram_gib=1.5, two_workers_ram_gib=4.,
                             pause_ram_gib=.5, worker_wall_s=1200, coordinator_wall_s=2700),
                primary_interactions=sum(c["settings"]["updates"]*c["settings"]["rollout"] for c in cases), gamma_fixed=1., heldout_test_executed=False,
                claim="Single-seed training diagnostic only, not a fair algorithm ranking",
                extension=("Final diagnostic cycle; no automatic further training, seeds or formal experiments."
                           if study == "role_credit" else
                           "No automatic long training. At most control/candidate confirmation on seed87035 after review."))
    plan["plan_hash"] = h.digest(plan)
    root.mkdir(parents=True, exist_ok=False)
    h.write_json(root/"plan.json", plan)


def load_plan(root):
    plan = h.read_json(root/"plan.json")
    if h.digest({k:v for k,v in plan.items() if k!="plan_hash"}) != plan["plan_hash"]:
        raise ValueError("Plan hash changed")
    if plan.get("study") not in STUDIES or plan["cases"] != [dict(variant=v, settings=registered_settings(v)) for v in STUDIES[plan["study"]]]:
        raise ValueError("Budget or predeclared cases changed")
    if h.code_hashes() != plan["source_hashes"] or h.command(["git","rev-parse","HEAD"]) != plan["commit"]:
        raise ValueError("Frozen code changed")
    return plan


def directory(root, case):
    s = case["settings"]
    return root/"cases"/case["variant"]/"cells/probe"/("%s_LR%d_seed%d" % (s["method"],s["lr"],s["seed"]))


def run(root, plan):
    if plan.get("execution") == "github-only-standard-cpu-jobs":
        raise ValueError("Do not duplicate the registered cloud cases locally")
    pending, active = list(plan["cases"]), []
    logs = root/"logs"
    logs.mkdir(exist_ok=False)
    started = time.monotonic()
    stopped = None
    def state(phase):
        h.write_json(root/"state.json",dict(phase=phase,pid=os.getpid(),
            worker_pids=[p.pid for _,p,_,_ in active],
            active=[c["variant"] for c,_,_,_ in active],pending=[c["variant"] for c in pending],
            stopped=stopped,free_ram_gib=h.free_ram_gib(),observed_utc=datetime.now(timezone.utc).isoformat()))
    with h.awake_lock(root):
        try:
            while pending or active:
                ram = h.free_ram_gib()
                if ram < plan["runtime"]["pause_ram_gib"] or time.monotonic()-started > plan["runtime"]["coordinator_wall_s"]:
                    stopped = "Resource or wall-time boundary"
                    pending.clear()
                    for c,_,_,_ in active:
                        marker=directory(root,c).parent/"STOP_AFTER_UPDATE"
                        marker.parent.mkdir(parents=True,exist_ok=True)
                        marker.touch(exist_ok=True)
                    state("stopping")
                limit = 2 if ram >= plan["runtime"]["two_workers_ram_gib"] else 1
                if not stopped and pending and len(active)<limit and ram>=plan["runtime"]["start_ram_gib"]:
                    case=pending.pop(0)
                    stdout=(logs/(case["variant"]+".log")).open("x",encoding="utf-8")
                    stderr=(logs/(case["variant"]+".err.log")).open("x",encoding="utf-8")
                    proc=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"worker",
                        "--out",str(root),"--case",case["variant"],"--execute"],cwd=ROOT,stdout=stdout,stderr=stderr)
                    active.append((case,proc,stdout,stderr))
                    state("running")
                    print("START",case["variant"],proc.pid,flush=True)
                for item in list(active):
                    case,proc,stdout,stderr=item
                    code=proc.poll()
                    if code is None:
                        continue
                    stdout.close();stderr.close();active.remove(item)
                    with (root/"resource_release.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(dict(variant=case["variant"],pid=proc.pid,exit_code=code,
                            free_ram_gib=h.free_ram_gib()))+"\n")
                    print("EXIT",case["variant"],code,flush=True)
                    if code:
                        stopped="Worker failed or safely paused: "+case["variant"]
                        pending.clear()
                    state("running" if active else "between_cases")
                if pending or active:
                    time.sleep(2)
            state("reporting")
            code=subprocess.run([sys.executable,str(Path(__file__).resolve()),"report","--out",str(root)],cwd=ROOT).returncode
            if code:
                stopped="Report or integrity audit requires review"
            state("needs_review" if stopped else "complete")
        finally:
            if active:
                for case,_,_,_ in active:
                    marker=directory(root,case).parent/"STOP_AFTER_UPDATE"
                    marker.parent.mkdir(parents=True,exist_ok=True);marker.touch(exist_ok=True)
                for _,proc,stdout,stderr in active:
                    proc.wait();stdout.close();stderr.close()


def report(root, plan):
    import paper_convergence as pc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries, points, audits = [], [], {}
    fig, axes=plt.subplots(1,2,figsize=(12,4))
    for case in plan["cases"]:
        path=directory(root,case)
        s=case["settings"]
        if (path/"manifest.json").exists():
            manifest=h.read_json(path/"manifest.json")
            cfg,train=pc.configuration(s)
            if manifest["environment"]!=cfg or manifest["training"]!=train:
                raise ValueError("Executed physical or estimator configuration differs")
            if s.get("training_profile") == "role_credit_v6":
                import torch
                saved = torch.load(path/"latest.pt", map_location="cpu", weights_only=False)
                frozen = train["frozen_actor_stages"]
                if saved["learner"]["frozen_actor_stages"] != frozen:
                    raise ValueError("Checkpoint actor-stage contract differs from manifest")
                with torch.random.fork_rng():
                    torch.manual_seed(s["seed"])
                    initial = pc.learner_for(s["method"], train, cfg).state(for_evaluation=True)
                for key, parameter in initial["actors"].items():
                    if key.split(".", 1)[0] in frozen and not torch.equal(
                            parameter, saved["learner"]["actors"][key]):
                        raise ValueError("Frozen actor parameter changed: " + key)
        audit_plan=dict(plan,stages={"probe":[dict(id=path.name,owner="local",settings=s)]},
                        runtime=dict(h.RUNTIME),gate=dict(tail_points=3,max_cost_ratio=1.25,
                                                         max_qos_increase=.05,max_tail_qos=.1))
        verdict=h.gate(audit_plan,path.parents[2],"probe")
        audits[case["variant"]]=verdict
        if not verdict["passed"]:
            continue
        val=[r for r in h.rows(path/"validation.csv") if r["method"]=="TMPPO"]
        series=[]
        for update in sorted({int(r["update"]) for r in val}):
            group=[r for r in val if int(r["update"])==update]
            row=dict(variant=case["variant"],update=update,
                     **{k:pc.validation_mean(group,k,"pooled_task_cost") for k in
                        ("mean_task_cost","qos_rate","active_wait_s","amplified_publication_ratio")})
            series.append(row);points.append(row)
        selected=h.read_json(path/"selection.json")
        tail=series[-3:]
        summary=dict(variant=case["variant"],initial_cost=series[0]["mean_task_cost"],
                     final_cost=series[-1]["mean_task_cost"],selected_update=selected["update"],
                     selected_cost=selected["validation_cost"],tail_cost=sum(r["mean_task_cost"] for r in tail)/3,
                     tail_qos=sum(r["qos_rate"] for r in tail)/3,
                     final_qos=series[-1]["qos_rate"],wall_s=h.read_json(path/"complete.json")["wall_s"])
        summary["confirmation_eligible"]=(selected["update"]>0 and summary["tail_cost"]<=.99*summary["initial_cost"]
                                           and summary["tail_qos"]<=series[0]["qos_rate"]+.01)
        summaries.append(summary)
        for ax,key in zip(axes,("mean_task_cost","qos_rate")):
            ax.plot([r["update"] for r in series],[r[key] for r in series],marker="o",label=case["variant"])
            ax.set(xlabel="Policy updates",ylabel=key);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Learning-contract ablation | same physical protocol | one seed | raw validation points")
    fig.tight_layout();fig.savefig(root/"ablation.png",dpi=170);plt.close(fig)
    pc.csv_write(root/"summary.csv",summaries);pc.csv_write(root/"validation_points.csv",points)
    h.write_json(root/"audit.json",dict(all_integrity_passed=len(summaries)==len(plan["cases"]),audits=audits,
                                      test_executed=False,publication_grade=False))
    h.write_json(root/"selection.json",dict(
        eligible=[s["variant"] for s in summaries if s["confirmation_eligible"]],
        final_learning_rate_allowed=False,auto_long_training=False,
        note="Eligibility permits another bounded validation experiment, not a success claim"))
    print(json.dumps(summaries,indent=2),flush=True)
    return len(summaries)==len(plan["cases"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("init","run","worker","report"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--verification",type=Path)
    parser.add_argument("--case",choices=ALL_VARIANTS)
    parser.add_argument("--study",choices=tuple(STUDIES),default="lambda_entropy")
    parser.add_argument("--execute",action="store_true")
    a=parser.parse_args();root=a.out.resolve()
    if a.command=="init":
        initialize(root,a.verification,a.study);return
    plan=load_plan(root)
    if a.command=="report":
        raise SystemExit(0 if report(root,plan) else 1)
    if not a.execute:
        parser.error("Explicit --execute required")
    if a.command=="run":
        run(root,plan);return
    if not a.case:
        parser.error("Worker needs --case")
    if h.free_ram_gib() < plan["runtime"]["start_ram_gib"]:
        raise RuntimeError("Insufficient memory before starting registered worker")
    case=next(c for c in plan["cases"] if c["variant"]==a.case)
    import paper_convergence as pc
    try:
        ok=pc.train_cell(directory(root,case),case["settings"],wall_seconds=plan["runtime"]["worker_wall_s"])
    except Exception as error:
        path=directory(root,case)
        if path.exists():h.write_json(path/"failure.json",dict(type=type(error).__name__,message=str(error)))
        raise
    raise SystemExit(0 if ok else 2)


if __name__=="__main__":
    main()

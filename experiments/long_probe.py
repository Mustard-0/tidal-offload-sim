"""Explicit one-seed 2000-update cloud exploration; not a passed formal matrix."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import hybrid_campaign as hybrid

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = re.compile(r"longprobe-campaign-\d{8}T\d{6}Z\Z")
TAG = re.compile(r"paper-long-probe-(\d{8}T\d{6}Z)\Z")


def build_plan(campaign, commit, hashes):
    if not CAMPAIGN.fullmatch(campaign):
        raise ValueError("Invalid explicit long-probe identity")
    cells = []
    for method in hybrid.METHODS:
        for lr in (1, 2):
            settings = hybrid.settings("formal", method, lr, 87021)
            settings["diagnostics_endpoints_only"] = True
            cells.append(dict(id="%s_LR%d_seed87021" % (method, lr), owner="github", settings=settings))
    plan = dict(schema="authorized-long-probe-v1", campaign=campaign, commit=commit,
        source_hashes=hashes, stages=dict(probe=cells),
        runtime=dict(hybrid.RUNTIME, github_workers=6, local_workers=0),
        gate=dict(tail_points=3, max_cost_ratio=1.25,max_qos_increase=.05,max_tail_qos=.10),
        authorization="User explicitly requested long training despite the failed short screen",
        screen_passed=False, exploratory=True, formal_matrix=False, publication_grade=False,
        training_interactions=1536000, final_test_executed=False,
        heldout_test_seeds=[9848031,9848047,9848063,9848089],
        extension_policy="No automatic extra seeds, reruns, formal matrix or other studies",
        claim="One training seed; validation only; no full-transcript DP guarantee")
    plan["plan_hash"] = hybrid.digest(plan)
    return plan


def load_plan(path):
    plan = hybrid.read_json(path)
    payload = {k:v for k,v in plan.items() if k!="plan_hash"}
    if hybrid.digest(payload) != plan["plan_hash"]:
        raise ValueError("Plan hash differs")
    expected = build_plan(plan["campaign"],plan["commit"],plan["source_hashes"])
    if any(plan.get(k)!=v for k,v in expected.items() if k!="plan_hash"):
        raise ValueError("Budgets, seeds or explicit authorization differ")
    if not plan.get("verification",{}).get("passed") or plan["verification"]["source_hashes"]!=plan["source_hashes"]:
        raise ValueError("Missing exact-source verification")
    if hybrid.code_hashes()!=plan["source_hashes"]:
        raise ValueError("Source changed since freezing")
    if hybrid.command(["git","rev-parse","HEAD"])!=plan["commit"]:
        raise ValueError("Commit differs")
    return plan


def initialize(out, verification):
    verified=hybrid.read_json(verification)
    if not verified.get("passed") or verified["source_hashes"]!=hybrid.code_hashes():
        raise ValueError("Verify exact source before freezing")
    if hybrid.command(["git","branch","--show-current"]) in ("main","master",""):
        raise ValueError("Use a development branch")
    if hybrid.command(["git","status","--porcelain","--","model","experiments",".github","config.json","requirements.txt"]):
        raise ValueError("Commit reviewed code before freezing")
    campaign="longprobe-campaign-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan=build_plan(campaign,hybrid.command(["git","rev-parse","HEAD"]),hybrid.code_hashes())
    plan["verification"]=verified
    plan["plan_hash"]=hybrid.digest({k:v for k,v in plan.items() if k!="plan_hash"})
    out.mkdir(parents=True,exist_ok=False)
    hybrid.write_json(out/"plan.json",plan)
    return plan


def prepare(out,repo,tag):
    match=TAG.fullmatch(tag or "")
    if not match or not re.fullmatch(r"[\w.-]+/[\w.-]+",repo or ""):
        raise ValueError("Invalid explicit repository/tag")
    out.mkdir(parents=True,exist_ok=True)
    campaign="longprobe-campaign-"+match.group(1)
    hybrid.command(["gh","release","download",campaign,"--repo",repo,"--pattern","plan.json","--dir",out])
    plan=load_plan(out/"plan.json")
    if plan["campaign"]!=campaign:
        raise ValueError("Release identity differs")
    return plan


def worker(plan,root,cell_id):
    hybrid.require_runtime(plan)
    cells=[c for c in plan["stages"]["probe"] if c["id"]==cell_id]
    if len(cells)!=1:
        raise ValueError("Unregistered cell")
    cell=cells[0]
    if hybrid.free_ram_gib()<plan["runtime"]["start_ram_gib"]:
        raise RuntimeError("Insufficient available RAM; do not start")
    out=root/"cells/probe"/cell_id
    if out.exists():
        raise FileExistsError("No automatic rerun or cloud resume")
    logs=root/"logs/probe"
    logs.mkdir(parents=True,exist_ok=True)
    args=hybrid.worker_command(cell,out,plan,"formal")
    with hybrid.awake_lock(root):
        with (logs/(cell_id+".log")).open("w",encoding="utf-8") as stdout, \
             (logs/(cell_id+".err.log")).open("w",encoding="utf-8") as stderr:
            process=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=stderr,
                                     text=True,encoding="utf-8",bufsize=1)
            hybrid.write_json(root/"worker.json",dict(pid=process.pid,cell=cell_id,exploratory=True))
            print("LONG_PROBE_START",cell_id,"pid",process.pid,flush=True)
            try:
                for line in process.stdout:
                    stdout.write(line)
                    stdout.flush()
                    print(line,end="",flush=True)
                    match=re.match(r"VALIDATE .+ update (\d+) cost ([\d.eE+-]+) best (\d+)",line)
                    if match:
                        publish_progress(plan,root,cell_id,match)
                code=process.wait()
            finally:
                if process.poll() is None:
                    (out.parent/"STOP_AFTER_UPDATE").touch()
                    process.wait()
    with (root/"resource_release.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(dict(cell=cell_id,exit_code=code,free_ram_gib=hybrid.free_ram_gib()))+"\n")
    return code


def publish_progress(plan,root,cell_id,match):
    import os
    run_id=os.environ.get("GITHUB_RUN_ID","")
    repo=os.environ.get("GITHUB_REPOSITORY","")
    attempt=os.environ.get("GITHUB_RUN_ATTEMPT","1")
    if not run_id.isdigit() or not attempt.isdigit() or not re.fullmatch(r"[\w.-]+/[\w.-]+",repo):
        return
    directory=root/"cells/probe"/cell_id/"progress"
    path=directory/("probe_progress_%s_run%s_attempt%s_update%06d.json" %
                    (cell_id,run_id,attempt,int(match.group(1))))
    hybrid.write_json(path,dict(plan_hash=plan["plan_hash"],cell=cell_id,run_id=run_id,attempt=attempt,
        validation_completed_update=int(match.group(1)),rounded_validation_cost=float(match.group(2)),
        best_update=int(match.group(3)),observed_utc=datetime.now(timezone.utc).isoformat(),
        note="Live validation diagnostic, not final task audit or durable-checkpoint attestation"))
    try:
        hybrid.command(["gh","release","upload",plan["campaign"],path,"--repo",repo])
    except (OSError,subprocess.CalledProcessError) as exc:
        print("PROGRESS_UPLOAD_WARNING",type(exc).__name__,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("init","prepare","worker","publish","audit"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--verification",type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--tag")
    parser.add_argument("--cell")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt",default="1")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    root=args.out.resolve()
    if args.command=="init":
        if not args.verification:
            parser.error("Supply --verification")
        plan=initialize(root,args.verification)
        print(json.dumps(dict(campaign=plan["campaign"],commit=plan["commit"],plan_hash=plan["plan_hash"])))
        return
    if args.command=="prepare":
        prepare(root,args.repo,args.tag)
        return
    plan=load_plan(root/"plan.json")
    if args.command=="audit":
        verdict=hybrid.gate(plan,root,"probe")
        verdict.update(exploratory=True,formal_matrix=False,publication_grade=False,
                       stability_passed_all=all(c["stable"] for c in verdict["cells"]) and len(verdict["cells"])==6)
        hybrid.write_json(root/"long_probe_audit.json",verdict)
        raise SystemExit(0 if verdict["passed"] else 1)
    if not args.execute:
        parser.error("Explicit --execute required")
    if args.command=="worker":
        raise SystemExit(worker(plan,root,args.cell))
    if not args.repo or not args.cell:
        parser.error("Publishing requires --repo and --cell")
    if args.cell!="preparation":
        cell=root/"cells/probe"/args.cell
        if cell.exists():
            derived=cell/"report"
            derived.mkdir(exist_ok=True)
            for source in cell.parent.iterdir():
                if source.is_file() and source.suffix in (".png",".csv",".json"):
                    shutil.copy2(source,derived/source.name)
    hybrid.publish_cloud_output(plan,root,args.repo,"probe",args.cell,args.run_id,args.attempt)


if __name__=="__main__":
    main()

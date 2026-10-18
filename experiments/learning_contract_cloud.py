"""Transfer only the four registered diagnostic cells through one public Release."""
import argparse
import json
from pathlib import Path
import re
import zipfile

import hybrid_campaign as h
from learning_contract_campaign import load_plan, directory, ALL_VARIANTS


def identity(repo,tag):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+",repo):
        raise ValueError("Invalid repository identity")
    match=re.fullmatch(r"(learning-contract|shared-team-value|role-actor-credit)-(\d{8}T\d{6}Z)",tag)
    if not match:
        raise ValueError("Invalid explicit diagnostic tag")
    prefixes={"learning-contract": "contract-campaign-", "shared-team-value": "shared-value-campaign-",
              "role-actor-credit": "role-credit-campaign-"}
    return prefixes[match.group(1)]+match.group(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","publish"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--repo",required=True)
    parser.add_argument("--tag",required=True)
    parser.add_argument("--case",choices=ALL_VARIANTS+("preparation",))
    parser.add_argument("--run-id")
    parser.add_argument("--attempt",default="1")
    args=parser.parse_args();root=args.out.resolve()
    release=identity(args.repo,args.tag)
    if args.command=="prepare":
        root.mkdir(parents=True,exist_ok=True)
        h.command(["gh","release","download",release,"--repo",args.repo,
                   "--pattern","plan.json","--dir",root])
        plan=load_plan(root)
        if plan.get("execution")!="github-only-standard-cpu-jobs":
            raise ValueError("Unregistered execution owner")
        return
    plan=load_plan(root)
    if not args.case or not re.fullmatch(r"\d+",args.run_id or "") or not args.attempt.isdigit():
        raise ValueError("Missing registered case or numeric run identity")
    if args.case=="preparation":
        paths=list((root.parent/"cloud_checks").glob("*"))
    else:
        case=next(c for c in plan["cases"] if c["variant"]==args.case)
        paths=[p for p in directory(root,case).rglob("*") if p.is_file()]
        paths+=list((root/"logs").glob(args.case+".*"))
    out=root/"uploads";out.mkdir(exist_ok=True)
    path=out/("contract_%s_run%s_attempt%s.zip"%(args.case,args.run_id,args.attempt))
    with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            name=("checks/"+p.name if args.case=="preparation" else p.relative_to(root).as_posix())
            archive.write(p,name)
        archive.writestr("execution_origin.json",json.dumps(dict(plan_hash=plan["plan_hash"],
            commit=plan["commit"],stage="contract-diagnostic",variant=args.case,
            run_id=args.run_id,attempt=args.attempt)))
    if path.stat().st_size>=2*1024**3:
        raise ValueError("Diagnostic result exceeds the registered transfer size")
    h.command(["gh","release","upload",release,path,"--repo",args.repo])


if __name__=="__main__":main()

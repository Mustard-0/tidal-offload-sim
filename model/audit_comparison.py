"""Audit every frozen comparison job, including unfavorable results."""
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
from common import read_csv, read_json, save_json, write_csv


def audit(run):
    run=Path(run); cfg=read_json(run/"config.json"); plan=read_json(run/"plan.json")
    hashes=read_json(run/"source_hashes.json")
    reference={}; summary=[]
    for job in plan["jobs"]:
        path=run/job["id"]; complete=read_json(path/"complete.json")
        manifest=read_json(path/"manifest.json")
        if manifest["source_hashes"]!=hashes or complete["job"]!=job:
            raise ValueError(f"Source/job mismatch: {job['id']}")
        expected=0 if job["method"]=="Greedy" else cfg["training"]["updates"]
        if complete["updates"]!=expected or complete["environment_steps"]!=expected*cfg["training"]["rollout_steps"]:
            raise ValueError(f"Wrong training budget: {job['id']}")
        if expected:
            train=read_csv(path/"train_raw.csv")
            if [int(r["update"]) for r in train]!=list(range(1,expected+1)):
                raise ValueError("Missing updates")
            if any(not math.isfinite(float(v)) for r in train for v in r.values() if v!=""):
                raise ValueError("Nonfinite training diagnostics")
            val=defaultdict(list)
            for r in read_csv(path/"validation_raw.csv"): val[int(r["update"])].append(r)
            points={0,*range(cfg["training"]["validation_interval"],expected+1,cfg["training"]["validation_interval"]),expected}
            if set(val)!=points or any({int(r["trace_seed"]) for r in rows}!=set(cfg["training"]["validation_seeds"]) for rows in val.values()):
                raise ValueError("Missing independent validation point/trace")
            costs={u:np.mean([float(r["mean_task_cost"]) for r in rows]) for u,rows in val.items()}
            if complete["selected_update"]!=min(costs,key=costs.get):
                raise ValueError("Checkpoint was not selected by independent validation")
        tests=read_csv(path/"test_raw.csv")
        if {int(r["trace_seed"]) for r in tests}!=set(cfg["training"]["test_seeds"]):
            raise ValueError("Missing independent test trajectory")
        tasks=defaultdict(list)
        for r in read_csv(path/"test_tasks.csv"): tasks[int(r["trace_seed"])].append(r)
        for test in tests:
            seed=int(test["trace_seed"]); cohort=tasks[seed]; n=int(test["arrivals"])
            if len(cohort)!=n or len({r["task_id"] for r in cohort})!=n:
                raise ValueError("Missing or duplicated task")
            signature={r["task_id"]:tuple(r[k] for k in ("user","region","arrival_s","task_profile","task_bits","task_cycles","deadline_s")) for r in cohort}
            paired=tuple(test[k] for k in ("trace_hash","phase_s","window_s","arrivals"))
            if seed not in reference: reference[seed]=(signature,paired)
            if signature!=reference[seed][0] or paired!=reference[seed][1]:
                raise ValueError(f"Unpaired external tasks: {job['id']}, seed {seed}")
            completed=[r for r in cohort if int(r["completed"])]
            e=cfg["environment"]
            measured={"completion_rate":len(completed)/n,"qos_rate":sum(int(r["qos"]) for r in cohort)/n,
                      "energy_j_per_arrival":sum(float(r["energy_j"]) for r in cohort)/n,
                      "delay_s":np.mean([float(r["end_s"])-float(r["arrival_s"]) for r in completed]),
                      "mean_task_cost":np.mean([e["weight_time"]*(float(r["end_s"])-float(r["arrival_s"]))/e["time_ref_s"]+e["weight_energy"]*float(r["energy_j"])/e["energy_ref_j"]+e["weight_violation"]*int(r["qos"]) for r in cohort])}
            for key,value in measured.items():
                if not np.isclose(value,float(test[key]),rtol=1e-7,atol=1e-9):
                    raise ValueError(f"Task-level accounting failed: {key}, {job['id']}")
        row=dict(job=job["id"],method=job["method"],lr=job["lr"],selected_update=complete["selected_update"],
                 training_updates=expected,training_seeds=int(expected>0),test_traces=len(tests))
        for key in ("delay_s","energy_j_per_arrival","qos_rate","completion_rate","mean_task_cost","ql_m","active_wait_s","amplified_publication_ratio"):
            values=[float(r[key]) for r in tests if r[key] not in (None,"")]
            row[key]=float(np.mean(values)) if values else None
            row[key+"_trace_sd"]=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(row)
    write_csv(run/"audited_summary.csv",summary)
    save_json(run/"audit.json",dict(passed=True,jobs=len(summary),source_hashes=hashes,
                                  evidence="one training seed; phase-stratified short-window evaluation, not full-cycle formal evidence"))
    return summary


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("run",type=Path)
    print(audit(parser.parse_args().run))

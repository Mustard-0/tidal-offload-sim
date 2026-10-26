"""Frozen four-configuration comparison, with at most two independent workers."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, read_json, save_json, source_hashes, write_csv
from environment import GreedyPolicy
from evaluation import evaluate
from run import new_directory, plan, train_job


def live(pid):
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle: return False
        code=wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


def run_worker(out, job_id, device):
    if read_json(out/"source_hashes.json")!=source_hashes():
        raise ValueError("Worker source differs from frozen snapshot")
    cfg=read_json(out/"config.json")
    job=next(j for j in read_json(out/"plan.json")["jobs"] if j["id"]==job_id)
    path=out/job_id
    try:
        done=train_job(cfg,job,path,device,plot_progress=False)
    except Exception as exc:
        save_json(path/"failure.json",{"job":job,"error":repr(exc)})
        raise
    return 0 if done else 2


def launch(config_path, out, resume, workers, device):
    cfg=read_json(config_path)
    if resume:
        if read_json(out/"config.json")!=cfg or read_json(out/"source_hashes.json")!=source_hashes():
            raise ValueError("Resume requires the saved configuration and source")
        meta=read_json(out/"launcher.json")
        if live(meta["pid"]) or any(live(p) for p in meta.get("worker_pids",[])):
            raise RuntimeError("An existing launcher/worker is still alive; do not duplicate it")
    else:
        out=new_directory(f'convergence_{cfg["training"]["updates"]}updates',out)
        save_json(out/"config.json",cfg)
        save_json(out/"source_hashes.json",source_hashes())
        save_json(out/"plan.json",dict(kind="convergence",jobs=plan(cfg,"convergence",cfg["training"]["seed"]),formal_eval=False))
    jobs=read_json(out/"plan.json")["jobs"]
    meta=dict(pid=os.getpid(),worker_pids=[],workers=workers,device=device,status="baseline",run=str(out))
    save_json(out/"launcher.json",meta)
    print(f"RUN={out} PARENT_PID={os.getpid()}",flush=True)
    if not (out/"greedy_validation.csv").exists():
        rows,_,_=evaluate(GreedyPolicy(cfg["environment"]),cfg,"Greedy",cfg["training"]["validation_seeds"])
        write_csv(out/"greedy_validation.csv",rows)
    greedy=next(j for j in jobs if j["method"]=="Greedy")
    train_job(cfg,greedy,out/greedy["id"],"cpu",plot_progress=False)
    # Interleave algorithms so both LR1 configurations run first.
    queue=sorted([j for j in jobs if j["method"]!="Greedy" and not (out/j["id"]/"complete.json").exists()],key=lambda j:(j["lr"],j["method"]!="TMAPO"))
    active=[]; stopped=False; failed=False
    while queue or active:
        if (out/"STOP_AFTER_UPDATE").exists(): stopped=True
        while queue and len(active)<workers and not stopped:
            job=queue.pop(0)
            stdout=(out/(job["id"]+".log")).open("a",encoding="utf-8")
            stderr=(out/(job["id"]+".err.log")).open("a",encoding="utf-8")
            flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
            process=subprocess.Popen([sys.executable,str(out/"source"/"launch_comparison.py"),"--worker",job["id"],"--out",str(out),"--device",device],cwd=out/"source",stdout=stdout,stderr=stderr,creationflags=flags)
            active.append((process,job,stdout,stderr))
            print(f"START={job['id']} PID={process.pid}",flush=True)
        meta.update(worker_pids=[p.pid for p,_,_,_ in active],status="stopping" if stopped else "training")
        save_json(out/"launcher.json",meta)
        if stopped and not active: break
        time.sleep(10)
        remaining=[]
        for process,job,stdout,stderr in active:
            code=process.poll()
            if code is None:
                remaining.append((process,job,stdout,stderr)); continue
            stdout.close(); stderr.close()
            print(f"END={job['id']} EXIT={code}",flush=True)
            if code!=0:
                stopped=True
                failed|=code!=2
                save_json(out/"STOP_AFTER_UPDATE",{"reason":"peer worker failed or paused","job":job["id"],"exit_code":code})
        active=remaining
        from plots import plot_run
        try:
            plot_run(out)
        except Exception as exc:
            save_json(out/"plot_failure.json",{"error":repr(exc),"raw_training_preserved":True})
            print(f"PLOT_ERROR={exc!r}; training continues",flush=True)
    meta.update(worker_pids=[],status="failed" if failed else "paused" if stopped else "complete")
    meta["plot_error"]=(out/"plot_failure.json").exists()
    save_json(out/"launcher.json",meta)
    if not stopped:
        from audit_comparison import audit
        audit(out)
    print(f"STATUS={meta['status']} RUN={out}",flush=True)
    return 1 if failed else 2 if stopped else 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--workers",type=int,choices=[1,2],default=2)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=parser.parse_args()
    if args.worker:
        if not args.out: parser.error("worker requires --out")
        sys.exit(run_worker(args.out,args.worker,args.device))
    if args.resume and not args.out: parser.error("resume requires --out")
    sys.exit(launch(args.config,args.out,args.resume,args.workers,args.device))

"""Single-seed, sequential pilot suite. No training occurs without a train command."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from common import ROOT,read_json,save_json,write_csv,manifest,seed_all,digest,source_hashes
from environment import Environment
from learners import make_policy
from evaluation import evaluate,quality_check,value_table


def training_phase(config, seed, episode):
    c, t = config["environment"], config["training"]
    strata = t.get("training_phase_strata", 1)
    if strata < 1:
        raise ValueError("training_phase_strata must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8, episode//strata]))
    order = rng.permutation(strata)
    return float(((order[episode%strata]+.5)*c["tide_period_s"]/strata) % c["tide_period_s"])


def plan(config,kind,seed):
    e=config["experiments"]
    jobs=[]
    if kind=="privacy":
        for method in e["privacy_methods"]:
            for eps in e["epsilons"]:
                jobs.append(dict(method=method,epsilon_req=eps,users_per_region=config["environment"]["users_per_region"],lr=0))
    elif kind=="main":
        for method in e["main_methods"]:
            for users in e["users_per_region"]:
                jobs.append(dict(method=method,epsilon_req=config["environment"]["epsilon_req"],users_per_region=users,lr=0))
    elif kind in ("convergence", "baseline-tuning"):
        methods = ("TMAPO","IPPO") if kind == "convergence" else ("TMAPO","IPPO","MARATD3-Adapted","DPFL-Adapted")
        for method in methods:
            for lr in (0,1):
                jobs.append(dict(method=method,epsilon_req=config["environment"]["epsilon_req"],users_per_region=config["environment"]["users_per_region"],lr=lr))
    for j in jobs:
        j.update(seed=seed,kind=kind)
        j["id"]=f'{j["method"]}_M{j["users_per_region"]}_eps{j["epsilon_req"]:g}_LR{j["lr"]+1}_s{seed}'
    if kind=="convergence":
        jobs.insert(0,dict(method="Greedy",epsilon_req=config["environment"]["epsilon_req"],
                           users_per_region=config["environment"]["users_per_region"],lr=0,seed=seed,kind=kind,
                           id=f'Greedy_M{config["environment"]["users_per_region"]}_eps{config["environment"]["epsilon_req"]:g}_LR1_s{seed}'))
    return jobs


def selected_learning_rates(config, paths):
    selected = {}
    for path in paths:
        if path is None:
            continue
        saved = read_json(path.parent/"config.json")
        environment = {k:v for k,v in config["environment"].items() if k != "users_per_region"}
        previous_environment = {k:v for k,v in saved["environment"].items() if k != "users_per_region"}
        if previous_environment != environment:
            raise ValueError("Learning-rate selection used a different environment; retune with this configuration")
        for key, value in config["training"].items():
            if key not in ("updates", "actor_lr", "critic_lr", "seed") and saved["training"].get(key) != value:
                raise ValueError(f"Training protocol changed since learning-rate selection: {key}")
        hashes = read_json(path.parent/"source_hashes.json")
        current = source_hashes()
        for name in ("environment.py", "physics.py", "privacy.py", "workload.py", "learners.py", "evaluation.py"):
            if hashes.get(name) != current[name]:
                raise ValueError(f"Learning-rate selection used different model code: {name}")
        for method, entry in read_json(path).items():
            lr = entry["lr"]
            if not isinstance(lr, int) or not 0 <= lr < len(config["experiments"]["learning_rates"]):
                raise ValueError(f"Invalid learning-rate index for {method}")
            if saved["experiments"]["learning_rates"][lr] != config["experiments"]["learning_rates"][lr]:
                raise ValueError("Selected learning-rate values changed")
            selected[method] = entry
    return selected


def apply_selection(jobs, selected):
    for j in jobs:
        key = "TMAPO" if j["method"].startswith("TMAPO") else j["method"]
        if key in selected:
            j["lr"] = selected[key]["lr"]
            j["id"] = j["id"].rsplit("_LR",1)[0]+f'_LR{j["lr"]+1}_s{j["seed"]}'
    return jobs


def new_directory(kind,requested=None):
    out=Path(requested) if requested else ROOT/"runs"/(kind+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True,exist_ok=False)
    (out/"source").mkdir()
    for p in ROOT.glob("*.py"): shutil.copy2(p,out/"source"/p.name)
    return out


def atomic_torch(path,data):
    tmp=path.with_suffix(".tmp")
    torch.save(data,tmp); os.replace(tmp,path)


def rng_state():
    return {"numpy":np.random.get_state(),"python":random.getstate(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(r):
    np.random.set_state(r["numpy"]); random.setstate(r["python"]); torch.set_rng_state(r["torch"].cpu())
    if r["cuda"] and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in r["cuda"]])


def train_job(config,job,path,device,formal=False,wall_seconds=None,plot_progress=True):
    config=copy.deepcopy(config); c=config["environment"]; t=config["training"]
    c.update(users_per_region=job["users_per_region"],epsilon_req=job["epsilon_req"])
    t["actor_lr"],t["critic_lr"]=config["experiments"]["learning_rates"][job["lr"]]
    if (path/"complete.json").exists(): return True
    path.mkdir(parents=True,exist_ok=True)
    expected=manifest(config,job)
    expected.update(device=str(device),torch_version=str(torch.__version__),formal_evaluation=bool(formal))
    if (path/"manifest.json").exists() and read_json(path/"manifest.json")!=expected:
        raise ValueError("Refusing resume with changed config, source, or platform; use a new output directory")
    save_json(path/"manifest.json",expected)
    seed_all(job["seed"],t["torch_threads"])
    policy=make_policy(job["method"],config,device)
    seed=job["seed"]; episode=0; env_steps=0; start_update=0; best=math.inf
    def make_env(ep):
        s=seed+100003*ep
        phase=training_phase(config,seed,ep)
        return Environment(c,s,job["method"],float(phase))
    env=make_env(episode)
    train_rows=[]; validation_rows=[]
    latest=path/"latest.pt"
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        policy.load(saved["policy"]); restore_rng(saved["rng"])
        env=saved["environment"]; episode=saved["episode"]; env_steps=saved["env_steps"]
        start_update=saved["update"]; best=saved["best"]
        train_rows=saved["train_rows"]; validation_rows=saved["validation_rows"]
    clock=time.monotonic()
    def checkpoint(update):
        atomic_torch(latest,{"policy":policy.state(),"rng":rng_state(),"environment":env,
                            "episode":episode,"env_steps":env_steps,"update":update,"best":best,
                            "train_rows":train_rows,"validation_rows":validation_rows})
        if train_rows: write_csv(path/"train_raw.csv",train_rows)
        if validation_rows: write_csv(path/"validation_raw.csv",validation_rows)
    def validate(update):
        nonlocal best
        keep=rng_state()
        eval_policy=make_policy(job["method"],config,device)
        eval_policy.load(copy.deepcopy(policy.state(for_evaluation=True)))
        vals,_,_=evaluate(eval_policy,config,job["method"],t["validation_seeds"],formal)
        restore_rng(keep)
        cost=float(np.mean([v["mean_task_cost"] for v in vals]))
        validation_rows.extend(dict(update=update,environment_steps=env_steps,**v) for v in vals)
        if cost<best:
            best=cost
            atomic_torch(path/"best_validation.pt",{"policy":policy.state(for_evaluation=True),"update":update,"environment_steps":env_steps,"validation_cost":cost})
        print(f'{job["id"]} update={update}/{t["updates"]} steps={env_steps} validation_cost={cost:.4f}',flush=True)
        return cost
    if job["method"] not in ("Greedy","Local-Only"):
        if not validation_rows:
            validate(0)
            checkpoint(0)
        episode_total=t["episode_steps"]+math.ceil(c["lifetime_s"]/c["dt_s"])+2
        for update in range(start_update+1,t["updates"]+1):
            policy.set_progress(update, t["updates"])
            rewards=[]
            interaction_start=time.monotonic()
            for _ in range(t["rollout_steps"]):
                policy.deterministic=False
                reward,records,context,glob=env.step(policy,arrivals=env.slot<t["episode_steps"])
                done=env.slot>=episode_total
                if done and env.tasks:
                    raise RuntimeError("Training arrival cohort did not drain before the fixed episode boundary")
                policy.remember(env.agent_rewards(t.get("local_credit_fraction", 0.)),records,done)
                env_steps+=1; rewards.append(reward)
                if done:
                    episode+=1; env=make_env(episode); policy.reset_history()
                    if t.get("target_episodes") and episode>=t["target_episodes"]:
                        break
            context,glob,*_=env.observations()
            if device == "cuda": torch.cuda.synchronize()
            interaction_seconds=time.monotonic()-interaction_start
            update_start=time.monotonic()
            diagnostics=policy.update(context,glob)
            if device == "cuda": torch.cuda.synchronize()
            row=dict(update=update,environment_steps=env_steps,completed_episodes=episode,
                     simulated_seconds=env_steps*c["dt_s"],training_phase_s=env.phase_s,
                     interaction_seconds=interaction_seconds,optimization_seconds=time.monotonic()-update_start,
                     actor_lr=policy.aopt.param_groups[0]["lr"],critic_lr=policy.copt.param_groups[0]["lr"],
                     mean_step_reward=float(np.mean(rewards)),**diagnostics)
            if not all(math.isfinite(v) for v in row.values()):
                raise FloatingPointError(f"Nonfinite training diagnostic: {row}")
            train_rows.append(row)
            if update%t["validation_interval"]==0 or update==t["updates"]:
                validate(update)
            stop_requested=(path/"STOP_AFTER_UPDATE").exists() or (path.parent/"STOP_AFTER_UPDATE").exists()
            stopping=bool((stop_requested or (wall_seconds and time.monotonic()-clock>=wall_seconds)) and update<t["updates"])
            if update%t.get("checkpoint_interval",10)==0 or update%t["validation_interval"]==0 or update==t["updates"] or stopping:
                checkpoint(update)
                if plot_progress and (update%t["validation_interval"]==0 or update==t["updates"]):
                    from plots import plot_run
                    plot_run(path.parent)
            else:
                write_csv(path/"train_raw.csv",train_rows)
            if stopping:
                save_json(path/"paused.json",{"reason":"stop marker" if stop_requested else "wall-clock budget","update":update,"resume":"same command --resume RUN"})
                return False
        best_state=torch.load(path/"best_validation.pt",map_location=device,weights_only=False)
        policy.load(best_state["policy"])
        chosen=best_state["update"]
    else:
        chosen=0
    tests,tasks,batches=evaluate(policy,config,job["method"],t["test_seeds"],formal,collect=True)
    write_csv(path/"test_raw.csv",tests)
    if tasks: write_csv(path/"test_tasks.csv",tasks)
    if batches: write_csv(path/"test_batches.csv",batches)
    save_json(path/"complete.json",{"job":job,"updates":len(train_rows),"environment_steps":env_steps,
              "completed_training_episodes":episode,"budget_unit":"episodes" if t.get("target_episodes") else "policy_updates",
              "best_validation_cost":best if math.isfinite(best) else None,"selected_update":chosen,
              "evaluation":"full-cycle" if formal else "short pilot, not final paper evidence",
              "test_mean_cost":float(np.mean([r["mean_task_cost"] for r in tests]))})
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["plan","check","value","privacy","convergence","baseline-tuning","main","plot"])
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--out",type=Path); parser.add_argument("--resume",type=Path)
    parser.add_argument("--run",type=Path,help="Existing run to plot")
    parser.add_argument("--kind",choices=["privacy","main","convergence","baseline-tuning"],default="privacy")
    parser.add_argument("--seed",type=int,default=2026)
    budget=parser.add_mutually_exclusive_group()
    budget.add_argument("--updates",type=int)
    budget.add_argument("--episodes",type=int,help="Complete training episodes, distinct from policy updates; potentially expensive")
    parser.add_argument("--rollout-steps",type=int)
    parser.add_argument("--episode-steps",type=int); parser.add_argument("--hidden",type=int)
    parser.add_argument("--validation-interval",type=int)
    parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    parser.add_argument("--job",help="Run one exact ID from plan (recommended first)")
    parser.add_argument("--method"); parser.add_argument("--epsilon",type=float)
    parser.add_argument("--users",type=int,help="Per-region users, never global total")
    parser.add_argument("--states",type=int); parser.add_argument("--repeats",type=int)
    parser.add_argument("--samples",type=int)
    parser.add_argument("--formal-eval",action="store_true",help="300 s cohort after 30 s warmup; no extra training seeds")
    parser.add_argument("--max-seconds",type=float,help="Stop at an update boundary and save an exact checkpoint")
    parser.add_argument("--selection",type=Path,help="Convergence selection.json for main suite")
    parser.add_argument("--baseline-selection",type=Path,help="Validation learning-rate selection for adapted baselines")
    args=parser.parse_args()
    for name in ("users","states","repeats","samples","max_seconds"):
        if getattr(args,name) is not None and getattr(args,name)<=0:
            parser.error(name+" must be positive")
    if args.command=="plot":
        from plots import plot_run
        if not args.run: parser.error("plot requires --run")
        plot_run(args.run); return
    config=read_json(args.config)
    for arg,key in ((args.updates,"updates"),(args.rollout_steps,"rollout_steps"),(args.episode_steps,"episode_steps"),(args.hidden,"hidden"),(args.validation_interval,"validation_interval")):
        if arg is not None:
            if arg<1: parser.error(key+" must be positive")
            config["training"][key]=arg
    if config["training"]["hidden"]%4: parser.error("hidden must be divisible by 4")
    if args.episodes is not None:
        if args.episodes < 1: parser.error("episodes must be positive")
        t=config["training"]; c=config["environment"]
        t["target_episodes"]=args.episodes
        total=t["episode_steps"]+math.ceil(c["lifetime_s"]/c["dt_s"])+2
        t["updates"]=math.ceil(args.episodes*total/t["rollout_steps"])
    if args.users:
        config["environment"]["users_per_region"]=args.users
    kind=args.kind if args.command=="plan" else args.command
    jobs=plan(config,kind,args.seed)
    if args.command=="plan":
        print(json.dumps(jobs,indent=2)); return
    if args.command in ("check","value"):
        out=new_directory(args.command,args.out); save_json(out/"config.json",config)
        if args.command=="check": quality_check(config,out,args.samples)
        else:
            quality_check(config,out,args.samples or 10000)
            value_table(config,out,args.states or config["experiments"]["value_states"],args.repeats or config["experiments"]["value_repeats"],args.seed)
        print(out); return
    selected = selected_learning_rates(config, (args.selection, args.baseline_selection))
    if kind in ("main", "privacy"):
        apply_selection(jobs, selected)
    if args.job: jobs=[j for j in jobs if j["id"]==args.job]
    if args.method: jobs=[j for j in jobs if j["method"]==args.method]
    if args.epsilon is not None: jobs=[j for j in jobs if j["epsilon_req"]==args.epsilon]
    if args.users and kind=="main": jobs=[j for j in jobs if j["users_per_region"]==args.users]
    if not jobs: parser.error("No matching jobs; use plan to inspect IDs")
    for j in jobs:
        key = "TMAPO" if j["method"].startswith("TMAPO") else j["method"]
        if args.formal_eval and kind in ("main", "privacy") and j["method"] not in ("Greedy", "Local-Only") and key not in selected:
            parser.error(f"Formal {key} requires validation-selected learning rate; run baseline-tuning first")
    if args.resume:
        out=args.resume
        if not out.is_dir(): parser.error("resume directory missing")
        if read_json(out/"config.json")!=config: parser.error("Use the exact saved config with --config for resume")
        if read_json(out/"source_hashes.json")!=source_hashes(): parser.error("Source changed; run the saved source or create a new run")
    else:
        out=new_directory(kind,args.out)
        save_json(out/"config.json",config); save_json(out/"source_hashes.json",source_hashes())
    requested=dict(kind=kind,jobs=jobs,formal_eval=args.formal_eval)
    if (out/"plan.json").exists() and read_json(out/"plan.json")!=requested:
        parser.error("Resume must preserve the job plan and evaluation protocol")
    save_json(out/"plan.json",requested)
    if kind in ("convergence", "baseline-tuning") and not (out/"greedy_validation.csv").exists():
        from environment import GreedyPolicy
        rows,_,_=evaluate(GreedyPolicy(config["environment"]),config,"Greedy",config["training"]["validation_seeds"],args.formal_eval)
        write_csv(out/"greedy_validation.csv",rows)
    clock=time.monotonic()
    for job in jobs:
        remaining=None if args.max_seconds is None else args.max_seconds-(time.monotonic()-clock)
        if remaining is not None and remaining<=0: break
        try:
            done=train_job(config,job,out/job["id"],args.device,args.formal_eval,remaining)
        except Exception as exc:
            save_json(out/"failure.json",{"job":job,"error":repr(exc)})
            raise
        if not done: break
    from plots import plot_run
    plot_run(out)
    print(out,flush=True)


if __name__=="__main__": main()

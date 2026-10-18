#!/usr/bin/env python3
"""Bounded actual W/L/O + route-PPO train/evaluate smoke test.

Runs the repo's existing 3-head torch PPO with a centralized critic, but
'new_method_environment' routes only USER and ROUTE heads; old BATCH head
is intentionally unused. The new protocol batches only same-slot O choices
among distinct users with the same public region and task class.

This validates training and physical metrics, NOT final manuscript claims:
single training seed, short simulated trajectories, total M=90, no burst
Markov chain, no full 2500-update budget, conditional one-batch DP only.
"""
from __future__ import annotations
import copy,csv,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model"))
from new_method_environment import NewManuscriptEnvironment
from environment import Task, N_ROUTE, OBS_DIM
from learners import PPO

OUTPUT=ROOT/"results"/"new_method_joint_wlo_smoke"
EVAL_SEEDS=(481953,729604)
BASE_USER_PER_REGION=30
BASE_EPS=.5
UPDATES=int(os.getenv("SMOKE_UPDATES","8"))
ROLLOUT=int(os.getenv("SMOKE_ROLLOUT","48"))
ARRIVAL_STEPS=int(os.getenv("SMOKE_TEST_ARRIVAL_STEPS","500"))
POLICIES=("mapppo_wlo_route","all_local","immediate_remote",
          "public_greedy_wlo","deadline_wait_remote")


def new_config():
    old=json.loads((ROOT/"config.json").read_text())
    c=copy.deepcopy(old["environment"])
    t=copy.deepcopy(old["training"])
    c.update(
      users_per_region=BASE_USER_PER_REGION,
      # Protocol/device parameters from draft; task profile order is public.
      task_profiles=[
        dict(name="interactive",probability=1/3,bits=50_000.,cycles=80e6,deadline_s=.20),
        dict(name="compute",probability=1/3,bits=400_000.,cycles=240e6,deadline_s=.35),
        dict(name="data",probability=1/3,bits=800_000.,cycles=400e6,deadline_s=.55)],
      task_bits=400_000.,task_cycles=240e6,deadline_s=.35,
      cpu_bs=80e9,cpu_sat=30e9,cpu_ue=1.2e9,
      bandwidth_bs_hz=400e6,bandwidth_sat_hz=20e6,
      wait_max_s=.05,prediction_horizon_s=.05,
      epsilon_req=BASE_EPS,delta_req=1e-6,
      weight_time=.25,weight_energy=.25,
      weight_quality=.25,weight_violation=.25,
      quality_ref_m=1000.,time_ref_s=.5,energy_ref_j=.24,
      project_public_region=True,
      min_service_reserve_s=.04,
      # A small *training smoke-test* demand; explicitly NOT manuscript rho=.5.
      arrival_per_user_s=.8,
      lifetime_s=2.,tide_period_s=300.)
    # Do not leak high-cost prior to W: match bounded event-time rewards.
    t.update(hidden=32,epochs=1,minibatch=256,torch_threads=1,
             ppo_clip=.2,actor_cost_prior_scale=.15,
             actor_lr=3e-4,critic_lr=7e-4,
             gamma=.99,gae_lambda=.95)
    return c,t


class SimpleBaseline:
    def __init__(self,name,cfg):
        self.name,self.c=name,cfg
    def begin(self,*args): pass
    def act(self,obs,mask,stage,agents):
        answers=[]
        for feature,allowed in zip(obs,mask):
            if stage=="route":
                time_cost=feature[:N_ROUTE]*self.c["weight_time"]
                energy=feature[N_ROUTE:2*N_ROUTE]*self.c["weight_energy"]
                remaining=feature[45]*self.c["time_ref_s"]
                violation=(feature[:N_ROUTE]*self.c["time_ref_s"]>remaining)
                score=time_cost+energy+violation*self.c["weight_violation"]
                score=np.where(allowed,score,1e6)
                answers.append(int(np.argmin(score)))
            elif stage=="user":
                # 0 WAIT, 1 LOCAL, 2 OFFLOAD
                if self.name=="all_local":
                    pref=(1,2,0)
                elif self.name=="immediate_remote":
                    pref=(2,1,0)
                elif self.name=="deadline_wait_remote":
                    age=float(feature[12])*.5
                    remaining=float(feature[14])*.5
                    pref=(0,2,1) if age<.04 and remaining>.10 else (2,1,0)
                else:  # Greedy compares public estimated costs
                    pref=(1,2,0) if feature[17]<=feature[18] else (2,1,0)
                answers.append(next(i for i in pref if allowed[i]))
            else:raise AssertionError(stage)
        return np.asarray(answers,dtype=int),[]


def test_group_invariant(cfg):
    cfg=copy.deepcopy(cfg);cfg["users_per_region"]=3
    env=NewManuscriptEnvironment(cfg,seed=411,collect=True)
    for u,profile in ((0,"interactive"),(1,"interactive"),(2,"compute")):
        j=Task(u,u,0,0.,50_000.,80e6,.2,profile=profile,arrival_slot=0)
        env.tasks[u]=j;env.pending[u].append(u);env.arrivals+=1;env.measure_arrivals+=1
    rule=SimpleBaseline("immediate_remote",cfg)
    env.step(rule,arrivals=False)
    groups=sorted((x["n_distinct"],x["task_class"]) for x in env.batch_rows)
    assert groups==[(1,"compute"),(2,"interactive")],groups
    assert env.step_cost>0
    assert env.tasks[0].batch_n==2 and env.tasks[2].batch_n==1
    # Report for each source user is sampled once; class never co-mingles.
    print("INVARIANT PASS grouped public class, distinct users, QL reward",flush=True)


def rollout(policy,cfg,seed,arrival_steps,training=False):
    env=NewManuscriptEnvironment(cfg,seed=seed,measure_start=0,
                                 measure_end=arrival_steps*cfg["dt_s"],collect=True)
    warm=0
    for slot in range(arrival_steps+math.ceil(cfg["lifetime_s"]/cfg["dt_s"])+10):
        arrivals=slot<arrival_steps
        ret,records,context,glob=env.step(policy,arrivals=arrivals)
        if training:
            policy.remember(env.agent_rewards(.7),records,False)
        warm+=1
        if not arrivals and not env.tasks and env.slot>arrival_steps:
            break
    else:
        raise RuntimeError("Drain boundary exceeded; check task state machine")
    metrics=env.metrics()
    return metrics,env


def main():
    t0=time.monotonic()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    c,t=new_config()
    torch.set_num_threads(1)
    test_group_invariant(c)
    torch.manual_seed(87021);np.random.seed(87021)
    trainer=PPO("TMAPO",t,c,device="cpu")
    train=[]
    env=NewManuscriptEnvironment(c,seed=87021,phase_s=40.)
    for update in range(1,UPDATES+1):
        rlog=[]
        trainer.set_progress(update,UPDATES)
        for _ in range(ROLLOUT):
            r,recs,ctx,glob=env.step(trainer)
            trainer.remember(env.agent_rewards(.7),recs,False)
            rlog.append(r)
        ctx,glob,*_=env.observations()
        diag=trainer.update(ctx,glob)
        row=dict(update=update,simulated_s=env.time,
                 mean_reward=float(np.mean(rlog)),
                 **{k:float(v) for k,v in diag.items()
                    if isinstance(v,(int,float,np.floating))})
        train.append(row)
        print("TRAIN",json.dumps(row),flush=True)
    trainer.deterministic=True
    results=[]
    batch_records=[]
    for seed in EVAL_SEEDS:
        for name in POLICIES:
            p=trainer if name=="mapppo_wlo_route" else SimpleBaseline(name,c)
            metrics,e=rollout(p,c,seed,ARRIVAL_STEPS)
            results.append(dict(seed=seed,method=name,**metrics))
            for row in e.batch_rows:
                batch_records.append(dict(seed=seed,method=name,**row))
            print("TEST",seed,name, json.dumps(
                 {k:metrics.get(k) for k in ("arrivals","completion_rate",
                   "qos_rate","delay_s","energy_j_per_arrival","ql_m",
                   "publication_ratio","report_weighted_batch_n",
                   "amplified_publication_ratio","mean_task_cost")}),flush=True)
    def save(path,rows):
        if not rows:return
        fields=list(rows[0])
        with (OUTPUT/path).open("w",encoding="utf-8",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
            wr.writeheader();wr.writerows(rows)
    save("train.csv",train)
    save("test_by_seed.csv",results)
    save("batch_traces.csv",batch_records)
    summary=[]
    for name in POLICIES:
        xs=[x for x in results if x["method"]==name]
        r=dict(method=name)
        for key in ("arrivals","completion_rate","qos_rate","delay_s",
          "energy_j_per_arrival","ql_m","publication_ratio",
          "report_weighted_batch_n","amplified_publication_ratio",
          "mean_task_cost"):
            vals=[float(x[key]) for x in xs if x[key] is not None]
            r[key]=float(np.mean(vals)) if vals else float("nan")
        summary.append(r)
    save("summary.csv",summary)
    # The "all_local" baseline can offload when local queue admission is
    # masked, hence it is a local-preferred benchmark, not strict local-only.
    assert any(r["publication_ratio"]>0 for r in summary if r["method"]=="immediate_remote")
    manifest=dict(
      experiment="New manuscript user-WLO plus same-slot class shuffle, learned route PPO smoke",
      scope="partial real model TRAIN/EVAL, not formal 2500-update TMAPO",
      updates=UPDATES,rollout_steps=ROLLOUT,
      trained_actor_heads=["user-WLO","route-11"],
      old_unused_actor_head="batch",
      training_seed=87021,test_seeds=EVAL_SEEDS,
      eval_arrival_steps=ARRIVAL_STEPS,
      total_users=3*c["users_per_region"],arrival_rate_per_user_s=c["arrival_per_user_s"],
      simulated_environment=c,training_hyperparameters=t,
      ql_definition="Euclidean location error AFTER common public-region projection",
      quality_ref_m=1000,
      privacy_postprocessing="public fixed-radius disk projection (no epsilon relaxation)",
      assumptions=["Markov bursts not modeled","fixed balanced region populations",
        "one task HOL per user per slot","same public region and public class",
        "conditional one-batch DP accounting, not longitudinal proof",
        "exact user action timing observability not in DP accountant",
        "single CPU pilot, two test seeds"],
      elapsed_s=time.monotonic()-t0)
    (OUTPUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key,title,scale in (
        ("qos_rate","QoS violation (%)",100),
        ("completion_rate","Task completion (%)",100),
        ("delay_s","Completed-task delay (ms)",1000),
        ("energy_j_per_arrival","Mean energy per arrival (J)",1),
        ("ql_m","Mean QL of published reports (m)",1),
        ("publication_ratio","Offloaded/published share (%)",100),
        ("report_weighted_batch_n","Report-weighted distinct batch users",1)):
        fig,ax=plt.subplots(figsize=(9,4.8))
        labels=[r["method"].replace("_","\n") for r in summary]
        values=[r[key]*scale for r in summary]
        ax.bar(labels,values)
        ax.set_ylabel(title)
        ax.set_title("New manuscript W/L/O + route PPO pilot (not formal TMAPO)")
        ax.tick_params(axis="x",labelsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT/f"{key}.png",dpi=160)
        plt.close(fig)
    lines=["# New manuscript W/L/O + route PPO: live training smoke",
           "",f"Trained {UPDATES} PPO updates × {ROLLOUT} slots; M=90, two held-out seeds.",
           "All scheme results are paired by seed. Conditional DP only; no multi-report proof.",
           "No Markov burst process or formally converged MAPPO.","",
           "| Method | QoS fail | Completion | QL (m) | Offload | Wait-aware cost |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        def pct(k):return f"{100*r[k]:.2f}%"
        lines.append(f"| {r['method']} | {pct('qos_rate')} | {pct('completion_rate')} | "
          f"{r['ql_m']:.1f} | {pct('publication_ratio')} | {r['mean_task_cost']:.4f} |")
    lines+=["","QL is measured AFTER same-public-region deterministic projection;",
       "original unprojected planar-Laplace QL is NOT interchangeable.",
       "Caution: lower published-report QL may be caused by fewer",
       "publications; compare publication share and completion simultaneously.",
       f"Wall time {time.monotonic()-t0:.1f}s; this is a bounded research smoke test."]
    report="\n".join(lines)+"\n"
    (OUTPUT/"REPORT.md").write_text(report,encoding="utf-8")
    print(report,flush=True)
    print("PASS actual new W/L/O + 11-way route actor pilot",flush=True)

if __name__=="__main__":
    main()

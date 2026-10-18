#!/usr/bin/env python3
"""Run cross-slot W/L/O release+route PPO and paired rule baselines.

An actual integrated environment smoke experiment, NOT 2000-update training.
Two actor networks are updated: shared W/L/O also gates cross-slot public
class pool (masked WAIT/RELEASE), route selects among 11 feasible nodes.
"""
import copy,csv,json,math,os,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model"))
sys.path.insert(0,str(ROOT/"experiments"))
from cross_slot_environment import CrossSlotWLOEnvironment
from new_method_joint_wlo_smoke import new_config as previous_config
from environment import Task,N_ROUTE
from learners import PPO

USERS_PER_REGION=int(os.getenv("CROSS_USERS_PER_REGION","30"))
LAMBDA=float(os.getenv("CROSS_ARRIVAL_PER_USER_S",".8"))
WAIT_MAX=float(os.getenv("CROSS_WAIT_MAX_S",".18"))
UPDATES=int(os.getenv("CROSS_UPDATES","48"))
ROLLOUT=int(os.getenv("CROSS_ROLLOUT","64"))
TEST_STEPS=int(os.getenv("CROSS_TEST_STEPS","400"))
VALIDATION_SEEDS=(814001,814019)
VALIDATION_STEPS=150
VALIDATE_EVERY=8
SEEDS=tuple(int(v) for v in os.getenv("CROSS_TEST_SEEDS","481953,729604,814113,814229").split(","))
EPSILON=float(os.getenv("CROSS_EPSILON","0.2"))
PROJECT_REGION=os.getenv("CROSS_PROJECT_REGION","1")=="1"
TRAIN_SEED=int(os.getenv("CROSS_TRAIN_SEED","87021"))
OUT=ROOT/"results"/f"cross_slot_WLO_M{USERS_PER_REGION*3}_lambda{LAMBDA:g}"
METHODS=("learned_crossslot","immediate_remote","fixed50",
         "fixed100","fixed180","forecast_release","local_preferred")


def configuration():
    cfg,train=previous_config()
    cfg.update(users_per_region=USERS_PER_REGION,
               arrival_per_user_s=LAMBDA,
               wait_max_s=WAIT_MAX,
               prediction_horizon_s=WAIT_MAX,
               project_public_region=PROJECT_REGION,
               epsilon_req=EPSILON,delta_req=1e-6,
               # Deadline and service reserve are never adjusted after results.
               min_service_reserve_s=.04)
    train.update(hidden=32,epochs=2,minibatch=256,actor_cost_prior_scale=.15,
                 actor_lr=3e-4,critic_lr=7e-4,entropy=.01,
                 linear_lr_decay=False)
    return cfg,train


class CrossSlotRule:
    is_rule_based=True
    def __init__(self,name,cfg):
        self.name,self.cfg=name,cfg
        self.env=None
    def begin(self,*args): pass
    def act(self,obs,mask,stage,agents):
        actions=[]
        for v,allow in zip(obs,mask):
            if stage=="route":
                cost=(self.cfg["weight_time"]*v[:N_ROUTE]+
                      self.cfg["weight_energy"]*v[N_ROUTE:2*N_ROUTE]+
                      self.cfg["weight_violation"]*
                      (v[:N_ROUTE]>v[45]))
                actions.append(int(np.argmin(np.where(allow,cost,np.inf))))
            elif stage=="user":
                if self.name=="strict_local_only":
                    # Strict local comparator. If local compute is full, the
                    # admission logic rejects rather than switching to remote.
                    actions.append(1)
                    continue
                if self.name=="local_preferred":
                    pref=(1,2,0) if v[17]<=v[18] else (2,1,0)
                else:
                    pref=(2,1,0)
                actions.append(next(a for a in pref if allow[a]))
            elif stage=="group":
                slot=self.env.slot
                # obs[59] is batch size; obs[60] is oldest waiting age.
                wait_age=v[60]*self.cfg["wait_max_s"]
                count=v[20]*self.cfg["users_per_region"]
                threshold=self.env.group_target or self.cfg["users_per_region"]+1
                predicted_add=v[24]*20
                if not allow[0] or self.name=="immediate_remote":
                    a=2
                elif self.name.startswith("fixed"):
                    period={"fixed50":5,"fixed100":10,"fixed180":18}[self.name]
                    a=2 if (slot+1)%period==0 else 0
                elif self.name=="forecast_release":
                    # Pre-register a causal, not hindsight, forecast heuristic.
                    poor_future=(count+predicted_add<.9*threshold)
                    a=2 if (wait_age>=.02 and poor_future) or count>=threshold else 0
                elif self.name=="local_preferred":
                    a=2
                else:raise ValueError(self.name)
                actions.append(a)
            else:raise AssertionError(stage)
        return np.asarray(actions,dtype=int),[]


class ScriptedInvariantRule:
    is_rule_based=True
    def begin(self,*args):pass
    def act(self,obs,mask,stage,agents):
        if stage=="route":return np.array([np.flatnonzero(m)[0] for m in mask]),[]
        if stage=="user":return np.array([2 for _ in agents]),[]
        if stage=="group":
            out=[]
            for v,m in zip(obs,mask):
                # Need two different users to release, or forced deadline.
                out.append(2 if v[20]>=2/3-1e-8 or not m[0] else 0)
            return np.array(out),[]
        raise ValueError(stage)


def tests(cfg):
    c=copy.deepcopy(cfg)
    c.update(users_per_region=3,arrival_per_user_s=.05)
    env=CrossSlotWLOEnvironment(c,seed=411,collect=True)
    policy=ScriptedInvariantRule()
    # Same public region and class in *different* time slots.
    def attach(uid,idx,profile="interactive"):
        j=Task(idx,uid,0,env.time,50_000.,80e6,.2,
               profile=profile,arrival_slot=env.slot)
        env.tasks[idx]=j;env.pending[uid].append(idx)
        env.arrivals+=1;env.measure_arrivals+=1
    attach(0,0)
    env.step(policy,arrivals=False)
    assert not env.batch_rows
    assert len(env.class_staging[(0,"interactive")])==1
    attach(1,1)
    env.step(policy,arrivals=False)
    groups=[r for r in env.batch_rows if r["task_class"]=="interactive"]
    assert len(groups)==1 and groups[0]["n_distinct"]==2,groups
    assert env.tasks[0].batch_n==2 and env.tasks[1].batch_n==2
    assert env.tasks[0].release==env.tasks[1].release
    # A different class is never merged into this batch.
    attach(2,2,profile="compute")
    for _ in range(26):
        if not env.tasks.get(2):break
        env.step(policy,arrivals=False)
    assert any(x["task_class"]=="compute" and x["n_distinct"]==1 for x in env.batch_rows)
    assert not any(env.class_staging.values())
    assert not env.staged_users
    print("TEST_PASS cross-slot pooling, task-class isolation, user uniqueness, hard release deadline",flush=True)
    # Regression: the forecast gate MUST derive age from obs[60], not
    # obs[59] which is normalized batch size. The two values can differ.
    probe=CrossSlotRule("forecast_release",c)
    probe.env=type("MockGateState",(),{"slot":3,"group_target":20})()
    feature=np.zeros((1,64),dtype=float)
    mask=np.array([[True,False,True]])
    feature[0,59]=0.99  # very LARGE (irrelevant) normalized batch size
    feature[0,60]=0.00  # zero actual waiting age: should not release yet
    assert int(probe.act(feature,mask,"group",np.array([0]))[0][0])==0
    feature[0,59]=0.01  # very SMALL (irrelevant) normalized batch size
    feature[0,60]=0.21  # 0.21*0.18=37.8ms: should release if forecast poor
    assert int(probe.act(feature,mask,"group",np.array([0]))[0][0])==2
    print("TEST_PASS forecast gate uses real oldest age, not batch-size index",flush=True)



def play(policy,c,seed,steps):
    if hasattr(policy,"reset_history"):
        policy.reset_history()
    env=CrossSlotWLOEnvironment(c,seed=seed,measure_start=0,
                                measure_end=steps*c["dt_s"],collect=True)
    if getattr(policy,"is_rule_based",False):policy.env=env
    stop=steps+math.ceil(c["lifetime_s"]/c["dt_s"])+5
    for slot in range(stop):
        env.step(policy,arrivals=slot<steps)
        if slot>=steps and not env.tasks:break
    else:raise RuntimeError("Task cohort did not drain")
    m=env.metrics()
    assert m["class_staging_left"]==0
    assert m["arrivals"]==m["completed"]+m["failed"],m
    return m,env


def main():
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    c,t=configuration()
    for k in ("weight_quality","quality_ref_m","weight_time","time_ref_s","weight_energy","energy_ref_j","weight_violation"):
        assert c[k] > 0, "Joint reward missing "+k
    torch.set_num_threads(1)
    tests(c)
    torch.manual_seed(TRAIN_SEED);np.random.seed(TRAIN_SEED)
    learner=PPO("TMAPO",t,c,device="cpu")
    env=CrossSlotWLOEnvironment(c,seed=TRAIN_SEED,phase_s=40.)
    training=[]
    best_validation_cost=float("inf")
    best_eval_state=None
    selected_update=None
    for update in range(1,UPDATES+1):
        learner.deterministic=False
        learner.set_progress(update, UPDATES)
        reward=[]
        for _ in range(ROLLOUT):
            r,rec,ctx,glob=env.step(learner)
            learner.remember(env.agent_rewards(.7),rec,False)
            reward.append(r)
        ctx,glob,*_=env.observations()
        logs=learner.update(ctx,glob)
        diagnostics=dict(update=update,simulated_s=env.time,
                         mean_reward=float(np.mean(reward)),
                         **{k:float(v) for k,v in logs.items()})
        if update%VALIDATE_EVERY==0 or update==UPDATES:
            learner.deterministic=True
            validation=[play(learner,c,vs,VALIDATION_STEPS)[0]
                        for vs in VALIDATION_SEEDS]
            learner.deterministic=False
            score=float(np.mean([v["mean_task_cost"] for v in validation]))
            diagnostics["validation_cost"]=score
            if score<best_validation_cost:
                best_validation_cost=score
                best_eval_state=copy.deepcopy(learner.state(for_evaluation=True))
                selected_update=update
            diagnostics["best_validation_cost"]=best_validation_cost
        training.append(diagnostics)
        if update==1 or update%5==0 or update==UPDATES:
            print("TRAIN",training[-1],flush=True)
    assert best_eval_state is not None
    learner.load(best_eval_state)
    learner.deterministic=True
    torch.save(learner.state(for_evaluation=True),OUT/"trained_ppo_checkpoint.pt")
    rows=[];batches=[]
    for seed in SEEDS:
        for method in METHODS:
            p=learner if method=="learned_crossslot" else CrossSlotRule(method,c)
            m,e=play(p,c,seed,TEST_STEPS)
            rows.append(dict(seed=seed,method=method,**m))
            batches.extend(dict(seed=seed,method=method,**row)
                           for row in e.batch_rows)
            print("RESULT",seed,method,"QoS",round(m["qos_rate"],4),
                  "QL",round(m["ql_m"] or 0,1),"n",round(m["report_weighted_batch_n"] or 0,2),
                  "Amp",round(m["amplified_publication_ratio"] or 0,5),
                  "published",round(m["publication_ratio"],3),
                  "cost",round(m["mean_task_cost"],4),flush=True)
    def csv_write(p,records):
        if not records:return
        keys=sorted({k for x in records for k in x})
        with (OUT/p).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    csv_write("train.csv",training)
    csv_write("test_by_seed.csv",rows)
    csv_write("batch_trace.csv",batches)
    summary=[]
    metric_names=["qos_rate","mean_task_cost","completion_rate","delay_s",
       "energy_j_per_arrival","ql_m","ql_theory_m","publication_ratio",
       "report_weighted_batch_n","amplified_publication_ratio",
       "class_gate_wait_actions","class_gate_release_actions","maximum_distinct_class_batch"]
    for method in METHODS:
        xs=[x for x in rows if x["method"]==method]
        rec={"method":method,"total_users":3*c["users_per_region"]}
        for k in metric_names:
            vals=[float(x[k]) for x in xs if x.get(k) is not None]
            rec[k]=float(np.mean(vals)) if vals else float("nan")
        summary.append(rec)
    csv_write("summary.csv",summary)
    # Paired comparisons across independently seeded, shared workload traces.
    # This does not turn the short training pilot into a publication claim.
    from scipy.stats import t as t_distribution
    for seed in SEEDS:
        hashes={r["trace_hash"] for r in rows if r["seed"]==seed}
        assert len(hashes)==1, ("Different arrival traces within paired seed",seed,hashes)
    paired=[]
    ref="forecast_release"
    for method in METHODS:
        if method==ref:continue
        for metric in ("mean_task_cost","qos_rate","delay_s",
                       "energy_j_per_arrival","ql_m","active_wait_s",
                       "publication_ratio","amplified_publication_ratio"):
            differences=[]
            for seed in SEEDS:
                a=next(row for row in rows if row["seed"]==seed and row["method"]==method)
                b=next(row for row in rows if row["seed"]==seed and row["method"]==ref)
                if a.get(metric) is None or b.get(metric) is None:continue
                differences.append(float(a[metric])-float(b[metric]))
            if len(differences)<2:continue
            arr=np.asarray(differences,dtype=float)
            ci=float(t_distribution.ppf(.975,len(arr)-1)*arr.std(ddof=1)/np.sqrt(len(arr)))
            paired.append(dict(method=method,baseline=ref,metric=metric,
                               independent_seeds=len(arr),diff_mean=float(arr.mean()),
                               ci95_halfwidth=ci))
    csv_write("paired_vs_forecast.csv",paired)
    report=[
        "# Cross-slot learned W/L/O + public-class adaptive release",
        "",
        f"Training: {UPDATES} PPO updates x {ROLLOUT} environment steps, "
        f"total M={c['users_per_region']*3}, per-user arrivals={LAMBDA}/s.",
        f"Epsilon {EPSILON:g}, delta 1e-6; public region projection={PROJECT_REGION}.",
        "Same held-out seeds for all policies; two model actors: release/user and route.",
        "",
        "| Method | QoS fail | Cost | QL (m) | Published | Mean K | Amp share |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for x in summary:
        report.append(f"| {x['method']} | {100*x['qos_rate']:.2f}% | "
          f"{x['mean_task_cost']:.3f} | {x['ql_m']:.1f} | "
          f"{100*x['publication_ratio']:.1f}% | "
          f"{x['report_weighted_batch_n']:.2f} | "
          f"{100*x['amplified_publication_ratio']:.3f}% |")
    report+=["","Exploratory training with joint WLO-route PPO; not fully converged submission evidence.",
       "The W/L/O/user actor weights are reused at a public class-group gate.",
       "Group release timing and task/public metadata may leak information;",
       "conditional one-batch accountant is not a longitudinal DP guarantee.",
       "Batch size = distinct public group users, not count of repeated tasks.",
       "Group actor is implemented by the SAME user release network,",
       "with LOCAL masked at the gate. Route actor uses DP report.",
       "All-local is not tested: local-preferred may offload if cheaper/feasible.",
       "Changes in publication ratio can change QL denominators.",
       f"Wall time {time.monotonic()-start:.1f}s."]
    (OUT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    manifest=dict(stage="bounded actual WLO + cross-slot class release train/eval",
        main_not_modified=True,methods=METHODS,
        release_actor_shared_between_user_and_group=True,
        routing_actor_nodes=11,
        c=c,t=t,training_seed=TRAIN_SEED,validation_seeds=VALIDATION_SEEDS,
        selected_checkpoint_update=selected_update,
        best_validation_cost=best_validation_cost,test_seeds=SEEDS,
        updates=UPDATES,rollout=ROLLOUT,test_steps=TEST_STEPS,
        region_class_release_requires_distinct_users=True,
        delta=1e-6,epsilon=EPSILON,
        privacy_full_transcript_dp_proof=False,
        notes=["no Markov burst process","short trajectories",
               "same-class public metadata and release timestamps assumed",
               "one training initialization and only two test seeds",
               "cross-slot waiting capped by max wait and task deadline"],
        wall_s=round(time.monotonic()-start,2))
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs=[
        ("qos_rate","QoS violation (%)",100),
        ("mean_task_cost","Mean multiobjective task cost",1),
        ("delay_s","Completed-task delay (ms)",1000),
        ("energy_j_per_arrival","Energy per generated task (J)",1),
        ("ql_m","Published QL after region projection (m)",1),
        ("publication_ratio","Published fraction (%)",100),
        ("report_weighted_batch_n","Report-weighted unique users per batch",1),
        ("amplified_publication_ratio","Amplified published reports (%)",100)]
    for key,title,scale in specs:
        fig,ax=plt.subplots(figsize=(8.5,4.7))
        xs=[r["method"].replace("_","\n") for r in summary]
        ys=[r[key]*scale for r in summary]
        ax.bar(xs,ys)
        ax.set(ylabel=title,title=f"Cross-slot region/class WLO, M={3*USERS_PER_REGION}, eps={EPSILON:g}")
        ax.tick_params(axis="x",labelsize=7.5)
        fig.tight_layout()
        fig.savefig(OUT/f"{key}.png",dpi=150)
        plt.close(fig)
    print("\n".join(report),flush=True)
    print("PASS cross-slot environment and trained WLO+route PPO",flush=True)


if __name__=="__main__":
    main()

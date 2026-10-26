"""Paired external workloads, independent validation and held-out test traces."""
from __future__ import annotations
import copy
import math
import numpy as np
from environment import Environment, GreedyPolicy
from privacy import local_epsilon, planar_noise, calibrate, delta_bound
from common import write_csv, save_json


def evaluation_phase(config, seed, warm, measure):
    c, t = config["environment"], config["training"]
    if t.get("evaluation_phase_schedule") == "stratified":
        for bank in (t["validation_seeds"], t["test_seeds"]):
            if seed in bank:
                center = (bank.index(seed)+.5)*c["tide_period_s"]/len(bank)
                return float((center-warm-measure/2) % c["tide_period_s"])
    return float(np.random.default_rng(np.random.SeedSequence([seed,9])).uniform(0,c["tide_period_s"]))


def evaluate(policy,config,method,seeds,formal=False,collect=False):
    t=config["training"]; c=config["environment"]
    warm=t["formal_warmup_s" if formal else "pilot_warmup_s"]
    measure=t["formal_measure_s" if formal else "pilot_measure_s"]
    rows,tasks,batches=[],[],[]
    for seed in seeds:
        phase=evaluation_phase(config,seed,warm,measure)
        env=Environment(c,seed,method,phase,measure_start=warm,measure_end=warm+measure,collect=collect)
        if hasattr(policy,"reset_history"): policy.reset_history()
        policy.deterministic=True
        slots=round((warm+measure)/c["dt_s"])
        max_slots=slots+math.ceil(c["lifetime_s"]/c["dt_s"])+2
        for _ in range(max_slots):
            env.step(policy,arrivals=env.slot<slots)
            if env.slot>=slots and not env.tasks:
                break
        else:
            raise RuntimeError("Evaluation failed to drain its arrival cohort")
        rows.append(dict(trace_seed=seed,phase_s=phase,window_s=measure,**env.metrics()))
        tasks.extend(dict(trace_seed=seed,**r) for r in env.rows)
        batches.extend(dict(trace_seed=seed,**r) for r in env.batch_rows)
    return rows,tasks,batches


def quality_check(config,out,samples=None):
    c=config["environment"]; e=config["experiments"]
    rng=np.random.default_rng(501)
    count=samples or e["ql_samples"]
    rows=[]
    for req in e["epsilons"]:
        for n in e["ql_batch_counts"]:
            cc=dict(c,epsilon_req=req)
            eps=local_epsilon(cc,n)
            noise=planar_noise(rng,eps,count)
            distances=np.linalg.norm(noise,axis=1)
            expected=2/eps
            stderr=math.sqrt(2/count)/eps
            eps0=eps*2*c["region_radius_m"]
            delta=delta_bound(eps0,n,req)
            z=(float(distances.mean())-expected)/stderr
            rows.append(dict(epsilon_req=req,n_distinct=n,epsilon0=eps0,epsilon_loc_per_m=eps,
                             delta_bound=delta,ql_sample_m=float(distances.mean()),ql_theory_m=expected,
                             mean_z_score=z,passed=abs(z)<6 and delta<=c["delta_req"]+1e-12))
    write_csv(out/"ql_check.csv",rows)
    passed=all(r["passed"] for r in rows)
    save_json(out/"check.json",{"passed":passed,"samples_each":count,"interpretation":"finite-batch conditional bound, untruncated planar Laplace mean"})
    if not passed:
        raise RuntimeError("QL/accountant check failed; inspect ql_check.csv before training")
    return rows


class BatchProbePolicy(GreedyPolicy):
    """Public, fixed release clock to obtain actual batches without training."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.slot=-1

    def begin(self,context,glob,n_user):
        super().begin(context,glob,n_user)
        self.slot+=1

    def act(self,obs,mask,stage,agents):
        if stage=="batch":
            desired_release = self.slot%5==4
            return np.where(mask[:,1] & (desired_release | ~mask[:,0]),1,0),[]
        return super().act(obs,mask,stage,agents)


def value_table(config,out,states,repeats,seed=2026):
    """Frozen-backlog counterfactuals, not a replacement for full-system evaluation.

    Simulate each feasible candidate once per state. Reusing that deterministic
    physical outcome across noise replicates is exact with fixed background and
    deterministic channels. Only the target report and selected node vary.
    """
    c=config["environment"]
    rows=[]; captured=0
    phases=[i*c["tide_period_s"]/4 for i in range(4)]
    quotas=[states//4+int(i<states%4) for i in range(4)]
    for phase_index,(phase,quota) in enumerate(zip(phases,quotas)):
        if not quota: continue
        policy=BatchProbePolicy(c)
        env=Environment(c,seed+phase_index*1000,"TMAPO",phase_s=phase,collect=False)
        phase_captured=0
        max_slots=max(2000,quota*20)
        for _ in range(max_slots):
            context,glob,mask,users,ready=env.observations()
            if env.time>=max(.5,config["training"]["pilot_warmup_s"]) and env.slot%5==0:
                # Limit correlated captures at a single instant.
                indexes=sorted(set([0,len(ready)//2])) if ready else []
                for index in indexes:
                    if phase_captured>=quota: break
                    target=ready[index]
                    _,feasible,_=env.route_features(target)
                    if not feasible.any(): continue
                    # Fix other requests' decisions across every location alternative.
                    choices=[]
                    for other in ready:
                        if other.id!=target.id:
                            ob,ma,_=env.route_features(other)
                            a,_=policy.act(ob[None],ma[None],"route",np.array([env.u+other.region]))
                            choices.append((other.id,int(a[0])))
                    outcomes={}
                    for node in np.flatnonzero(feasible):
                        trial=copy.deepcopy(env); trial.collect=True; trial.rows=[]
                        for jid,a in choices: trial.assign(trial.tasks[jid],a)
                        trial.assign(trial.tasks[target.id],int(node))
                        for _ in range(math.ceil(c["lifetime_s"]/c["dt_s"])+2):
                            trial.advance(trial.time+c["dt_s"])
                            if target.id not in trial.tasks: break
                        result=next(r for r in trial.rows if r["task_id"]==target.id)
                        outcomes[int(node)]=result
                    rng=np.random.default_rng(np.random.SeedSequence([seed,10,phase_index,captured]))
                    unit=planar_noise(rng,1.,repeats)
                    truth=env.xy[target.user]
                    report_sets={"public_center":np.tile(c["region_centers_m"][target.region],(repeats,1)),
                                 "no_amplification":truth+unit/local_epsilon(c,target.batch_n,True),
                                 "amplification":truth+unit/local_epsilon(c,target.batch_n),
                                 "true_position_reference":np.tile(truth,(repeats,1))}
                    for label,reports in report_sets.items():
                        for rep,report in enumerate(reports):
                            _,ma,cost=env.route_features(target,report)
                            node=int(np.argmin(np.where(ma,cost,np.inf))); r=outcomes[node]
                            rows.append({"state":captured,"phase_index":phase_index,"region":target.region,
                                         "task_id":target.id,"time_s":env.time,"batch_n":target.batch_n,
                                         "information":label,"replicate":rep,"chosen_node":node,
                                         "ql_m":float(np.linalg.norm(report-truth)),"completed":r["completed"],
                                         "delay_s":r["end_s"]-r["arrival_s"] if r["completed"] else None,
                                         "energy_j":r["energy_j"],"qos":r["qos"]})
                    captured+=1
                    phase_captured+=1
                    if captured%100==0: print(f"value states {captured}/{states}",flush=True)
            if phase_captured>=quota: break
            env.step(policy)
        if phase_captured!=quota:
            raise RuntimeError(f"Phase {phase_index}: only {phase_captured}/{quota} published states")
    if captured!=states:
        raise RuntimeError(f"Only {captured}/{states} published states; increase capture horizon or examine policy")
    write_csv(out/"value_raw.csv",rows)
    summary=[]
    for label in report_sets:
        subset=[r for r in rows if r["information"]==label]
        completed=[r["delay_s"] for r in subset if r["completed"]]
        summary.append({"information":label,"states":states,"repeats":repeats,
                        "completed_delay_s":float(np.mean(completed)) if completed else None,
                        "energy_j_per_task":float(np.mean([r["energy_j"] for r in subset])),
                        "qos_rate":float(np.mean([r["qos"] for r in subset])),
                        "ql_m":float(np.mean([r["ql_m"] for r in subset]))})
    write_csv(out/"value_table.csv",summary)
    batches=np.array([r["batch_n"] for r in rows if r["replicate"]==0 and r["information"]=="public_center"])
    amplified=np.array([local_epsilon(c,int(n))>local_epsilon(c,int(n),True) for n in batches])
    save_json(out/"scope.json",{"source":"regional release probe with five-slot preference and mandatory threshold/deadline gate; no trained policy; fixed backlog counterfactual",
              "future_arrivals":False,"paired_noise":True,"not_a_full_system_result":True,
              "states":states,"repeats":repeats,"states_not_independent_training_seeds":True,
              "capture_start_s":min(r["time_s"] for r in rows),"capture_end_s":max(r["time_s"] for r in rows),
              "phase_offsets_s":phases,"batch_n_mean":float(batches.mean()),"batch_n_min":int(batches.min()),
              "batch_n_max":int(batches.max()),"amplified_state_fraction":float(amplified.mean()),
              "snapshot_sampling":"four tide phases, up to two states at a sampled slot"})
    return summary

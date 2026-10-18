#!/usr/bin/env python3
"""Frozen four-experiment protocol: plan, preflight and a bounded convergence training cell.

Run in repository root: python experiments/paper_four_studies.py preflight
Research exploration; old legacy shuffle accountant is NOT a formal Geo-I DP proof.
"""
from __future__ import annotations
import argparse,copy,csv,json,os,sys,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model"))
sys.path.insert(0,str(ROOT/"experiments"))
from cross_slot_wlo_train import configuration, CrossSlotRule, play, tests
from cross_slot_environment import CrossSlotWLOEnvironment
from cross_slot_learners import CrossSlotPPO,CrossSlotTD3

REQUIRED_MAIN=("TMAPO","IPPO","Greedy","Local-Only","NoTidal",
               "ATO-SLA-Adapted","MARATD3-Adapted")
CONVERGENCE_METHODS=("TMAPO","IPPO","MARATD3-Adapted")
LEARNING_RATES=((1e-4,3e-4),(3e-4,1e-3))
MAIN_USERS=(30,45,60,75,90)   # GLOBAL users, divisible by 3
PRIVACY_USERS=(30,60,90,150,210,300)
PRIVACY_EPS=(.1,.2,.5,1.,2.)
TRAIN_SEEDS=(87021,87035,87049)
VALIDATION_SEEDS=(814001,814019)
EXPLORATORY_TEST_SEEDS=(481953,729604,814113,814229)
# Earlier smoke tests already exposed the old streams. Reserve fresh blind
# streams before any final evaluation; do not relabel explored seeds as unseen.
TEST_SEEDS=(9848031,9848047,9848063,9848089)
PRIVACY_POLICIES=("Geo-I","Fixed-Shuffle","TMPPO-Shuffle")
WEIGHT_PAIRS=((.1,.5),(.2,.4),(.3,.3),(.4,.2),(.5,.1))
SENSITIVITY_PAIRS=(("weight_time","weight_energy"),
                   ("weight_time","weight_quality"),
                   ("weight_energy","weight_quality"))
RESULTS=ROOT/"results"/"paper_four_studies"

def full_config(users=60,epsilon=.2,hide_tide=False):
    if users%3:raise ValueError("Total users must divide evenly among three regions")
    cfg,train=configuration()
    cfg.update(users_per_region=users//3,arrival_per_user_s=3.0,
               wait_max_s=.18,prediction_horizon_s=.18,
               epsilon_req=float(epsilon),delta_req=1e-6,
               project_public_region=False,hide_tidal_observations=bool(hide_tide),
               weight_time=.25,weight_energy=.25,weight_quality=.25,
               weight_violation=1.0,quality_ref_m=10000.,
               min_service_reserve_s=.04)
    train.update(hidden=32,epochs=2,minibatch=256,actor_cost_prior_scale=2.,
                 gamma=1.,gae_lambda=.98,entropy=.01,
                 linear_lr_decay=False,torch_threads=1)
    return cfg,train

def paper_plan():
    convergence=[dict(method=m,lr_index=i+1,actor_lr=a,critic_lr=c,train_seed=sd,
                      users=60,epsilon=.2)
                 for m in CONVERGENCE_METHODS
                 for i,(a,c) in enumerate(LEARNING_RATES)
                 for sd in TRAIN_SEEDS]
    scaling=[dict(users=n,method=m) for n in MAIN_USERS for m in REQUIRED_MAIN]
    privacy=[dict(users=n,epsilon=e,policy=p)
             for n in PRIVACY_USERS for e in PRIVACY_EPS for p in PRIVACY_POLICIES]
    sensitivity=[]
    for first,second in SENSITIVITY_PAIRS:
        for x,y in WEIGHT_PAIRS:
            w={"weight_time":.2,"weight_energy":.2,
               "weight_quality":.2,"weight_violation":1.0}
            w[first]=x;w[second]=y
            assert abs(sum(w.values())-1.8)<1e-12
            sensitivity.append(dict(pair=[first,second],weights=w))
    return dict(convergence=convergence,scaling=scaling,privacy=privacy,
                sensitivity=sensitivity,baseline_methods=REQUIRED_MAIN,
                seed_protocol=dict(training=TRAIN_SEEDS,validation=VALIDATION_SEEDS,
                                   heldout_test=TEST_SEEDS,retired_exploratory=EXPLORATORY_TEST_SEEDS),
                denominator_policy={"delay":"completed tasks only",
                                    "energy":"generated tasks",
                                    "QoS":"generated tasks",
                                    "QL":"published reports only, report publication ratio alongside",
                                    "location_error":"raw unprojected planar-Laplace report"},
                DP_status="legacy accounting exploratory ONLY; not validated as full-trajectory Geo-I DP",
                weighting="QoS penalty fixed at 1.0; baseline raw QL reference 10000m for epsilon 0.2",
                execution_status={"convergence":"start with bounded preflight",
                                  "scaling":"blocked until frozen validated convergence checkpoint",
                                  "privacy":"paired released report sets and no-projection test required",
                                  "sensitivity":"separate training per weight pair (15 cells)"})

def csv_write(path,records):
    if not records:return
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in records for k in row))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)

def preflight():
    c,t=full_config()
    assert not c["project_public_region"] and not c["hide_tidal_observations"]
    assert abs(sum(c[k] for k in ("weight_time","weight_energy","weight_quality"))-.75)<1e-12
    assert c["weight_violation"]==1.0 and c["quality_ref_m"]==10000.
    tests(c)
    # A strict Local-Only policy must NEVER publish a location.
    a,env=play(CrossSlotRule("strict_local_only",c),c,711043,40)
    assert a["published"]==0 and a["publication_ratio"]==0 and a["ql_m"] is None,a
    # NoTidal uses identical physical traffic and masks explicit tidal observables.
    d,_=full_config(hide_tide=True)
    standard=CrossSlotWLOEnvironment(c,seed=814017)
    nt=CrossSlotWLOEnvironment(d,seed=814017)
    standard.generate();nt.generate()
    x,g,*_=standard.observations()
    y,h,*_=nt.observations()
    assert standard.arrival_hash==nt.arrival_hash
    assert np.all(y[:nt.u,3:6]==0)
    np.testing.assert_array_equal(x[:nt.u,20],y[:nt.u,20])
    assert np.all(h[25:34]==0) and np.all(h[43:46]==0)
    assert not d["project_public_region"] and d["hide_tidal_observations"]
    print("PREFLIGHT_PASS class pool, strict local, no-project, NoTidal input masking, matched physical arrivals",flush=True)

def make_learner(method,t,c):
    if method in ("TMAPO","IPPO","NoTidal"):
        return CrossSlotPPO("IPPO" if method=="IPPO" else "TMAPO",t,c,device="cpu")
    if method=="MARATD3-Adapted":
        return CrossSlotTD3("MARATD3-Adapted",t,c,device="cpu")
    raise ValueError("Training unsupported: "+method)

def bounded_train(method="TMAPO",lr=1,seed=87021,updates=4,rollout=32,test_steps=75,
                  eval_seeds=(481953,729604)):
    # Retain the old small command, but do not reuse its unsafe fixed-directory
    # loop or consume the paper's held-out test set during development.
    from datetime import datetime
    from paper_convergence import train_cell
    if method=="NoTidal":
        raise ValueError("NoTidal belongs to study 2, not the convergence pilot")
    if not 0 < updates <= 128:
        raise ValueError("This compatibility entry is a bounded pilot only")
    out=RESULTS/("pilot_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    settings=dict(method="TMPPO" if method=="TMAPO" else method,lr=lr,seed=seed,
                  updates=updates,rollout=rollout,episode_steps=1024,phase_count=3,
                  validate_every=max(1,updates//2),validation_steps=test_steps,
                  warmup_steps=20,validation_seeds=list(VALIDATION_SEEDS),
                  validation_phases=[30.,130.,230.])
    train_cell(out,settings,wall_seconds=600.)
    print("TRAIN_CELL_PASS",str(out),"validation only; held-out tests untouched",flush=True)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("plan","preflight","pilot"))
    p.add_argument("--method",choices=CONVERGENCE_METHODS+("NoTidal",),default="TMAPO")
    p.add_argument("--lr",type=int,choices=(1,2),default=1)
    p.add_argument("--seed",type=int,default=87021)
    p.add_argument("--updates",type=int,default=4)
    p.add_argument("--rollout",type=int,default=32)
    p.add_argument("--test-steps",type=int,default=75)
    a=p.parse_args()
    if a.command=="plan":
        print(json.dumps(paper_plan(),ensure_ascii=False,indent=2))
    else:
        preflight()
        if a.command=="pilot":
            bounded_train(a.method,a.lr,a.seed,a.updates,a.rollout,a.test_steps)

if __name__=="__main__":
    main()

"""Tiny CPU integration checks, deliberately not a scientific experiment."""
from datetime import datetime
import copy
from common import ROOT,read_json,save_json
from run import plan,train_job,new_directory
from plots import plot_run
from evaluation import quality_check,value_table


def main():
    config=read_json(ROOT/"config.json")
    config["environment"]["users_per_region"]=2
    config["environment"]["arrival_per_user_s"]=40.0
    config["training"].update(updates=2,rollout_steps=8,episode_steps=12,hidden=16,epochs=1,
                              validation_interval=1,pilot_warmup_s=.01,pilot_measure_s=.1,
                              validation_seeds=[61001],td3_start_steps=2,td3_batch=2,td3_replay_slots=16)
    out=new_directory("SMOKE_NOT_EXPERIMENT")
    save_json(out/"config.json",config)
    quality_check(config,out,3000)
    value_table(config,out,states=3,repeats=2)
    for kind in ("privacy","convergence","main"):
        folder=out/kind; folder.mkdir()
        jobs=plan(config,kind,2026)
        if kind=="privacy":
            jobs=[j for j in jobs if j["epsilon_req"]==1]
            for j in jobs:
                j["users_per_region"]=2; j["id"]=j["id"].replace("_M100_","_M2_")
        if kind=="main":
            jobs=[j for j in jobs if j["users_per_region"]==20 and j["method"] in ("Greedy","Local-Only","TMAPO-NoPred","TMAPO-NoWait")]
            for j in jobs:
                j["users_per_region"]=2; j["id"]=j["id"].replace("_M20_","_M2_")
        save_json(folder/"plan.json",{"kind":kind,"jobs":jobs,"formal_eval":False})
        for j in jobs:
            if not train_job(config,j,folder/j["id"],"cpu"): raise RuntimeError("Unexpected pause")
        plot_run(folder)
    # Same-source paused continuation must match an uninterrupted CPU run.
    import torch
    for method in ("TMAPO","MARATD3-Adapted"):
        job=plan(config,"convergence",2026)[0]
        job["method"]=method; job["id"]=job["id"].replace("TMAPO",method)
        for name,pause in (("continuous",False),("resumed",True)):
            path=out/(method+"_"+name)
            train_job(config,job,path,"cpu",wall_seconds=.0001 if pause else None)
            if pause: train_job(config,job,path,"cpu")
        a=torch.load(out/(method+"_continuous")/"latest.pt",weights_only=False)
        b=torch.load(out/(method+"_resumed")/"latest.pt",weights_only=False)
        assert a["train_rows"]==b["train_rows"]
        assert a["environment"].arrival_hash==b["environment"].arrival_hash
        key="actors" if method=="TMAPO" else "actor"
        for k,v in a["policy"][key].items(): assert torch.equal(v,b["policy"][key][k])
    save_json(out/"smoke_passed.json",{"scope":"CPU wiring only, no convergence claim","resume_exact":True,
                                     "privacy_jobs":3,"convergence_jobs":4,"main_jobs":4})
    print(out)


if __name__=="__main__": main()

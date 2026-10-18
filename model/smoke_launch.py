"""Small end-to-end launcher check; never confused with the 2000-update run."""
from datetime import datetime
from common import ROOT, read_json, save_json
from launch_comparison import launch


if __name__=="__main__":
    cfg=read_json(ROOT/"config.json")
    cfg["environment"]["users_per_region"]=3
    cfg["training"].update(updates=2,rollout_steps=8,episode_steps=16,validation_interval=1,
                           hidden=16,epochs=1,minibatch=32,torch_threads=1,pilot_warmup_s=0.,pilot_measure_s=1.)
    config=ROOT/("smoke_config_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".json")
    save_json(config,cfg)
    code=launch(config,None,False,2,"cpu")
    if code: raise SystemExit(code)

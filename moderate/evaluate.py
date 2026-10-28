"""Only source-verified terminal policies, new full-cycle held-out trajectories."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from .protocol import configuration,POPULATIONS,TEST_SEEDS,digest
from .policies import evaluation_policy
from .runner import infer
from .environment import Environment,GeoIEnvironment
from .records import flush
from experiments.lr_screen.runtime import torch,np,seed_all,json_write,csv_write
from environment import GreedyPolicy


def evaluate(out,method,seed,weights=None,maximum_seconds=10800):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    model=None
    if weights:
        saved=torch.load(weights,map_location='cpu',weights_only=False)
        assert saved['method']==method and saved['seed']==seed and not saved['component_only']
        assert saved['training']['updates']==1208
        model=saved['policy']
    rows=[];started=time.monotonic()
    for n in POPULATIONS:
        cfg,t=configuration(method,seed,n)
        p=GreedyPolicy(cfg) if method in ('Greedy','GeoI-Greedy') else evaluation_policy(method,t,cfg,model)
        for trace in TEST_SEEDS:
            seed_all(int(np.random.SeedSequence([trace,seed,2803]).generate_state(1)[0]))
            folder=out/'cases'/f'N{n}_t{trace}'
            if method in ('Greedy','GeoI-Greedy'):
                cls=GeoIEnvironment if method=='GeoI-Greedy' else Environment
                e=cls(cfg,seed=trace,phase_s=0.,family=method,measure_start=5.,measure_end=305.,collect=True)
                e.arrival_stop_slot=30500;chunk=0
                for slot in range(30703):
                    e.step(p,arrivals=slot<30500)
                    if slot%100==99:
                        flush(e,folder,chunk);chunk+=1
                    if e.slot>=30500 and not e.tasks:
                        break
                else:
                    raise AssertionError('Rule drainage failed')
                flush(e,folder,chunk)
                m=dict(e.metrics(),full_input_hash=e.full_input_hash,users=e.u,trace_seed=trace,
                    phase_s=0.,warmup_s=5.,measure_s=300.,all_tasks_resolved=True)
            else:
                m=infer(p,cfg,method,trace,0.,5.,300.,folder)
            rows.append(dict(method=method,training_seed=seed,**m))
            json_write(folder/'metrics.json',rows[-1]);csv_write(out/'test_metrics.csv',rows)
            if time.monotonic()-started>maximum_seconds:
                json_write(out/'incomplete.json',dict(reason='evaluation wall budget',scenes=len(rows)))
                return False
            print('Full-cycle held-out scene saved',flush=True)
    json_write(out/'complete.json',dict(completed=True,method=method,seed=seed,scenes=len(rows),training=False,
        model_identity=saved['identity_hash'] if weights else None))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--weights',type=Path);a=p.parse_args()
    raise SystemExit(0 if evaluate(a.out,a.method,a.seed,a.weights) else 2)

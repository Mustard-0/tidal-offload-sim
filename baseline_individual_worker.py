"""Evaluate one frozen baseline arm with the same simulator and paired inputs."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import baseline_individual_protocol as spec
from baseline_individual_rules import with_adaptive_local,with_public_route_rule
from energy_qos_worker import parent_store,weights_for
from energy_qos_telemetry import with_task_telemetry
from evaluation_adapters import GREEDY_PAIR,evaluation_environment_types
from focus_fallback import heavy_fallback
from local_sched_worker import scheduler
import optimization2_worker as store
from review_checks import bootstrap,check_metrics
from scalability_protocol import settings


class Transport(store.Transport):
    def __init__(self):
        assert os.environ['GITHUB_REPOSITORY']==store.REPOSITORY
        assert os.environ['GITHUB_REF']=='refs/heads/'+spec.BRANCH and os.environ['GITHUB_RUN_ATTEMPT']=='1'
        self.api=store.GitHub(os.environ['GH_TOKEN'])
        self.key=bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
        self.commit,self.run_id=os.environ['GITHUB_SHA'],os.environ['GITHUB_RUN_ID']
        self.release=self.api.request('GET',store.BASE+'/releases/tags/'+spec.CAMPAIGN)
        packet=spec.read_packet(self.release,self.commit)
        self.inputs_sha256=store.sha(packet)
        raw,_=store.unseal(packet,self.key,dict(campaign=spec.CAMPAIGN,source_commit=self.commit,purpose='review-inputs'))
        self.inputs=json.loads(raw)
        assert self.inputs['source_base']==spec.BASE and self.inputs['plan']==json.loads(json.dumps(spec.plan()))
        for p,h in {**self.inputs['inherited_sha256'],**self.inputs['plan']['inherited_review_hashes'],
                    **self.inputs['new_sha256']}.items():
            assert store.sha(Path(p).read_bytes().replace(b'\r\n',b'\n'))==h,p
        self.plan_hash=spec.digest(self.inputs['plan'])
        self.assets={a['name']:a for a in self.api.request('GET',
            store.BASE+f"/releases/{self.release['id']}/assets?per_page=100")}


def execute(t,method,arm,result):
    bootstrap()
    from experiments.lr_screen.runtime import torch,seed_all
    from learners import cost_prior
    from pilot_worker import module_hash
    from pressure9.policies import evaluation_policy

    torch.set_num_threads(1)
    plain,tt,greedy,local=evaluation_environment_types(optimize_greedy=method in GREEDY_PAIR and arm=='targeted')
    if method=='Local-Only':
        env_type=scheduler(plain,'edf') if arm=='control' else with_adaptive_local(plain)
    else:
        env_type=heavy_fallback(tt if method=='TMAPO' else plain)
        if method not in spec.RULES and arm=='targeted':
            env_type=with_public_route_rule(env_type,method)
    env_type=with_task_telemetry(env_type)
    seed=None if method in spec.RULES else spec.SEED
    cell=method if seed is None else f'{method}_seed{seed}'
    parent=parent_store(t) if seed is not None and method!='TMAPO' else None
    weights,expected=weights_for(t,parent,cell,method,seed)
    deadline=time.monotonic()+spec.WALL
    for n,trace,phase in spec.CASES:
        if time.monotonic()>=deadline:
            raise TimeoutError('No time for next complete scene')
        seed_all(trace)
        cfg,train=settings(method,seed,n)
        options=dict(seed=trace,phase_s=phase,measure_start=spec.WARM,measure_end=spec.WARM+spec.MEASURE)
        env=env_type(cfg,**options) if method=='TMAPO' else env_type(cfg,family=method,**options)
        if seed is None:
            policy=local() if method=='Local-Only' else greedy(cfg)
            model=None
        else:
            policy=evaluation_policy(method,train,cfg,weights)
            model=module_hash(policy)
            expected=model if expected is None else expected
            assert model==expected
            if method=='TMAPO':
                original=policy.policy_logits
                def logits(stage,obs):
                    value=original(stage,obs)
                    return value+cost_prior('route',obs,cfg,1.) if stage=='route' else value
                policy.policy_logits=logits
        stop=round((spec.WARM+spec.MEASURE)/cfg['dt_s'])
        with torch.no_grad():
            for i in range(stop+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
                env.step(policy,arrivals=i<stop)
                if env.slot>=stop and not env.tasks:
                    break
                if i%100==99 and time.monotonic()>=deadline:
                    raise TimeoutError(f'Wall budget during N={n}')
            else:
                raise AssertionError('Natural drainage did not finish')
        assert seed is None or module_hash(policy)==model
        m=check_metrics(env)
        for k in ('mean_queue_pressure','high_queue_region_time_fraction','remote_compute_utilization',
                  'region_0_high_queue_fraction','region_1_high_queue_fraction','region_2_high_queue_fraction'):
            m.pop(k,None)
        result['rows'].append(dict(method=method,arm=arm,training_seed=seed,users=n,trace_seed=trace,
            phase_s=phase,measure_s=spec.MEASURE,model_hash=model,full_input_hash=env.full_input_hash,
            route_edits=getattr(env,'measured_route_edits',0),local_edits=getattr(env,'measured_queue_edits',0),
            local_masks=getattr(env,'measured_local_mask',0),energy_by_task_group=env.accounting(m),**m))
        print(f'Completed {method} {arm} N={n} trace={trace} phase={phase}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('method')
    p.add_argument('arm')
    a=p.parse_args()
    assert (a.method,a.arm) in spec.ARMS
    store.CAMPAIGN=spec.CAMPAIGN
    t=Transport()
    value=dict(method=a.method,arm=a.arm,completed=False,rows=[])
    try:
        assert store.within_dispatch_window(t)
        execute(t,a.method,a.arm,value)
        assert len(value['rows'])==len(spec.CASES)
        value['completed']=True
    except Exception as e:
        value['error']=dict(type=type(e).__name__,message=str(e))
    name=a.method+'.'+a.arm
    t.small(name+'.enc',value,name)
    return value['completed']


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

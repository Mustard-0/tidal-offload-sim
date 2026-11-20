"""Run one predeclared single-seed configuration on a free CPU worker."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import optimization2_worker as store
from energy_qos_telemetry import with_task_telemetry
from evaluation_adapters import evaluation_environment_types
from focus_fallback import heavy_fallback
import focus_protocol as spec
from review_checks import bootstrap,check_metrics
from scalability_protocol import settings


class FocusTransport(store.Transport):
    def __init__(self):
        assert os.environ['GITHUB_REPOSITORY']==store.REPOSITORY
        assert os.environ['GITHUB_REF']=='refs/heads/'+spec.BRANCH
        assert os.environ['GITHUB_RUN_ATTEMPT']=='1'
        self.api=store.GitHub(os.environ['GH_TOKEN'])
        self.key=bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
        self.commit,self.run_id=os.environ['GITHUB_SHA'],os.environ['GITHUB_RUN_ID']
        self.release=self.api.request('GET',store.BASE+'/releases/tags/'+spec.CAMPAIGN)
        packet=spec.decode(self.release,self.commit)
        self.inputs_sha256=store.sha(packet)
        raw,_=store.unseal(packet,self.key,dict(campaign=spec.CAMPAIGN,source_commit=self.commit,purpose='review-inputs'))
        self.inputs=json.loads(raw)
        assert self.inputs['plan']==json.loads(json.dumps(spec.plan())) and self.inputs['source_base']==spec.BASE
        for p,h in {**self.inputs['inherited_sha256'],**self.inputs['plan']['inherited_review_hashes'],
                    **self.inputs['new_sha256']}.items():
            assert store.sha(Path(p).read_bytes().replace(b'\r\n',b'\n'))==h,p
        self.plan_hash=spec.digest(self.inputs['plan'])
        self.assets={a['name']:a for a in self.api.request('GET',
            store.BASE+f"/releases/{self.release['id']}/assets?per_page=100")}


def run(t,variant,value):
    bootstrap()
    from experiments.lr_screen.runtime import torch,seed_all
    from learners import cost_prior
    from pilot_worker import module_hash
    from pressure9.policies import evaluation_policy
    from scalability_worker import prior_policy

    torch.set_num_threads(1)
    _,tt,_,_=evaluation_environment_types()
    env_type=with_task_telemetry(heavy_fallback(tt) if variant.endswith('_local') else tt)
    policy_state,_=prior_policy(t,spec.SEED)
    deadline=time.monotonic()+spec.WALL
    expected=None
    for track,n,trace,phase,measure in spec.CASES:
        if time.monotonic()>=deadline:
            raise TimeoutError('No time for next scene')
        seed_all(trace)
        cfg,train=settings('TMAPO',spec.SEED,n)
        policy=evaluation_policy('TMAPO',train,cfg,policy_state)
        before=module_hash(policy)
        expected=before if expected is None else expected
        assert before==expected
        if variant!='control':
            original=policy.policy_logits

            def logits(stage,obs):
                result=original(stage,obs)
                return result+cost_prior('route',obs,cfg,1.) if stage=='route' else result

            policy.policy_logits=logits
        env=env_type(cfg,seed=trace,phase_s=phase,measure_start=spec.WARM,measure_end=spec.WARM+measure)
        slots=round((spec.WARM+measure)/cfg['dt_s'])
        with torch.no_grad():
            for i in range(slots+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
                env.step(policy,arrivals=i<slots)
                if env.slot>=slots and not env.tasks:
                    break
                if i%100==99 and time.monotonic()>=deadline:
                    raise TimeoutError(f'Wall budget within N={n}')
            else:
                raise AssertionError('Natural drainage did not finish')
        assert module_hash(policy)==before
        m=check_metrics(env)
        value['rows'].append(dict(track=track,users=n,trace_seed=trace,phase_s=phase,measure_s=measure,
            method='TMAPO',training_seed=spec.SEED,model_hash=expected,full_input_hash=env.full_input_hash,
            local_masks=getattr(env,'measured_local_mask',0),energy_by_task_group=env.accounting(m),**m))
        print(f'Completed {variant} {track} N={n} trace={trace} phase={phase}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('variant',choices=spec.VARIANTS)
    v=p.parse_args().variant
    store.CAMPAIGN=spec.CAMPAIGN
    t=FocusTransport()
    m=dict(variant=v,seed=spec.SEED,completed=False,rows=[])
    try:
        assert store.within_dispatch_window(t)
        run(t,v,m)
        assert len(m['rows'])==len(spec.CASES)
        m['completed']=True
    except Exception as e:
        m['error']=dict(type=type(e).__name__,message=str(e))
    t.small(v+'.enc',m,v)
    return m['completed']


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

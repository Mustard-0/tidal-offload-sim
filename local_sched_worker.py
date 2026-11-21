"""Run a nonpreemptive local scheduler without changing capacity, admission or cost."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

from energy_qos_telemetry import with_task_telemetry
from evaluation_adapters import evaluation_environment_types
import local_sched_protocol as spec
import optimization2_worker as store
from review_checks import bootstrap,check_metrics
from scalability_protocol import settings


def ordered_waiting(queue,tasks,now,speed,rule):
    if len(queue)<3 or rule=='fifo':
        return list(queue)
    head,*waiting=queue
    head_done=now+tasks[head].remaining/speed
    def order(k):
        j=tasks[k]
        due=j.arrival+j.deadline
        if rule=='edf':
            return due,j.arrival,j.id
        assert rule=='deadline_spt'
        late=head_done+j.remaining/speed>due+1e-9
        return late,j.remaining,due,j.arrival,j.id
    return [head,*sorted(waiting,key=order)]


def scheduler(base,rule):
    class LocalQueue(base):
        def __init__(self,*args,**kwargs):
            self.clock_totals=defaultdict(lambda:[0,0,0.,0.])
            super().__init__(*args,**kwargs)

        def enqueue_compute(self,j,node):
            assert node == j.user and node < self.u
            super().enqueue_compute(j,node)
            q=self.compute[node]
            if q and q[-1]==j.id and rule!='fifo':
                chosen=ordered_waiting(q,self.tasks,self.time,self.speed(node),rule)
                q.clear()
                q.extend(chosen)

        def end(self,j,failed=False):
            if self.measured(j):
                group=self.clock_totals[j.profile]
                group[0]+=1
                if not failed:
                    assert j.compute_start>=j.compute_ready>=j.arrival-1e-9
                    group[1]+=1
                    group[2]+=max(0., j.compute_start-j.arrival)
                    group[3]+=self.time-j.arrival
            return super().end(j,failed=failed)

    return LocalQueue


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


def run(rule, result):
    bootstrap()
    plain,_,_,local=evaluation_environment_types()
    env_type=with_task_telemetry(scheduler(plain,rule))
    limit=time.monotonic()+spec.WALL
    for _,n,trace,phase,measure in spec.FRESH:
        if time.monotonic()>=limit:
            raise TimeoutError('Wall budget before next complete scene')
        cfg,_=settings('Local-Only',None,n)
        env=env_type(cfg,family='Local-Only',seed=trace,phase_s=phase,measure_start=spec.WARM,measure_end=spec.WARM+measure)
        policy=local()
        stop=round((spec.WARM+measure)/cfg['dt_s'])
        for i in range(stop+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
            env.step(policy,arrivals=i<stop)
            if env.slot>=stop and not env.tasks:
                break
            if i%100==99 and time.monotonic()>=limit:
                raise TimeoutError(f'Wall budget within N={n}')
        else:
            raise AssertionError('Natural drainage did not finish')
        m=check_metrics(env)
        assert sum(z[0] for z in env.clock_totals.values())==m['arrivals']
        assert sum(z[1] for z in env.clock_totals.values())==m['completed']
        result['rows'].append(dict(method='Local-Only',rule=rule,users=n,trace_seed=trace,phase_s=phase,
            measure_s=measure,full_input_hash=env.full_input_hash,clock_groups=dict(env.clock_totals),
            energy_by_task_group=env.accounting(m),**m))
        print(f'Completed {rule} N={n} trace={trace} phase={phase}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('rule',choices=spec.VARIANTS)
    rule=p.parse_args().rule
    store.CAMPAIGN=spec.CAMPAIGN
    t=Transport()
    result=dict(rule=rule,completed=False,rows=[])
    try:
        assert store.within_dispatch_window(t)
        run(rule,result)
        assert len(result['rows'])==len(spec.FRESH)
        result['completed']=True
    except Exception as e:
        result['error']=dict(type=type(e).__name__,message=str(e))
    t.small(rule+'.enc',result,rule)
    return result['completed']


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

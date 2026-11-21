"""Retain per-load receipts and shared deadline-ratio counts without raw tasks."""

import argparse
from collections import Counter
from hashlib import sha256
from struct import pack

import baseline_individual_worker as base
from energy_qos_telemetry import with_task_telemetry
import lowload_protocol as spec
import mixscreen_worker as mix
import optimization2_worker as store


def with_deadline_sweep(cls):
    class Sweep(with_task_telemetry(cls)):
        def __init__(self,*args,**kwargs):
            self.sweep=Counter()
            self._times_and_users=sha256()
            super().__init__(*args,**kwargs)

        def generate(self):
            first=self.serial
            super().generate()
            for jid in range(first,self.serial):
                j=self.tasks[jid]
                self._times_and_users.update(pack('<III',int(j.id),int(j.user),int(j.arrival_slot)))

        def end(self,task,failed=False):
            if self.measured(task):
                age=self.time-task.arrival
                for percent in spec.RATIOS:
                    self.sweep[percent] += int(failed or age>(percent/100.)*task.deadline+1e-9)
            return super().end(task,failed=failed)

        def metrics(self):
            r=super().metrics()
            n=r['arrivals']
            values=[self.sweep[p] for p in spec.RATIOS]
            assert values==sorted(values) and 0<=values[0]<=values[-1]<=n
            assert values[0]==round(n*r['qos_rate']) and values[1]==r['qos_95_count']
            return dict(r,arrival_order_hash=self._times_and_users.hexdigest(),
                        deadline_sweep_counts={str(p):self.sweep[p] for p in spec.RATIOS})
    return Sweep


class Rows(list):
    def __init__(self,t,name,method,arm):
        super().__init__()
        self.t,self.name,self.method,self.arm=t,name,method,arm

    def append(self,row):
        super().append(row)
        n=row['users']
        block=[r for r in self if r['users']==n]
        assert len(block)<=len(spec.MIXES)*len(spec.TRACES)
        if len(block)==len(spec.MIXES)*len(spec.TRACES):
            name=f'{self.name}.n{n}'
            self.t.small(name+'.enc',dict(method=self.method,arm=self.arm,users=n,
                completed=True,rows=block),name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('method')
    p.add_argument('arm')
    x=p.parse_args()
    assert (x.method,x.arm) in spec.ARMS
    store.CAMPAIGN=spec.CAMPAIGN
    base.spec=mix.spec=spec
    mix.settings=spec.settings
    mix.with_task_telemetry=with_deadline_sweep
    t=base.Transport()
    name=x.method+'.'+x.arm
    result=dict(method=x.method,arm=x.arm,completed=False,
                rows=Rows(t,name,x.method,x.arm))
    try:
        assert store.within_dispatch_window(t)
        mix.evaluate(t,x.method,x.arm,result)
        assert len(result['rows'])==len(spec.CASES)
        result['completed']=True
    except Exception as err:
        result['error']=dict(type=type(err).__name__,message=str(err))
    t.small(name+'.enc',result,name)
    return result['completed']


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

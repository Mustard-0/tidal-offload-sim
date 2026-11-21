"""Opt-in deployment changes, independent of policy weights and the energy meter."""

from collections import Counter
from math import isclose

import numpy as np


MIN_DELAY_GAIN_S = .0005
MIN_ENERGY_GAIN_J = .002


def repair_route(chosen, visible, estimates, time_ref, energy_ref):
    n = len(visible)
    t = np.asarray(estimates[:n])*time_ref
    e = np.asarray(estimates[n:2*n])*energy_ref
    options = np.flatnonzero(visible & (t <= t[chosen]-MIN_DELAY_GAIN_S)
                             & (e <= e[chosen]-MIN_ENERGY_GAIN_J))
    return min(options, key=lambda k:(e[k],t[k],int(k))) if len(options) else chosen


def dispatch_order(queue, tasks, now, speed):
    head, *waiting = queue
    if len(waiting)<2:
        return list(queue)
    def deadline(k):
        t=tasks[k]
        return t.arrival+t.deadline
    def score(order):
        t=now+tasks[head].remaining/speed
        missed=0
        finish_sum=0.
        for key in order:
            t+=tasks[key].remaining/speed
            missed+=int(t>deadline(key)+1e-9)
            finish_sum+=t-now
        return missed,round(finish_sum,10)
    options=(waiting,
             sorted(waiting,key=lambda k:(deadline(k),k)),
             sorted(waiting,key=lambda k:(tasks[k].remaining,deadline(k),k)))
    return [head,*min(options,key=score)]


def with_dispatch(base):
    class Dispatch(base):
        def __init__(self,*args,**kwargs):
            self._wrapped_policy=None
            self._just_routed=None
            self.dispatch_counts=Counter()
            self._completed_phases=Counter()
            super().__init__(*args,**kwargs)

        def route_features(self,j,xy=None,reservations=None):
            if xy is None and reservations is not None:
                self._just_routed=j
            return super().route_features(j,xy=xy,reservations=reservations)

        def step(self,policy,arrivals=True):
            if policy is not self._wrapped_policy:
                assert self._wrapped_policy is None
                self._wrapped_policy=policy
                original=policy.act
                def act(obs,mask,stage,agents):
                    selected,records=original(obs,mask,stage,agents)
                    if stage!='route':
                        return selected,records
                    assert self._just_routed is not None and len(selected)==1 and not records
                    j=self._just_routed
                    self._just_routed=None
                    before=int(selected[0])
                    after=int(repair_route(before,mask[0],obs[0],
                                           self.cfg['time_ref_s'],self.cfg['energy_ref_j']))
                    assert mask[0,after]
                    if after!=before and hasattr(policy,'stage_counts'):
                        # MARATD3's next observation consumes executed frequencies.
                        agent=int(agents[0])
                        share=1./policy.stage_counts[agent,2]
                        policy.executed[agent,6+before]-=share
                        policy.executed[agent,6+after]+=share
                        policy.prev=policy.executed.copy()
                    if self.measured(j):
                        self.dispatch_counts['routed']+=1
                        self.dispatch_counts['moved_route']+=int(after!=before)
                    return np.asarray([after]),records
                policy.act=act
            return super().step(policy,arrivals=arrivals)

        def enqueue_compute(self,j,node):
            super().enqueue_compute(j,node)
            q=self.compute[node]
            if node<self.u or len(q)<3 or q[-1]!=j.id:
                return
            ordered=dispatch_order(q,self.tasks,self.time,self.speed(node))
            if ordered!=list(q):
                assert ordered[0]==q[0]
                q.clear()
                q.extend(ordered)
                if self.measured(j):
                    self.dispatch_counts['reordered_queue']+=1

        def end(self,j,failed=False):
            if self.measured(j) and not failed:
                dispatch=j.node_choice_time if j.node>=self.u else j.compute_ready
                assert j.arrival-1e-8<=dispatch<=j.compute_ready+1e-8<=j.compute_start+2e-8
                assert j.compute_start<=self.time+1e-8
                for key,value in (
                    ('before_dispatch',dispatch-j.arrival),
                    ('uplink_and_propagation',j.compute_ready-dispatch),
                    ('compute_queue',j.compute_start-j.compute_ready),
                    ('compute_and_return',self.time-j.compute_start)):
                    self._completed_phases[key]+=max(0.,value)
            return super().end(j,failed=failed)

        def metrics(self):
            m=super().metrics()
            done=m['completed']
            assert isclose(sum(self._completed_phases.values()),done*(m['delay_s'] or 0.),
                           rel_tol=1e-8,abs_tol=1e-7)
            return dict(m,completed_phase_seconds=dict(self._completed_phases),
                        **{k:self.dispatch_counts[k] for k in ('routed','moved_route','reordered_queue')})

    return Dispatch

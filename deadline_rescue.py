"""Preserve on-time policy choices; repair only a predicted routing failure."""

from collections import Counter

import numpy as np

from physical_dispatch import with_dispatch


def rescue_route(chosen, visible, delay, energy, occupied_s, service_s, cap_s, slack):
    assert visible[chosen]
    fits = occupied_s + service_s <= cap_s + 1e-9
    if fits[chosen] and delay[chosen] <= slack + 1e-9:
        return chosen, False
    # A replacement must be expected to admit and finish before the original deadline.
    options = np.flatnonzero(visible & fits & (delay <= slack + 1e-9)
                             & (delay < delay[chosen] - .0005)
                             & (energy <= energy[chosen] + 1e-9))
    result = min(options,key=lambda k:(energy[k],delay[k],int(k))) if len(options) else chosen
    return int(result), True


def with_deadline_rescue(base):
    class Rescue(with_dispatch(base)):
        def __init__(self,*args,**kwargs):
            self.rescue_counts=Counter()
            super().__init__(*args,**kwargs)
            ground=np.arange(self.b+self.s)<self.b
            self._rescue_cpu_hz=np.where(ground,self.cfg['cpu_bs'],self.cfg['cpu_sat'])
            self._rescue_caps_seconds=np.where(ground,self.cfg['queue_cap_s'][1],self.cfg['queue_cap_s'][2])

        def step(self,policy,arrivals=True):
            if policy is not self._wrapped_policy:
                assert self._wrapped_policy is None
                self._wrapped_policy=policy
                original=policy.act

                def act(obs,mask,stage,agents):
                    chosen,records=original(obs,mask,stage,agents)
                    if stage!='route':
                        return chosen,records
                    task=self._just_routed
                    assert task is not None and len(chosen)==1 and not records
                    self._just_routed=None
                    n=len(mask[0])
                    z=obs[0]
                    t=self.cfg['time_ref_s']
                    before=int(chosen[0])
                    after,risk=rescue_route(before,mask[0],z[:n]*t,z[n:2*n]*self.cfg['energy_ref_j'],
                        z[2*n:3*n]*t,task.cycles/self._rescue_cpu_hz,self._rescue_caps_seconds,
                        task.deadline-(self.time-task.arrival))
                    if after!=before and hasattr(policy,'stage_counts'):
                        # Recurrent evaluation consumes the actual executed action.
                        agent=int(agents[0])
                        share=1./policy.stage_counts[agent,2]
                        policy.executed[agent,6+before]-=share
                        policy.executed[agent,6+after]+=share
                        policy.prev=policy.executed.copy()
                    if self.measured(task):
                        self.dispatch_counts['routed']+=1
                        self.dispatch_counts['moved_route']+=int(after!=before)
                        self.rescue_counts['at_risk']+=int(risk)
                        self.rescue_counts['normal_kept']+=int(not risk)
                    return np.asarray([after]),records
                policy.act=act
            # Keep the verified non-preemptive queue and accounting, bypass broad rerouting.
            return base.step(self,policy,arrivals=arrivals)

        def metrics(self):
            m=super().metrics()
            counts={k:self.rescue_counts[k] for k in ('at_risk','normal_kept')}
            assert counts['at_risk']+counts['normal_kept']==m['routed']
            assert 0<=m['moved_route']<=counts['at_risk']
            return dict(m,**counts)
    return Rescue

"""Distinct, deployment-only baseline candidates using legal public route state."""

import numpy as np

from local_sched_worker import ordered_waiting


def route_mask(method,legal,delay,joules,queue,capacity_s,demand,coeff,speed,slack,ground):
    legal=np.asarray(legal,dtype=bool)
    assert legal.any()
    ground=np.arange(len(legal))<ground
    service=demand/speed
    overflow=queue+service-capacity_s
    ontime=legal & (delay<=slack+1e-9)
    if method=='IPPO' and np.count_nonzero(ontime)>1:
        # Compare upload energy among actions that still fit the deadline.
        upload=np.maximum(0.,joules-demand*coeff)
        fastest=float(delay[ontime].min())
        contenders=ontime & (delay<=fastest+.01)
        take=contenders & (upload<=upload[contenders].min()+.005)
        return take
    if method=='PPO' and np.all(overflow[legal]>1e-8):
        # Existing code falls back to visibility when all queues are full.
        return legal & (overflow<=overflow[legal].min()+.005)
    if method=='SD-SAC-Adapted':
        pressured=(~ground) & (overflow>-.2*capacity_s)
        if np.any(ontime & ~pressured):
            return legal & ~pressured
    if method=='MARATD3-Adapted':
        reduce=np.zeros(len(legal),dtype=bool)
        for j in np.flatnonzero(legal & ~ground):
            equivalent=ontime & ground & (delay<=delay[j]+.005) & (joules+.005<=joules[j])
            reduce[j]=equivalent.any()
        return legal & ~reduce
    return legal


def with_public_route_rule(base,method):
    class Routed(base):
        def __init__(self,*args,**kwargs):
            self.measured_route_edits=0
            super().__init__(*args,**kwargs)

        def route_features(self,task,xy=None,reservations=None):
            obs,mask,cost=super().route_features(task,xy=xy,reservations=reservations)
            if xy is not None or reservations is None:
                return obs,mask,cost
            n=len(mask)
            on_ground=np.arange(n)<self.b
            speeds=np.where(on_ground,self.cfg['cpu_bs'],self.cfg['cpu_sat'])
            caps=np.where(on_ground,self.cfg['queue_cap_s'][1],self.cfg['queue_cap_s'][2])
            coeff=np.where(on_ground,self.cfg['energy_per_cycle'][1],self.cfg['energy_per_cycle'][2])
            scale=self.cfg['time_ref_s']
            new=route_mask(method,mask,obs[:n]*scale,obs[n:2*n]*self.cfg['energy_ref_j'],
                obs[2*n:3*n]*scale,caps,task.cycles,coeff,speeds,
                task.deadline-(self.time-task.arrival),self.b)
            assert new.any() and np.all(~new | mask)
            if not np.array_equal(mask,new):
                self.measured_route_edits+=int(self.measured(task))
                obs[3*n:4*n]=new
            return obs,new,cost

    return Routed


def adaptive_order(queue,tasks,now,speed):
    def score(order):
        time=now
        late=0
        delay=0.
        for k in order:
            t=tasks[k]
            time+=t.remaining/speed
            late+=int(time>t.arrival+t.deadline+1e-9)
            delay+=time-t.arrival
        return late,delay
    edf=ordered_waiting(queue,tasks,now,speed,'edf')
    spt=ordered_waiting(queue,tasks,now,speed,'deadline_spt')
    return min((edf,spt),key=score)


def with_adaptive_local(base):
    class LocalAdaptive(base):
        def __init__(self,*args,**kwargs):
            self.measured_queue_edits=0
            super().__init__(*args,**kwargs)

        def enqueue_compute(self,j,node):
            assert node==j.user and node<self.u
            super().enqueue_compute(j,node)
            q=self.compute[node]
            if len(q)>=3 and q[-1]==j.id:
                new=adaptive_order(q,self.tasks,self.time,self.speed(node))
                assert new[0]==q[0]
                self.measured_queue_edits+=int(list(q)!=new and self.measured(j))
                q.clear()
                q.extend(new)

    return LocalAdaptive

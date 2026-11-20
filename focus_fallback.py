"""Deployment-only fallback for heavy requests before privacy publication."""

from functools import cached_property


def prefer_local(visible, occupied, capacity, demand, local_work, local_cpu, age, deadline):
    return (age + (local_work+demand)/local_cpu <= deadline+1e-9 and
            not any(v and q+demand<=cap+1e-5 for v,q,cap in zip(visible,occupied,capacity)))


def heavy_fallback(base):
    import numpy as np

    class HeavyFallback(base):
        @cached_property
        def remote_caps(self):
            return np.asarray([self.speed(i)*self.cfg['queue_cap_s'][self.node_kind(i)]
                               for i in range(self.u,self.u+self.b+self.s)])

        def __init__(self,*args,**kwargs):
            self.measured_local_mask = 0
            super().__init__(*args,**kwargs)

        def _observations(self):
            obs, glob, masks, users, ready = super()._observations()
            snapshot = self._observation_snapshot
            assert snapshot is not None
            committed, queued = snapshot[2:4]
            occupied = committed+np.asarray(queued)
            for user in users:
                if not (masks[user,1] and masks[user,2]):
                    continue
                task = self.tasks[self.pending[user][0]]
                if task.profile != 'compute_heavy':
                    continue
                if prefer_local(self.visibility[task.region],occupied,self.remote_caps,task.cycles,
                                self.work(user),self.speed(user),self.time-task.arrival,task.deadline):
                    masks[user,2] = False
                    self.measured_local_mask += int(self.measured(task))
            return obs,glob,masks,users,ready

    return HeavyFallback

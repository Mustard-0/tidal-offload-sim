"""Cross-slot class-specific shuffler controlled by the shared W/L/O actor.

User actor: WAIT, LOCAL, OFFLOAD. Choosing OFFLOAD commits task to
a region x public-class pool. The *same* W/L/O actor then selects:
  WAIT (0) to accumulate more distinct users;
  RELEASE (2) to shuffle/publish the accumulated reports.
Action L (1) is masked at the group gate; the route actor is unchanged.
Thus only TWO trainable actor networks (release/user, route); no separate
learned third "batch" actor. The older PPO's unused batch head remains
allocated but receives zero gradients.

This is a protocol simulation, not an end-to-end adaptive/longitudinal DP
proof. A public release time and source-linked metadata can shrink anonymity.
"""
from __future__ import annotations
from collections import defaultdict,deque,Counter
import math
import numpy as np

from new_method_environment import NewManuscriptEnvironment
from environment import N_ROUTE
from privacy import local_epsilon,amplification_threshold


class CrossSlotWLOEnvironment(NewManuscriptEnvironment):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.class_staging=defaultdict(deque)  # (region, class) -> task IDs
        self.class_wait_decisions=0
        self.class_release_decisions=0
        self.sum_class_group_wait=0.
        self.max_class_group=0
        self.group_target=amplification_threshold(self.cfg)

    def end(self,j,failed=False):
        # Legacy parent expects staged tasks in region-wide self.staging[g].
        # For a class-specific pool, delete it here instead.
        if j.status=="staged":
            key=(j.region,j.profile)
            if j.id in self.class_staging.get(key,()):
                self.class_staging[key].remove(j.id)
            self.staged_users.discard(j.user)
            j.status="selected"
        return super().end(j,failed=failed)

    def _observations(self):
        context,glob,masks,users,ready=super()._observations()
        for u in users:
            j=self.tasks[self.pending[u][0]]
            key=(j.region,j.profile)
            pool=self.class_staging.get(key,())
            # Observed existing distinct users are not guaranteed to be
            # future released users if this user selects L instead.
            current=len(pool)
            population=int(self.region_counts[j.region])
            context[u,23]=current/population
            context[u,24]=(current+context[u,20]*population)/population
            context[u,25]=float(self.group_target or population+1)/population
        if self.cfg.get("hide_tidal_observations",False):
            # Remove explicit tidal signals, preserve actual arrivals and queues.
            context[:self.u,3:6]=0.
            context[:self.u,20]=0.
            context[self.u:,52]=0.
            context[self.u:,56:59]=0.
            glob[25:34]=0.
            glob[43:46]=0.
        return context,glob,masks,users,ready

    def release(self,tasks,reason="class_pool_adaptive_release"):
        if not tasks:return
        self.max_class_group=max(self.max_class_group,len(tasks))
        for j in tasks:
            self.sum_class_group_wait+=j.active_wait
        return super().release(tasks,reason=reason)

    def _group_obs(self,base,region,profile,ids):
        c=self.cfg
        obs=base.copy()
        pool=[self.tasks[i] for i in ids]
        n=len(pool)
        if n==0:raise AssertionError("No empty batch gates")
        oldest=min(pool,key=lambda j:j.arrival)
        youngest=min(pool,key=lambda j:j.deadline-(self.time-j.arrival))
        oldest_age=self.time-oldest.arrival
        min_remaining=min(j.deadline-(self.time-j.arrival) for j in pool)
        oldest_wait=max(j.active_wait for j in pool)
        max_wait=c["wait_max_s"]
        local_eps=local_epsilon(c,n,self.method=="TMAPO-NoAmp")
        population=int(self.region_counts[region])
        avail=population-n
        profiles=c.get("task_profiles",[])
        profile_prob=next((p["probability"] for p in profiles if p["name"]==profile),1/3)
        # Region-level Holt predictor / public arrival history; no future leak.
        region_rate=(population*c["arrival_per_user_s"]
                     if c.get("hide_tidal_observations",False)
                     else (self.previous_forecast[region] if self.windows_seen
                           else population*c["arrival_per_user_s"]))
        class_rate=max(0.,region_rate)*profile_prob
        remain=max(0.,min(max_wait-oldest_wait,
              min_remaining-c.get("min_service_reserve_s",.04)))
        expected_new=avail*(1-math.exp(-class_rate*remain/
                             population))
        est_n=min(population,max(n,round(n+expected_new)))
        future_eps=local_epsilon(c,est_n,self.method=="TMAPO-NoAmp")
        gain_now=1-c["epsilon_req"]/(local_eps*2*c["region_radius_m"])
        gain_future=1-c["epsilon_req"]/(future_eps*2*c["region_radius_m"])
        # Match the user actor's shared 3-actions logit cost prior:
        # action 0=WAIT (prior 0), 1=LOCAL masked, 2=RELEASE.
        obs[0:3]=[n/population,
                   oldest_wait/max(max_wait,c["dt_s"]),
                   min_remaining/max(c["deadline_s"],.01)]
        obs[17]=max(0.,oldest_wait/max_wait)  # masked L, not used for release comparison
        obs[18]=max(0.,n*0.001+ (1-gain_now))*0.20
        obs[20:26]=[n/population,local_eps*2*c["region_radius_m"],
                    remain/max(max_wait,c["dt_s"]),class_rate/100.,
                    expected_new/20.,gain_future-gain_now]
        obs[51:54]=[self.queue_pressure[region],
                    0 if c.get("hide_tidal_observations",False) else self.high_tide[region],
                    remain/max(max_wait,c["dt_s"])]
        obs[59:64]=[n/population,
                     oldest_wait/max(max_wait,c["dt_s"]),
                     min_remaining/max(c["deadline_s"],.01),
                     (n-(self.group_target or population+1))/population,
                     float(profile_prob)]
        can_hold=(remain>=c["dt_s"]-1e-9 and avail>0 and
                  self.time-oldest.arrival < max_wait-1e-9)
        mask=np.asarray([can_hold,False,True],dtype=bool)
        return obs,mask

    def step(self,policy,arrivals=True):
        c=self.cfg
        self.step_cost=0.
        self.user_step_cost[:]=0.
        if arrivals:self.generate()
        context,glob,masks,users,ready=self.observations()
        if (self.measure_start_slot <= self.slot and
            (self.measure_end_slot is None or self.slot < self.measure_end_slot)):
            self.pressure_samples+=1
            self.sum_queue_pressure+=self.queue_pressure
            self.sum_high_tide+=self.high_tide
        policy.begin(context,glob,self.u)
        records=[]
        if ready:
            order=np.random.default_rng(np.random.SeedSequence([self.seed,4,self.slot])).permutation(len(ready))
            reserved=(np.zeros(N_ROUTE,dtype=int),np.zeros(N_ROUTE))
            for ix in order:
                j=ready[int(ix)]
                feat,feasible,_=self.route_features(j,reservations=reserved)
                acts,rec=policy.act(feat[None,:],feasible[None,:],"route",
                                     np.asarray([self.u+j.region]))
                records.extend(rec)
                self.assign(j,int(acts[0]))
        if users:
            acts,rec=policy.act(context[users],masks[users],"user",np.asarray(users))
            records.extend(rec)
            for u,action in zip(users,acts):
                j=self.tasks[self.pending[u][0]]
                if action==0: # WAIT in user queue
                    j.active_wait+=c["dt_s"]
                elif action==1:
                    self.pending[u].popleft()
                    # Task is no longer pending. enqueue_compute may reject
                    # because of local queue capacity and call end(failed=True).
                    # end() must not attempt to remove it a second time.
                    j.status="selected"
                    self.enqueue_compute(j,int(u))
                elif action==2:
                    self.pending[u].popleft()
                    j.status="staged"
                    self.class_staging[(j.region,j.profile)].append(j.id)
                    self.staged_users.add(int(u))
                else:raise AssertionError("Unexpected user action")
        # Group gates use same "user" actor weights, not a separate batch actor.
        for (region,profile),ids in sorted(list(self.class_staging.items())):
            if not ids:continue
            if len({self.tasks[jid].user for jid in ids})!=len(ids):
                raise AssertionError("Group contains duplicate user")
            features,allowed=self._group_obs(context[self.u+region],region,profile,ids)
            agents=np.asarray([self.u+region],dtype=int)
            if getattr(policy,"is_rule_based",False):
                acts,rec=policy.act(features[None,:],allowed[None,:],
                                    "group",agents)
            else:
                acts,rec=policy.act(features[None,:],allowed[None,:],
                                    "user",agents)
            records.extend(rec)
            if acts[0]==0:
                self.class_wait_decisions+=1
                for jid in ids:
                    self.tasks[jid].active_wait+=c["dt_s"]
            else:
                if acts[0]!=2:raise AssertionError("Release gate chose unexpected action")
                self.class_release_decisions+=1
                batch=[self.tasks[jid] for jid in ids]
                self.class_staging[(region,profile)].clear()
                self.staged_users.difference_update(j.user for j in batch)
                self.release(batch,reason="adaptive_region_class")
        self.advance((self.slot+1)*c["dt_s"])
        self.slot+=1
        arrival_scale=max(self.u*c["arrival_per_user_s"]*c["dt_s"],1e-9)
        return -self.step_cost/arrival_scale,records,context,glob

    def metrics(self):
        out=super().metrics()
        out.update(protocol="cross_slot_region_class_adaptive_WLO",
            class_gate_wait_actions=self.class_wait_decisions,
            class_gate_release_actions=self.class_release_decisions,
            maximum_distinct_class_batch=self.max_class_group,
            class_staging_left=sum(len(a) for a in self.class_staging.values()))
        return out

"""New-manuscript W/L/O + per-(region, public task class) shuffle environment.

This module deliberately leaves legacy Environment unchanged. New behavior:
- each active user's HOL task can WAIT, execute LOCAL, or enter OFFLOAD;
- OFFLOAD tasks selected in the same 10ms slot, same region and same
  PUBLIC task class are randomized as one anonymous batch;
- ROUTE actor subsequently sees noisy report, never raw user coordinates;
- privacy location Quality Loss is charged once in team/agent reward AND metrics.
- report-aware observations expose actual same-class pending distinct users.

This is an *implementation smoke-test* of the new scheme, not a longitudinal
end-to-end DP proof. Public action timing, task metadata, group observability
and repeated-user privacy composition require a separate threat model.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from environment import Environment, N_ROUTE
from privacy import local_epsilon


class NewManuscriptEnvironment(Environment):
    def _observations(self):
        context, glob, masks, users, ready = super()._observations()
        pending_types = Counter()
        for u in users:
            task = self.tasks[self.pending[u][0]]
            pending_types[(task.region, task.profile)] += 1
        for u in users:
            j = self.tasks[self.pending[u][0]]
            age = self.time - j.arrival
            wait_remaining = self.cfg["wait_max_s"] - j.active_wait
            # A WAIT is permitted only if at least 1 slot of useful waiting
            # is possible without exhausting the stated service reserve.
            service_reserve = self.cfg.get("min_service_reserve_s", .04)
            masks[u, 0] = (
                wait_remaining >= self.cfg["dt_s"] - 1e-10 and
                j.deadline - age >= service_reserve + self.cfg["dt_s"] - 1e-10
            )
            n = pending_types[(j.region, j.profile)]
            # This is a PUBLIC count estimate, not the future real batch:
            # other users may choose L or W instead of O.
            context[u, 20] = n / max(1,self.cfg["users_per_region"])
            estimated_eps = local_epsilon(self.cfg, n, self.method == "TMAPO-NoAmp")
            context[u, 21] = estimated_eps*2*self.cfg["region_radius_m"]
            context[u, 22] = max(0.,wait_remaining) / max(self.cfg["wait_max_s"],self.cfg["dt_s"])
            # User "offload" cost prior includes the expected QL cost;
            # local execution does not publish a noisy location.
            quality_estimate = 2 / estimated_eps
            if self.cfg.get("project_public_region",False):
                quality_estimate = min(quality_estimate,
                    self.cfg["region_radius_m"]*1.07)
            context[u, 18] += (self.cfg.get("weight_quality", .25) *
                                quality_estimate /
                                self.cfg.get("quality_ref_m",1000.))
            if not masks[u].any():
                masks[u, 1] = True  # admitted or failed explicitly downstream
        return context,glob,masks,users,ready

    def release(self, tasks, reason="same_slot_region_class"):
        if not tasks:
            return
        same = {(j.region,j.profile) for j in tasks}
        if len(same)!=1 or len({j.user for j in tasks})!=len(tasks):
            raise AssertionError("New protocol batches require one public group and distinct users")
        super().release(tasks,reason=reason)
        scale = self.cfg.get("weight_quality",.25) / self.cfg.get("quality_ref_m",1000.)
        for j in tasks:
            if self.cfg.get("project_public_region",False):
                # Projecting an already DP-randomized report onto its PUBLIC
                # region is deterministic post-processing: it does not worsen
                # the one-report conditional shuffle/Geo-I guarantee.
                center=np.asarray(self.cfg["region_centers_m"][j.region],dtype=float)
                offset=j.report-center
                distance=float(np.linalg.norm(offset))
                radius=self.cfg["region_radius_m"]
                j.report=center+offset*min(1.,radius/max(distance,1e-12))
                before=j.ql
                j.ql=float(np.linalg.norm(j.report-self.xy[j.user]))
                if self.measured(j):
                    self.sum_ql += j.ql-before
            charge = scale*j.ql
            self.step_cost += charge
            self.cost_total += charge
            self.user_step_cost[j.user] += charge
        if self.collect and self.batch_rows:
            self.batch_rows[-1]["task_class"] = tasks[0].profile

    def end(self, j, failed=False):
        # Parent mean_task_cost excluded QL in the legacy manuscript.
        if self.measured(j) and j.report is not None:
            self.sum_cost += (self.cfg.get("weight_quality",.25)*
                              j.ql/self.cfg.get("quality_ref_m",1000.))
        super().end(j,failed=failed)

    def step(self, policy, arrivals=True):
        self.step_cost = 0.
        self.user_step_cost[:] = 0.
        if arrivals:
            self.generate()
        context,glob,masks,users,ready = self.observations()
        if (self.measure_start_slot <= self.slot and
            (self.measure_end_slot is None or self.slot < self.measure_end_slot)):
            self.pressure_samples += 1
            self.sum_queue_pressure += self.queue_pressure
            self.sum_high_tide += self.high_tide
        policy.begin(context,glob,self.u)
        records=[]
        if ready:
            order=np.random.default_rng(
                np.random.SeedSequence([self.seed,4,self.slot])
            ).permutation(len(ready))
            reserves=(np.zeros(N_ROUTE,dtype=int),np.zeros(N_ROUTE))
            for i in order:
                j=ready[int(i)]
                obs,feasible,_=self.route_features(j,reservations=reserves)
                acts,rec=policy.act(obs[None,:],feasible[None,:],"route",
                                    np.array([self.u+j.region]))
                records.extend(rec)
                self.assign(j,int(acts[0]))
        released={}
        if users:
            acts,rec=policy.act(context[users],masks[users],"user",np.asarray(users))
            records.extend(rec)
            for u,a in zip(users,acts):
                j=self.tasks[self.pending[u][0]]
                if a==0:  # WAIT, task remains in user's pending FIFO
                    j.active_wait += self.cfg["dt_s"]
                elif a==1:  # LOCAL, never publishes a location
                    self.pending[u].popleft()
                    # Task is no longer pending. enqueue_compute may reject
                    # because of local queue capacity and call end(failed=True).
                    # end() must not attempt to remove it a second time.
                    j.status="selected"
                    self.enqueue_compute(j,int(u))
                elif a==2:  # OFFLOAD, collect THIS-SLOT decisions
                    self.pending[u].popleft()
                    released.setdefault((j.region,j.profile),[]).append(j)
                else:
                    raise ValueError("Unexpected W/L/O action")
        for (region,profile),group in sorted(released.items()):
            self.release(group)
        self.advance((self.slot+1)*self.cfg["dt_s"])
        self.slot+=1
        arrival_scale=max(self.u*self.cfg["arrival_per_user_s"]*
                          self.cfg["dt_s"],1e-9)
        return -self.step_cost/arrival_scale,records,context,glob

    def metrics(self):
        d=super().metrics()
        d["protocol"]="same_slot_region_class_WLO"
        d["quality_cost_is_in_reward"]=True
        d["number_queued_pending"]=sum(len(x) for x in self.pending)
        return d

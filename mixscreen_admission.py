"""Add a bounded admission residual without changing actor observations, masks or DP."""

import numpy as np


def admission_delta(local_cost, remote_cost, local_on_time):
    if remote_cost is None:
        return -.5 if local_on_time else 0.
    return .5 * float(np.clip(local_cost - remote_cost, -1., 1.))


def with_admission_quote(base):
    class Quoted(base):
        def __init__(self,*args,**kwargs):
            self.admission_delta = {}
            self.measured_admission_edits = 0
            super().__init__(*args,**kwargs)

        def _observations(self):
            obs, glob, masks, users, ready = super()._observations()
            self.admission_delta = {}
            c = self.cfg
            by_user = self._observation_snapshot[1]
            quotes = {}
            for u in users:
                if not (masks[u,1] and masks[u,2]):
                    continue
                j = self.tasks[self.pending[u][0]]
                key = (j.region,j.bits,j.cycles,by_user[u])
                if key not in quotes:
                    z, visible, _ = self.route_features(j, xy=c['region_centers_m'][j.region])
                    n = len(visible)
                    ground = np.arange(n) < self.b
                    speed = np.where(ground,c['cpu_bs'],c['cpu_sat'])
                    caps = np.where(ground,c['queue_cap_s'][1],c['queue_cap_s'][2])
                    fits = visible & (z[2*n:3*n]*c['time_ref_s']+j.cycles/speed <= caps+1e-8)
                    quotes[key] = z[:n][fits], z[n:2*n][fits]
                times, joules = quotes[key]
                age = self.time - j.arrival
                local_time = age + (self.work(u)+j.cycles)/c['cpu_ue']
                local_energy = j.cycles*c['energy_per_cycle'][0]
                local_ontime = local_time <= j.deadline+1e-9
                cost_local = (c['weight_time']*local_time/c['time_ref_s'] +
                              c['weight_energy']*local_energy/c['energy_ref_j'] +
                              c['weight_violation']*(not local_ontime))
                total_time = age + c['dt_s'] + times*c['time_ref_s']
                cost_remote = (c['weight_time']*total_time/c['time_ref_s'] +
                               c['weight_energy']*(joules+c['report_energy_j']/c['energy_ref_j']) +
                               c['weight_violation']*(total_time>j.deadline+1e-9))
                delta = admission_delta(cost_local,float(cost_remote.min()) if len(times) else None,
                                        local_ontime)
                self.admission_delta[u] = delta,self.measured(j)
            return obs,glob,masks,users,ready
    return Quoted


def attach_admission(policy, env):
    original = policy.act
    assert policy.deterministic

    def act(obs,mask,stage,agents):
        if stage != 'user' or not env.admission_delta:
            return original(obs,mask,stage,agents)
        logits = policy.policy_logits(stage,policy.tensor(obs))
        legal = policy.tensor(mask).bool()
        before = logits.masked_fill(~legal,-1e9).argmax(-1).cpu().numpy()
        pairs = [env.admission_delta.get(u,(0.,False)) for u in agents]
        delta = policy.tensor([d for d,_ in pairs])
        revised = logits.clone()
        revised[:,1] -= delta
        revised[:,2] += delta
        after = revised.masked_fill(~legal,-1e9).argmax(-1).cpu().numpy()
        env.measured_admission_edits += sum(bool(measured and a!=b)
            for a,b,(_,measured) in zip(before,after,pairs))
        return after,[]
    policy.act = act

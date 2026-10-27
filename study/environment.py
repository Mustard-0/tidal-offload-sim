"""TMAPO modules are explicit; basic baselines never receive them by accident."""
from types import SimpleNamespace
import math
from experiments.lr_screen.runtime import np
from experiments.lr_screen.restricted_environment import CandidateRestrictedEnvironment
from experiments.lr_screen.mechanism import report_for
from .wait_gate import fill_probability,positive_gain
from .credit import mean_preserving_credit


class Environment(CandidateRestrictedEnvironment):
    def __init__(self,*args,family='TMAPO',**kwargs):
        self.family=family
        self.wait_reasons={}
        self.gate_checks=self.gate_allowed=0
        self._gate_cache={}
        super().__init__(*args,**kwargs)

    def generate(self):
        self.time=self.slot*self.cfg['dt_s']
        return super().generate()

    def agent_rewards(self,local_fraction):
        if not local_fraction:
            return super().agent_rewards(0.)
        if self.family!='TMAPO':raise ValueError('Basic baseline credit must remain unchanged')
        costs=self.user_step_cost.tolist()
        if not math.isclose(math.fsum(costs),self.step_cost,rel_tol=1e-9,abs_tol=1e-9):
            raise AssertionError('Physical team cost is not conserved across owners')
        values,team=mean_preserving_credit(costs,self.region_counts.tolist(),
            self.cfg['arrival_per_user_s']*self.cfg['dt_s'],local_fraction)
        if not math.isclose(team,-self.step_cost/(self.u*self.cfg['arrival_per_user_s']*self.cfg['dt_s']),abs_tol=1e-8):
            raise AssertionError('Reported objective differs from physical credit')
        return np.asarray(values,dtype=np.float32)

    def _observations(self):
        context,glob,masks,users,ready=super()._observations()
        # Objective-weighted action hints are removed for EVERY learned method.
        if self.family in ('TMAPO','IPPO','PPO'):
            context[:self.u,17:19]=0.
            context[:self.u,26]=0.
        masks[:self.u,0]=False
        # An administrative end-of-observation signal is not network information.
        glob[46:48]=0.
        if self.family!='TMAPO':
            context[:self.u,4:6]=0.
            context[self.u:,57:59]=0.
            glob[28:34]=0.
        return context,glob,masks,users,ready

    def route_features(self,*args,**kwargs):
        obs,mask,cost=super().route_features(*args,**kwargs)
        if self.family!='TMAPO':obs[57:62]=0.
        return obs,mask,cost

    def _public_cost_samples(self,tasks,n,hold):
        """Finite public-location prior, NOT actual XY or the actor's decisions.

        Queue/link snapshot is held fixed. Charge the proposed hold once and
        include deadline crossing in the same physical cost. This is a model,
        not a promise that eventual deployed costs improve.
        """
        c=self.cfg
        j=max(tasks,key=lambda a:self.time-a.arrival)
        center=np.asarray(c['region_centers_m'][j.region],float)
        estimates=[]
        for i in range(c['wait_model_samples']):
            rng=np.random.default_rng(np.random.SeedSequence([104729,j.region,i]))
            radius=c['region_radius_m']*math.sqrt(rng.random());angle=2*math.pi*rng.random()
            synthetic=center+radius*np.array([math.cos(angle),math.sin(angle)])
            _,report=report_for(synthetic,center,c['region_radius_m'],self.eps0(n),
                                104729,i,c['report_shrink'])
            public=SimpleNamespace(bits=j.bits,cycles=j.cycles,user=j.user,
                arrival=j.arrival-hold,deadline=j.deadline,region=j.region,batch_n=n,
                epsilon=self.eps0(n)/(2*c['region_radius_m']),report=report)
            _,mask,decision_cost=self.route_features(public,xy=report)
            _,_,physical_cost=self.route_features(public,xy=synthetic)
            node=int(np.argmin(np.where(mask,decision_cost,np.inf)))
            estimates.append(float(physical_cost[node]))
        return estimates

    def wait_information(self,tasks):
        c=self.cfg;n=len({j.user for j in tasks});region=tasks[0].region
        population=int(self.region_counts[region]);target=self.group_target
        waited=max(j.active_wait for j in tasks)
        remaining=min(j.deadline-(self.time-j.arrival) for j in tasks)
        horizon=max(0.,min(c['wait_max_s']-waited,remaining-c['min_service_reserve_s']))
        info=dict(n=n,target=target,probability=0.,horizon_s=horizon,
            expected_quality_gain=0.,expected_wait_cost=c['weight_time']*c['dt_s']/c['time_ref_s'],
            allow=False,reason='basic_immediate' if self.family!='TMAPO' else 'insufficient_history',
            expected_physical_gain=0.,physical_gain_lower_bound=0.)
        if self.family!='TMAPO':return info
        if self.draining:info['reason']='no_new_arrivals';return info
        if target is None:info['reason']='no_attainable_accountant_gain';return info
        if n>=target:info['reason']='already_sufficient_release_now';return info
        if horizon<c['dt_s']-1e-9:info['reason']='wait_or_deadline_budget';return info
        key=(self.slot,tuple(sorted(j.id for j in tasks)))
        if key in self._gate_cache:return dict(self._gate_cache[key])
        if self.time<c['wait_min_history_s']:return info
        start=max(0.,self.time-c['wait_history_window_s'])
        history=self.admission_history[(region,tasks[0].profile)]
        while history and history[0][0]<start:history.popleft()
        exposure=self.time-start
        if exposure<c['wait_min_history_s']-1e-9:return info
        events=sum(start<=t<=self.time for t,user in history)
        for slots in (1,2):
            hold=slots*c['dt_s']
            if hold>horizon+1e-9:continue
            probability=fill_probability(population,n,target-n,events,exposure,hold,c['wait_confidence_z'])
            info.update(probability=max(info['probability'],probability),reason='unlikely_distinct_fill')
            if probability<c['wait_fill_probability']:continue
            current=self._public_cost_samples(tasks,n,0.)
            future=self._public_cost_samples(tasks,target,hold)
            allow,gain,lower=positive_gain(current,future,c['wait_model_margin'],c['wait_confidence_z'])
            info.update(allow=allow,expected_physical_gain=gain,physical_gain_lower_bound=lower,
                horizon_s=hold,reason='positive_physical_gain' if allow else 'holding_cost_exceeds_gain')
            if allow:break
        self._gate_cache[key]=dict(info)
        return info

    def _group_obs(self,base,region,profile,ids):
        obs,mask=super()._group_obs(base,region,profile,ids)
        info=self.wait_information([self.tasks[i] for i in ids])
        self.gate_checks+=1;self.gate_allowed+=int(info['allow'])
        reason=info['reason'];self.wait_reasons[reason]=self.wait_reasons.get(reason,0)+1
        # No user-queue waiting: only this actual staged group can hold.
        obs[24]=info['expected_physical_gain'] if self.family=='TMAPO' else 0.
        obs[25]=info['expected_wait_cost']
        return obs,mask

    def step(self,*args,**kwargs):
        self._gate_cache={}
        return super().step(*args,**kwargs)

    def metrics(self):
        m=super().metrics()
        m.update(gate_checks=self.gate_checks,gate_allowed=self.gate_allowed,
                 wait_reasons=dict(self.wait_reasons),forecast_windows=self.windows_seen)
        return m

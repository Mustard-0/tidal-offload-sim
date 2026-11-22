"""Train and measure each baseline in one bounded job; export only summaries."""

import argparse
import math
import time

import baseline_individual_worker as transport
from evaluation_adapters import evaluation_environment_types
import fresh_baseline_protocol as spec
import lowload_worker as accounting
import optimization2_worker as store
from paper_load_worker import with_latency_bins
from review_checks import bootstrap, check_metrics


def rule_environment(plain):
    import numpy as np
    class Quotes(plain):
        def route_features(self,task,xy=None,reservations=None):
            obs,visible,cost=super().route_features(task,xy=xy,reservations=reservations)
            if xy is None and reservations is not None:
                self.offer=cost
            return obs,visible,cost

        def _observations(self):
            obs,glob,mask,users,ready=super()._observations()
            cfg=self.cfg
            remote_n=self.b+self.s
            _,links,booked,queued,_=self._observation_snapshot
            cap=np.array([cfg['queue_cap_s'][self.node_kind(self.u+i)]*self.speed(self.u+i)
                          for i in range(remote_n)])
            dt=cfg['dt_s']
            delay=max(dt,math.ceil(cfg['privacy_processing_s']/dt-1e-9)*dt)
            report=cfg['report_energy_j']
            quote={}
            for u in users:
                j=self.tasks[self.pending[u][0]]
                key=(j.region,j.bits,j.cycles,links[u])
                if key not in quote:
                    features,visible,_=self.route_features(j,xy=cfg['region_centers_m'][j.region])
                    quote[key]=(features[:remote_n],features[remote_n:2*remote_n],
                                visible & (np.asarray(queued)+booked+j.cycles <= cap+1e-5))
                d,e,fit=quote[key]
                total=(self.time-j.arrival+delay)/cfg['time_ref_s']+d
                cost=cfg['weight_time']*total+cfg['weight_energy']*(e+report/cfg['energy_ref_j'])
                cost+=cfg['weight_violation']*(total*cfg['time_ref_s']>j.deadline)
                obs[u,18]=min(cost[fit]) if fit.any() else 100.
            return obs,glob,mask,users,ready
    return Quotes


class Rule:
    def __init__(self, env, local):
        self.env,self.local=env,local

    def begin(self, context, glob, n_user):
        pass

    def act(self,obs,mask,stage,agents):
        import numpy as np
        if self.local:
            assert stage=='user'
            return np.ones(len(obs),dtype=np.int64),[]
        if stage=='route':
            assert len(obs)==1
            return np.argmin(np.where(mask,self.env.offer[None,:],np.inf),axis=1),[]
        cost=np.column_stack((np.full(len(obs),np.inf),obs[:,17],obs[:,18]))
        return np.argmin(np.where(mask,cost,np.inf),axis=1),[]


def parameter_hash(policy):
    from hashlib import sha256
    return sha256(b''.join(x.detach().contiguous().numpy().tobytes() for net in (policy.actor,policy.critic)
                          for x in net.parameters())).hexdigest()


def train(method,seed,plain,limit):
    from experiments.lr_screen.runtime import seed_all
    from fresh_baseline_learners import controller,torch
    seed_all(seed)
    policy=controller(method)
    start=parameter_hash(policy)
    env=plain(spec.settings(method,spec.TRAIN_USERS),family=method,seed=4300000+seed,phase_s=55.)
    blocks=[]
    block_reward=0.
    for slot in range(spec.TRAIN_SLOTS):
        reward,*_=env.step(policy,arrivals=slot<spec.TRAIN_ADMIT)
        assert math.isfinite(reward)
        block_reward+=reward
        policy.finish(reward,slot+1==spec.TRAIN_SLOTS)
        if (slot+1)%spec.BLOCK==0:
            context,glob,*_=env.observations()
            stats=policy.learn(context,glob)
            assert all(math.isfinite(v) for v in stats.values())
            blocks.append(dict(slot=slot+1,reward=block_reward,**stats))
            block_reward=0.
            if time.monotonic()>=limit:
                raise TimeoutError('Training budget')
    check_metrics(env)
    assert any('actor_loss' in p for p in blocks)
    assert all(torch.isfinite(x).all() for net in (policy.actor,policy.critic) for x in net.parameters())
    final=parameter_hash(policy)
    assert start!=final
    policy.training=False
    return policy,dict(initial_hash=start,model_hash=final,input_hash=env.full_input_hash,
                       steps=env.slot,arrivals=env.measure_arrivals,updates=blocks)


def evaluate(method,seed,policy,model,env_type,n,trace,phase,limit):
    from experiments.lr_screen.runtime import seed_all,torch
    seed_all(trace)
    cfg=spec.settings(method,n)
    env=env_type(cfg,family=method,seed=trace,phase_s=phase,
                 measure_start=spec.WARM,measure_end=spec.WARM+spec.MEASURE)
    q=policy if policy is not None else Rule(env,method=='Local-Only')
    if method=='MARATD3-Adapted':
        q.history=None
    stop=round((spec.WARM+spec.MEASURE)/cfg['dt_s'])
    with torch.no_grad():
        for slot in range(stop+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
            env.step(q,arrivals=slot<stop)
            if env.slot>=stop and not env.tasks:
                break
            if slot%100==99 and time.monotonic()>=limit:
                raise TimeoutError(f'Evaluation wall budget N={n}')
        else:
            raise AssertionError('Cohort did not drain')
    metrics=check_metrics(env)
    if policy is not None:
        assert model==parameter_hash(policy)
    return dict(method=method,training_seed=seed or None,users=n,trace_seed=trace,phase_s=phase,
                measure_s=spec.MEASURE,model_hash=model,full_input_hash=env.full_input_hash,
                energy_by_task_group=env.accounting(metrics),**metrics)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('method')
    p.add_argument('seed',type=int)
    args=p.parse_args()
    method,seed=args.method,args.seed
    name=spec.cell_id(method,seed)
    bootstrap()
    from experiments.lr_screen.runtime import torch
    torch.set_num_threads(1)
    transport.spec=accounting.spec=spec
    store.CAMPAIGN=spec.CAMPAIGN
    t=transport.Transport()
    plain,*_=evaluation_environment_types()
    base=rule_environment(plain) if method.startswith('Greedy-') else plain
    env_type=with_latency_bins(accounting.with_deadline_sweep(base))
    value=dict(method=method,seed=seed,completed=False,rows=[],training=None)
    limit=time.monotonic()+spec.WALL
    try:
        assert store.within_dispatch_window(t)
        policy=None
        if seed:
            policy,value['training']=train(method,seed,plain,limit)
        model=value['training']['model_hash'] if seed else None
        trained=name+'.training'
        t.small(trained+'.enc',dict(method=method,seed=seed,training=value['training']),trained)
        for offset in range(0,len(spec.POPS),4):
            block=[]
            for n in spec.POPS[offset:offset+4]:
                for trace,phase in spec.TRACES:
                    assert time.monotonic()<limit
                    row=evaluate(method,seed,policy,model,env_type,n,trace,phase,limit)
                    value['rows'].append(row)
                    block.append(row)
                print(f'{name} finished N={n}',flush=True)
            group=f'{name}.block{offset//4}'
            t.small(group+'.enc',dict(method=method,seed=seed,rows=block),group)
        value['completed']=True
    except Exception as exc:
        value['error']=dict(type=type(exc).__name__,message=str(exc))
        print(repr(exc),flush=True)
    t.small(name+'.enc',value,name)
    return value['completed']


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

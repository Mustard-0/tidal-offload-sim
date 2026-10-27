"""Short natural-drain episodes, boundary checkpoints, no other experiment grid."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from experiments.lr_screen.runtime import (np,torch,seed_all,rng_state,rng_load,
    json_write,csv_write,checkpoint_write,available_ram_gib,runtime_info)
from experiments.lr_screen.algorithms import learner_for
from .environment import Environment
from .protocol import (configuration,plan,digest,source_hashes,PHASES,
                       VALIDATION_SEEDS,TEST_SEEDS,POPULATIONS)


def new_environment(cfg,method,seed,episode,collect=False):
    phase=PHASES[episode%len(PHASES)]
    trace=int(np.random.SeedSequence([seed,901,episode]).generate_state(1)[0])
    e=Environment(cfg,seed=trace,family=method,phase_s=phase,collect=collect)
    e.arrival_stop_slot=100
    return e


def episode(e,policy,learning=False):
    total=0.
    for slot in range(303):
        reward,records,_,_=e.step(policy,arrivals=slot<100)
        if not math.isfinite(float(reward)):raise FloatingPointError('Nonfinite reward')
        done=e.slot>=100 and not e.tasks
        if learning:policy.remember(e.agent_rewards(0.),records,done)
        total+=float(reward)
        if done:break
    else:raise AssertionError('Natural drainage exceeded lifetime')
    if e.arrivals!=e.ended or e.tasks:raise AssertionError('Unresolved terminal tasks')
    expected=-e.cost_total/(e.u*e.cfg['arrival_per_user_s']*e.cfg['dt_s'])
    if not math.isclose(total,expected,abs_tol=1e-8,rel_tol=1e-10):
        raise AssertionError('Terminal costs lost from episode reward')
    stats={}
    if learning:
        context,glob,*_=e.observations()
        stats=policy.update(context,glob)
        if not all(math.isfinite(float(v)) for v in stats.values()):
            raise FloatingPointError('Nonfinite PPO diagnostics')
        if any(not torch.isfinite(p).all() for module in (policy.actors,policy.critic) for p in module.parameters()):
            raise FloatingPointError('Nonfinite model parameter')
    return total,stats


def actor_hash(policy):
    return digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                   for k,v in policy.actors.state_dict().items()})


def evaluate(policy,method,seed,out,stage,terminal=False):
    before=actor_hash(policy);rows=[]
    traces=TEST_SEEDS if terminal else VALIDATION_SEEDS
    populations=POPULATIONS if terminal else (100,)
    modes=('sampled','argmax') if terminal else ('sampled',)
    phases=PHASES if terminal else (30.,150.)
    for n in populations:
        cfg,_=configuration(method,seed,n)
        for mode in modes:
            policy.deterministic=mode=='argmax'
            for trace in traces:
                for phase in phases:
                    seed_all(int(np.random.SeedSequence([trace,seed,20261028,int(phase)]).generate_state(1)[0]))
                    policy.reset_history()
                    e=Environment(cfg,seed=trace,family=method,phase_s=phase,
                                  measure_end=1.,collect=True)
                    e.arrival_stop_slot=100
                    episode(e,policy)
                    m=e.metrics()
                    key=f'{stage}_N{n}_{mode}_t{trace}_p{int(phase)}'
                    folder=out/'evaluation'/key;folder.mkdir(parents=True,exist_ok=False)
                    csv_write(folder/'tasks.csv',e.rows);csv_write(folder/'batches.csv',e.batch_rows)
                    if e.wait_decision_rows:csv_write(folder/'waits.csv',e.wait_decision_rows)
                    json_write(folder/'metrics.json',m)
                    rows.append(dict(method=method,training_seed=seed,users=n,mode=mode,
                        trace_seed=trace,phase_s=phase,**m))
    if actor_hash(policy)!=before:raise AssertionError('Evaluation modified actor weights')
    policy.deterministic=False
    csv_write(out/f'{stage}_metrics.csv',rows)
    return rows


def train(out,method,seed,episodes=2000,resume=False,maximum_seconds=10800,stop_at=None,component=False):
    cfg,t=configuration(method,seed)
    t['episodes']=t['updates']=episodes
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if available_ram_gib()<1.:raise MemoryError('Need1GiB available before learning')
    lock=out/'.worker.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    try:
        seed_all(seed)
        identity=dict(method=method,seed=seed,environment=cfg,training=t,
            plan_hash=digest(plan()),source_hashes=source_hashes(),component_only=component)
        ih=digest(identity)
        policy=learner_for(method,t,cfg,'cpu')
        rows=[];completed=env_steps=0;train_wall=0.
        if resume:
            saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
            if saved['identity_hash']!=ih or (out/'failure.json').exists():
                raise ValueError('Changed source/config or failed prefix')
            policy.load(saved['learner']);rng_load(saved['rng'])
            rows=saved['rows'];completed=saved['completed_episodes'];env_steps=saved['env_steps']
            train_wall=saved['training_wall_s']
        else:
            if (out/'latest.pt').exists():raise FileExistsError('Never overwrite an existing prefix')
            json_write(out/'manifest.json',dict(identity,identity_hash=ih,runtime=runtime_info('cpu')))
            if not component:
                state=rng_state();evaluate(policy,method,seed,out,'initial_validation');rng_load(state)
        def checkpoint():
            checkpoint_write(out/'latest.pt',dict(identity_hash=ih,learner=policy.state(),rng=rng_state(),
                rows=rows,completed_episodes=completed,env_steps=env_steps,training_wall_s=train_wall))
            csv_write(out/'episode_returns.csv',rows)
            json_write(out/'status.json',dict(status='training',method=method,seed=seed,
                episodes=completed,env_steps=env_steps,pending_tasks=0,training_wall_s=train_wall))
        if not resume:checkpoint()
        entered=last=time.monotonic()
        for index in range(completed,episodes):
            tick=time.monotonic()
            e=new_environment(cfg,method,seed,index)
            policy.reset_history();policy.set_progress(index+1,episodes)
            reward,stats=episode(e,policy,learning=True)
            completed=index+1;env_steps+=e.slot;train_wall+=time.monotonic()-tick
            m=e.metrics()
            rows.append(dict(episode=completed,episode_return=reward,env_steps=env_steps,
                arrivals=e.arrivals,ended=e.ended,drain_steps=e.slot-100,trace_hash=e.arrival_hash,
                qos_rate=m['qos_rate'],energy_j=m['energy_j_per_arrival'],delay_s=m['delay_s'],
                active_wait_s=m['active_wait_s'],max_active_wait_s=e.max_observed_wait,
                gate_allowed=e.gate_allowed,gate_checks=e.gate_checks,forecast_windows=e.windows_seen,
                wait_actions=e.class_wait_decisions,training_wall_s=train_wall,**stats))
            reason=('stop file' if (out/'STOP_AFTER_EPISODE').exists() else
                    'RAM below.5GiB' if available_ram_gib()<.5 else
                    'wall budget' if time.monotonic()-entered>=maximum_seconds else
                    'test stop' if stop_at and completed>=stop_at else None)
            if reason or completed%t['checkpoint_episodes']==0 or time.monotonic()-last>=t['checkpoint_seconds'] or completed==episodes:
                checkpoint();last=time.monotonic()
            if reason and completed<episodes:
                json_write(out/'paused.json',dict(reason=reason,episodes=completed));return False
        if not component:
            state=rng_state()
            evaluate(policy,method,seed,out,'terminal_validation')
            evaluate(policy,method,seed,out,'test',terminal=True)
            rng_load(state)
        json_write(out/'complete.json',dict(status='completed',method=method,seed=seed,
            episodes=completed,env_steps=env_steps,all_tasks_resolved=True,
            component_only=component,actor_hash=actor_hash(policy)))
        if (out/'paused.json').exists():(out/'paused.json').unlink()
        return True
    except BaseException as error:
        json_write(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        os.close(fd);lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=('TMAPO','IPPO','PPO'),required=True)
    p.add_argument('--seed',type=int,default=87021);p.add_argument('--resume',action='store_true')
    args=p.parse_args();train(args.out,args.method,args.seed,resume=args.resume)

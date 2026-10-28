"""Four continuing full cycles, exact resumable boundaries and terminal models."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time
import traceback
from .protocol import configuration,source_hashes,digest,VALIDATION_SEEDS
from .environment import Environment
from .policies import learner,weights_only,evaluation_policy,set_progress,update
from .records import flush,pooled
from experiments.lr_screen.runtime import (torch,np,seed_all,rng_state,rng_load,
    checkpoint_write,json_write,csv_write,available_ram_gib,runtime_info)


def model_hash(p,critic=False):
    module=(p.critic if hasattr(p,'actors') else p.q) if critic else (p.actors if hasattr(p,'actors') else p.actor)
    return digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in module.state_dict().items()})


def environment(cfg,t,method,seed,cycle):
    trace=int(np.random.SeedSequence([seed,2801,cycle]).generate_state(1)[0])
    e=Environment(cfg,seed=trace,phase_s=0.,family=method,collect=True)
    e.arrival_stop_slot=t['admission_steps']
    return e


def infer(policy,cfg,method,trace,phase,warmup,measure,out):
    e=Environment(cfg,seed=trace,phase_s=phase,family=method,
        measure_start=warmup,measure_end=warmup+measure,collect=True)
    stop=round((warmup+measure)/cfg['dt_s']);e.arrival_stop_slot=stop
    policy.reset_history()
    before=model_hash(policy)
    chunk=0
    with torch.no_grad():
        for slot in range(stop+203):
            e.step(policy,arrivals=slot<stop)
            if slot%100==99:
                flush(e,out,chunk);chunk+=1
            if e.slot>=stop and not e.tasks:
                break
        else:
            raise AssertionError('Natural inference drainage failed')
    flush(e,out,chunk)
    assert e.arrivals==e.ended and not e.tasks and model_hash(policy)==before
    return dict(e.metrics(),full_input_hash=e.full_input_hash,users=e.u,trace_seed=trace,
                phase_s=phase,warmup_s=warmup,measure_s=measure,all_tasks_resolved=True)


def validate(p,cfg,t,method,seed,out,block):
    path=out/f'validation_{block:04d}.json'
    if path.exists():
        return
    saved_rng=rng_state()
    rows=[]
    try:
        state=weights_only(p)
        clone=evaluation_policy(method,t,cfg,state)
        for trace in VALIDATION_SEEDS:
            for phase in (30.,150.):
                seed_all(int(np.random.SeedSequence([trace,seed,2802,int(phase)]).generate_state(1)[0]))
                folder=out/'validation_raw'/f'u{block}_t{trace}_p{int(phase)}'
                rows.append(infer(clone,cfg,method,trace,phase,5.,10.,folder))
        json_write(path,dict(update=block,env_steps=block*t['rollout_steps'],rows=rows,
            mean_task_cost=pooled(rows,'mean_task_cost'),delay_s=pooled(rows,'delay_s'),
            energy_j=pooled(rows,'energy_j_per_arrival'),qos_rate=pooled(rows,'qos_rate')))
    finally:
        rng_load(saved_rng)


def train(out,method,seed,resume=False,maximum_seconds=10800,component=False,stop_update=None):
    cfg,t=configuration(method,seed)
    if component:
        t.update(updates=6,cycle_steps=600,admission_steps=400,episodes=1,validation_updates=())
    out=Path(out)
    if not resume:
        out.mkdir(parents=True,exist_ok=False)
    if (out/'failure.json').exists() or (out/'complete.json').exists():
        raise ValueError('No blind resume of failed or completed cells')
    if available_ram_gib()<1.5:
        raise MemoryError('Need1.5GiB before initializing training')
    fd=os.open(out/'.worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode())
    p=None;block=steps=cycle=0;rows=[];cycles=[];env=None;wall=0.
    try:
        seed_all(seed)
        actual=runtime_info('cpu')
        runtime={k:actual[k] for k in ('numpy','scipy','torch','device','torch_threads')}
        runtime['python']=actual['python'].split()[0]
        identity=dict(method=method,seed=seed,environment=cfg,training=t,component_only=component,
            source_hashes=source_hashes(),runtime=runtime)
        ident=digest(identity)
        p=learner(method,t,cfg)
        initial_actor_hash=model_hash(p)
        if resume:
            manifest=__import__('json').loads((out/'manifest.json').read_text())
            assert manifest['identity_hash']==ident
            saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
            assert saved['identity_hash']==ident
            p.load(saved['learner']);rng_load(saved['rng'])
            block,steps,cycle,env=saved['update'],saved['env_steps'],saved['cycle'],saved['environment']
            rows,cycles,wall=saved['rows'],saved['cycles'],saved['training_wall_s']
            initial_actor_hash=saved['initial_actor_hash']
        else:
            json_write(out/'manifest.json',dict(identity,identity_hash=ident,execution_runtime=actual,
                source_commit=os.environ.get('GITHUB_SHA','local-unpublished')))
        started=last_checkpoint=time.monotonic()

        def checkpoint():
            checkpoint_write(out/'latest.pt',dict(identity_hash=ident,learner=p.state(),rng=rng_state(),
                update=block,env_steps=steps,cycle=cycle,environment=env,rows=rows,cycles=cycles,
                training_wall_s=wall,initial_actor_hash=initial_actor_hash))
            csv_write(out/'training.csv',rows);csv_write(out/'cycles.csv',cycles)
            json_write(out/'status.json',dict(update=block,env_steps=steps,cycle=cycle,
                pending_tasks=len(env.tasks) if env else 0,training_wall_s=wall,component_only=component,
                actor_updated=model_hash(p)!=initial_actor_hash,identity_hash=ident))

        if not resume:
            if not component:
                validate(p,cfg,t,method,seed,out,0)
            checkpoint()
        while block<t['updates']:
            if env is None:
                env=environment(cfg,t,method,seed,cycle);p.reset_history()
            tick=time.monotonic();set_progress(p,block+1,t['updates'])
            total=0.;done=False
            for slot in range(t['rollout_steps']):
                reward,records,_,_=env.step(p,arrivals=env.slot<env.arrival_stop_slot)
                assert math.isfinite(reward)
                done=env.slot==t['cycle_steps']
                p.remember(env.agent_rewards(t['local_credit_fraction']),records,done)
                total+=float(reward);steps+=1
                if done:
                    assert not env.tasks
            context,glob=env.observations()[:2] if done else env.prepare(env.slot<env.arrival_stop_slot)
            stats=update(p,context,glob)
            if not all(math.isfinite(float(v)) for v in stats.values()):
                raise FloatingPointError('Nonfinite learner diagnostic')
            for module in ((p.actors,p.critic) if hasattr(p,'actors') else (p.actor,p.q)):
                if any(not torch.isfinite(v).all() for v in module.parameters()):
                    raise FloatingPointError('Nonfinite model parameter')
            block+=1;wall+=time.monotonic()-tick
            flush(env,out/'training_raw',block)
            rows.append(dict(update=block,env_steps=steps,cycle=cycle,cycle_slot=env.slot,
                users=env.u,rollout_return=total,arrivals_so_far=env.arrivals,
                trace_hash=env.arrival_hash,full_input_hash=env.full_input_hash,
                training_wall_s=wall,**stats))
            if done:
                m=env.metrics()
                assert env.arrivals==env.ended and math.isclose(env.cost_total,m['mean_task_cost']*m['arrivals'],abs_tol=1e-5)
                cycles.append(dict(cycle=cycle,env_steps=steps,full_input_hash=env.full_input_hash,**m))
                cycle+=1;env=None
            if not component and block in t['validation_updates']:
                validate(p,cfg,t,method,seed,out,block)
            reason=('explicit stop' if (out/'STOP_AFTER_UPDATE').exists() else
                    'wall budget' if time.monotonic()-started>=maximum_seconds else
                    'component test stop' if stop_update and block>=stop_update else
                    'RAM below.5GiB' if available_ram_gib()<.5 else None)
            if reason or block%t['checkpoint_updates']==0 or block==t['updates'] or time.monotonic()-last_checkpoint>=t['checkpoint_seconds']:
                checkpoint()
                last_checkpoint=time.monotonic()
            if reason and block<t['updates']:
                json_write(out/'paused.json',dict(reason=reason,update=block,env_steps=steps));return False
        assert env is None and steps==t['updates']*t['rollout_steps']
        checkpoint_write(out/'terminal_policy.pt',dict(identity_hash=ident,method=method,seed=seed,
            environment=cfg,training=t,policy=weights_only(p),component_only=component))
        json_write(out/'complete.json',dict(update=block,env_steps=steps,cycles=cycle,pending_tasks=0,
            component_only=component,actor_hash=model_hash(p),critic_hash=model_hash(p,True)))
        if (out/'paused.json').exists():
            (out/'paused.json').unlink()
        return True
    except BaseException as error:
        json_write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            update=block,env_steps=steps,no_automatic_rerun=True))
        raise
    finally:
        os.close(fd);(out/'.worker.lock').unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--maximum-seconds',type=float,default=10800)
    a=p.parse_args();done=train(a.out,a.method,a.seed,a.resume,a.maximum_seconds)
    raise SystemExit(0 if done else 2)

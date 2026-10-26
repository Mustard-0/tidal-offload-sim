"""Equal-step continuing training with full mid-cycle resumable state."""
import argparse
import copy
import gzip
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from .protocol import ROOT, configuration, population, source_hashes, digest, VALIDATION_SEEDS, TEST_SEEDS, POPULATIONS
from .environment import Environment
from .policies import learner, weights_only, evaluation_policy, set_progress, update
from experiments.lr_screen.runtime import (torch,np,seed_all,rng_state,rng_load,
    checkpoint_write,json_write,csv_write,available_ram_gib,runtime_info)
from .metrics import audit_tasks, pooled
from environment import GreedyPolicy


def episode_environment(cfg,t,seed,cycle,collect=False):
    n=t['populations'][cycle%len(t['populations'])]
    c=population(cfg,n)
    c['observation_horizon_slots']=t['cycle_steps']
    e_seed=int(np.random.SeedSequence([seed,2701,cycle]).generate_state(1)[0])
    e=Environment(c,seed=e_seed,phase_s=(cycle%5)*60.,collect=collect,
        ablation=t.get('ablation'))
    e.arrival_stop_slot=t['cycle_steps']-t['drain_slots']
    return e


def model_hash(p):
    module=p.actors if hasattr(p,'actors') else p.actor
    return digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in module.state_dict().items()})


def gzip_rows(path,rows):
    if not rows:return
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(rows)


def evaluate_one(p,cfg,method,trace,phase,warmup=3.,measure=3.,collect=True,environment_class=None):
    cls=environment_class or Environment
    extra={} if environment_class else dict(ablation=method)
    e=cls(cfg,seed=trace,phase_s=phase,measure_start=warmup,
        measure_end=warmup+measure,collect=collect,**extra)
    slots=round((warmup+measure)/cfg['dt_s'])
    e.arrival_stop_slot=slots
    if hasattr(p,'reset_history'):p.reset_history()
    before=model_hash(p) if hasattr(p,'actor') or hasattr(p,'actors') else None
    with torch.no_grad():
        for slot in range(slots+203):
            e.step(p,arrivals=slot<slots)
            if e.slot>=slots and not e.tasks:break
        else:raise AssertionError('Evaluation drainage failed')
    m=e.metrics()
    if collect:audit_tasks(e,m)
    if e.arrivals!=e.ended or e.tasks:raise AssertionError('Evaluation lost tasks')
    if before is not None and before!=model_hash(p):raise AssertionError('Inference mutated weights')
    if hasattr(p,'buffer') and p.buffer:raise AssertionError('Evaluation stored training rollout')
    m.update(users=e.u,trace_seed=trace,phase_s=phase,actual_steps=e.slot,
             forecast_windows=e.windows_seen,max_active_wait_s=e.max_observed_wait)
    return m,e


def validate(p,cfg,t,method,out,block,component=False):
    path=out/f'validation_{block:04d}.json'
    if path.exists():return json.loads(path.read_text())
    old_rng=rng_state()
    rows=[]
    try:
        state=weights_only(p)
        seeds=VALIDATION_SEEDS[:1] if component else VALIDATION_SEEDS
        populations=(6,) if component else (200,300,400)
        phases=(30.,) if component else (30.,150.)
        for n in populations:
            c=population(cfg,n)
            clone=evaluation_policy(method,t,c,state)
            for trace in seeds:
                for phase in phases:
                    seed_all(int(np.random.SeedSequence([trace,2702,int(phase)]).generate_state(1)[0]))
                    m,_=evaluate_one(clone,c,method,trace,phase,.1 if component else 3.,.1 if component else 3.)
                    rows.append(m)
            del clone
        result=dict(block=block,env_steps=block*t['rollout_steps'],rows=rows,
            cost=pooled(rows,'mean_task_cost'),delay_s=pooled(rows,'delay_s'),
            energy_j=pooled(rows,'energy_j_per_arrival'),qos=pooled(rows,'qos_rate'),
            selection_eligible=not component,execution='sampled categorical / deterministic TD3 intent')
        json_write(path,result)
        return result
    finally:rng_load(old_rng)


def train(out,method,lr,seed,resume=False,max_seconds=10800,component=None,stop_update=None):
    cfg,t=configuration(method,lr,seed)
    t['ablation']=method if method.startswith('TMAPO-') else None
    if component:t.update(component)
    if t['cycle_steps']%t['rollout_steps'] or t['updates']*t['rollout_steps']%t['cycle_steps']:
        raise ValueError('Budget must contain full cycles and rollout boundaries')
    if t['drain_slots']*cfg['dt_s']<cfg['lifetime_s']:
        raise ValueError('Fixed drainage window shorter than task lifetime')
    out=Path(out)
    if not resume:out.mkdir(parents=True,exist_ok=False)
    if (out/'failure.json').exists() or (out/'complete.json').exists():
        raise ValueError('No blind resume of failure or completed result')
    lock=out/'.worker.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode())
    try:return _train(out,method,lr,seed,cfg,t,resume,max_seconds,bool(component),stop_update)
    finally:os.close(fd);lock.unlink()


def _train(out,method,lr,seed,cfg,t,resume,max_seconds,component,stop_update):
    seed_all(seed)
    actual=runtime_info('cpu')
    normalized_runtime={k:actual[k] for k in ('numpy','scipy','torch','device','torch_threads')}
    normalized_runtime['python']=actual['python'].split()[0]
    identity=dict(schema='highload-midcycle-v1',method=method,lr=lr,seed=seed,
        environment=cfg,training=t,source_hashes=source_hashes(),component_only=component,
        runtime=normalized_runtime)
    ident=digest(identity)
    p=learner(method,t,cfg)
    block=steps=cycle=0
    rows=[];cycles=[];env=None;wall=0.
    if resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['identity_hash']!=ident:raise ValueError('Changed scientific source/config/runtime')
        manifest.setdefault('execution_runtimes',[]).append(actual)
        json_write(out/'manifest.json',manifest)
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['identity_hash']!=ident:raise ValueError('Wrong checkpoint')
        p.load(saved['learner']);rng_load(saved['rng'])
        block,steps,cycle,env=saved['update'],saved['env_steps'],saved['cycle'],saved['environment']
        rows,cycles,wall=saved['rows'],saved['cycles'],saved['training_wall_s']
    else:
        json_write(out/'manifest.json',dict(identity,identity_hash=ident,
            source_commit=os.environ.get('GITHUB_SHA','local-unpublished'),
            execution_runtime=actual,execution_runtimes=[actual]))
    started=time.monotonic()

    def checkpoint():
        checkpoint_write(out/'latest.pt',dict(identity_hash=ident,learner=p.state(),rng=rng_state(),
            update=block,env_steps=steps,cycle=cycle,environment=env,rows=rows,cycles=cycles,training_wall_s=wall))
        csv_write(out/'training.csv',rows);csv_write(out/'cycles.csv',cycles)
        json_write(out/'checkpoint_status.json',dict(update=block,env_steps=steps,cycle=cycle,
            pending_tasks=len(env.tasks) if env else 0,prepared_slot=env.prepared_slot if env else None,
            training_wall_s=wall,component_only=component,identity_hash=ident))

    try:
        if not resume:
            validate(p,cfg,t,method,out,0,component)
            checkpoint()
        while block<t['updates']:
            if env is None:
                env=episode_environment(cfg,t,seed,cycle)
                p.reset_history()
            tick=time.monotonic()
            set_progress(p,block+1,t['updates'])
            total=0.;done=False
            for _ in range(t['rollout_steps']):
                reward,records,_,_=env.step(p,arrivals=env.slot<env.arrival_stop_slot)
                done=env.slot==t['cycle_steps']
                if not math.isfinite(float(reward)):raise FloatingPointError('Nonfinite team reward')
                p.remember(env.agent_rewards(0.),records,done)
                total+=float(reward);steps+=1
                if done and env.tasks:raise AssertionError('Fixed cycle did not naturally drain')
            # The next decision's actual CURRENT arrivals are prepared once.
            # No future arrivals are read; the cached state is saved for resume.
            context,glob=(env.observations()[:2] if done else env.prepare(env.slot<env.arrival_stop_slot))
            stats=update(p,context,glob)
            if not all(math.isfinite(float(v)) for v in stats.values()):raise FloatingPointError('Nonfinite optimizer diagnostic')
            modules=(p.actors,p.critic) if hasattr(p,'actors') else (p.actor,p.q)
            if any(not torch.isfinite(v).all() for module in modules for v in module.parameters()):
                raise FloatingPointError('Nonfinite learner parameter')
            block+=1;wall+=time.monotonic()-tick
            rows.append(dict(update=block,env_steps=steps,users=env.u,cycle=cycle,
                cycle_slot=env.slot,rollout_return=total,arrivals_so_far=env.arrivals,
                trace_hash=env.arrival_hash,forecast_windows=env.windows_seen,
                training_wall_s=wall,**stats))
            if done:
                m=env.metrics()
                if env.arrivals!=env.ended or not math.isclose(env.cost_total,m['mean_task_cost']*m['arrivals'],abs_tol=1e-5):
                    raise AssertionError('Cycle cost/conservation mismatch')
                cycles.append(dict(cycle=cycle,env_steps=steps,users=env.u,**m))
                cycle+=1;env=None
            if block in t['validation_updates']:
                validate(p,cfg,t,method,out,block,component)
            reason=('explicit stop' if (out/'STOP_AFTER_UPDATE').exists() else
                    'wall budget' if time.monotonic()-started>=max_seconds else
                    'component boundary' if stop_update and block>=stop_update else
                    'available RAM below .5GiB' if available_ram_gib()<.5 else None)
            if reason or block%t['checkpoint_updates']==0 or block==t['updates']:
                checkpoint()
            if reason and block<t['updates']:
                json_write(out/'paused.json',dict(reason=reason,update=block,env_steps=steps));return False
            print('UPDATE',method,block,'env_steps',steps,'users',rows[-1]['users'],flush=True)
        if env is not None or steps!=t['updates']*t['rollout_steps']:
            raise AssertionError('Incomplete fixed-step budget')
        json_write(out/'complete.json',dict(update=block,env_steps=steps,cycles=cycle,
            pending_tasks=0,component_only=component,terminal_validation=f'validation_{block:04d}.json'))
        if (out/'paused.json').exists():(out/'paused.json').unlink()
        return True
    except BaseException as error:
        json_write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            recover_from='last complete checkpoint only; no automatic retry',update=block,env_steps=steps))
        raise


def test_model(out,method,lr,seed,checkpoint_folder=None,max_seconds=10800):
    """Resumable final paired full-cycle inference. Not called by training."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg,t=configuration('TMAPO' if method in ('Greedy','GeoI-Greedy') else method,lr if lr else 1,seed if seed else 87021)
    model=None
    if checkpoint_folder:
        folder=Path(checkpoint_folder)
        manifest=json.loads((folder/'manifest.json').read_text())
        done=json.loads((folder/'complete.json').read_text())
        if done['component_only'] or done['env_steps']!=300000:raise ValueError('Not a formal trained model')
        saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if saved['identity_hash']!=manifest['identity_hash']:raise ValueError('Wrong model identity')
        model=learner(method,t,cfg);model.load(saved['learner']);state=weights_only(model)
        del model,saved
    started=time.monotonic();rows=[]
    for n in POPULATIONS:
        c=population(cfg,n)
        p=(GreedyPolicy(c) if method in ('Greedy','GeoI-Greedy') else evaluation_policy(method,t,c,state))
        for trace in TEST_SEEDS:
            case=out/f'N{n}_trace{trace}'
            if (case/'complete.json').exists():
                result=json.loads((case/'complete.json').read_text());rows.append(result['metrics']);continue
            seed_all(int(np.random.SeedSequence([trace,2703]).generate_state(1)[0]))
            if method=='GeoI-Greedy':
                from .geoi import evaluate_geoi
                m,e=evaluate_geoi(p,c,trace,0.,5.,300.)
            else:m,e=evaluate_one(p,c,method,trace,0.,5.,300.)
            case.mkdir(exist_ok=True)
            gzip_rows(case/'tasks.csv.gz',e.rows);gzip_rows(case/'batches.csv.gz',e.batch_rows)
            result=dict(method=method,lr=lr,seed=seed,metrics=m,
                model_identity=manifest['identity_hash'] if checkpoint_folder else None,
                hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in case.glob('*.gz')})
            json_write(case/'complete.json',result);rows.append(m)
            if time.monotonic()-started>max_seconds:
                json_write(out/'paused.json',dict(reason='wall budget',scenes=len(rows)));return False
        del p
    json_write(out/'complete.json',dict(method=method,lr=lr,seed=seed,scenes=len(rows),
        rows=rows,all_raw_audits_passed=True,model_identity=manifest['identity_hash'] if checkpoint_folder else None))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--method',required=True)
    p.add_argument('--lr',type=int,default=1);p.add_argument('--seed',type=int,default=87021)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-seconds',type=float,default=10800)
    args=p.parse_args()
    train(args.out,args.method,args.lr,args.seed,args.resume,args.max_seconds)

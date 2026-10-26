"""100 arrival slots per true episode; naturally drain and retain every cost."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
from .protocol import configuration, digest, source_hashes
from experiments.lr_screen.runtime import (np, torch, seed_all, rng_state, rng_load,
    checkpoint_write, json_write, csv_write, runtime_info, available_ram_gib)
from experiments.lr_screen.algorithms import learner_for
from experiments.lr_screen.restricted_environment import CandidateRestrictedEnvironment


def new_environment(cfg, seed, episode, arrival_steps):
    phase=((episode%5)+.5)*cfg['tide_period_s']/5
    episode_seed=int(np.random.SeedSequence([seed,901,episode]).generate_state(1)[0])
    env=CandidateRestrictedEnvironment(cfg,seed=episode_seed,phase_s=phase)
    env.arrival_stop_slot=arrival_steps
    return env


def train(out,method,resume=False,max_seconds=10800,component=None,stop_episode=None,lr=1,seed=87021):
    cfg,training=configuration(method,lr,seed)
    if component:
        training.update(component)
    training['updates']=training['episodes']
    training['episode_max_steps']=training['arrival_steps']+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3
    if available_ram_gib()<1.:
        raise RuntimeError('Less than 1GiB available RAM')
    out=Path(out)
    if (out/'STOP_AFTER_EPISODE').exists():
        raise RuntimeError('Explicit stop marker')
    if not resume:
        out.mkdir(parents=True,exist_ok=False)
    fd=os.open(out/'.worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode())
    try:
        return _train(out,method,cfg,training,resume,max_seconds,component,stop_episode,lr)
    finally:
        os.close(fd)
        (out/'.worker.lock').unlink()


def _train(out,method,cfg,training,resume,max_seconds,component,stop_episode,lr=1):
    seed_all(training['seed'])
    actual=runtime_info('cpu')
    runtime={k:actual[k] for k in ('python','numpy','scipy','torch','device','torch_threads')}
    identity=dict(schema='short100-natural-drain-v1',method=method,lr=lr,environment=cfg,
        training=training,runtime=runtime,source_hashes=source_hashes(),
        component_only=bool(component),transfer_integrated=True)
    identity_hash=digest(identity)
    if resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['identity_hash']!=identity_hash or (out/'complete.json').exists():
            raise ValueError('Changed source/config/runtime or already complete')
        manifest['execution_runtimes'].append(actual)
        json_write(out/'manifest.json',manifest)
    else:
        json_write(out/'manifest.json',dict(identity,identity_hash=identity_hash,
            execution_runtimes=[actual],source_commit=os.environ.get('GITHUB_SHA','local-component'),
            test_set_used=False,episode_definition='100 arrival slots plus full natural drainage'))
    learner=learner_for(method,training,cfg,'cpu')
    rows=[]
    completed=env_steps=0
    train_wall=0.
    environment=None
    if resume:
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['identity_hash']!=identity_hash:
            raise ValueError('Checkpoint identity mismatch')
        learner.load(saved['learner'])
        rng_load(saved['rng'])
        completed,env_steps,rows=saved['completed_episodes'],saved['env_steps'],saved['rows']
        train_wall=saved['training_wall_s']
        environment=saved['environment']
        if environment is not None and environment.tasks:
            raise ValueError('Resume requires a fully drained episode boundary')
        if (out/'episode_returns.csv').exists():
            archive=out/'recovery'/str(time.time_ns())
            archive.mkdir(parents=True)
            (archive/'episode_returns.csv').write_bytes((out/'episode_returns.csv').read_bytes())
        if rows:
            csv_write(out/'episode_returns.csv',rows)
    started=last_checkpoint=time.monotonic()
    checkpoint_episode=completed

    def checkpoint():
        checkpoint_write(out/'latest.pt',dict(identity_hash=identity_hash,learner=learner.state(),
            rng=rng_state(),environment=environment,completed_episodes=completed,
            update=completed,env_steps=env_steps,rows=rows,training_wall_s=train_wall))
        json_write(out/'checkpoint_status.json',dict(update=completed,env_steps=env_steps,
            completed_episodes=completed,total_arrival_slots=completed*training['arrival_steps'],
            total_drain_slots=env_steps-completed*training['arrival_steps'],
            training_wall_s=train_wall,component_only=bool(component),pending_tasks=0))

    try:
        if not resume:
            checkpoint()
        for episode in range(completed,training['episodes']):
            tick=time.monotonic()
            environment=new_environment(cfg,training['seed'],episode,training['arrival_steps'])
            learner.reset_history()
            learner.set_progress(episode+1,training['episodes'])
            total_return=0.
            for slot in range(training['episode_max_steps']):
                reward,records,_,_=environment.step(learner,arrivals=slot<training['arrival_steps'])
                if not math.isfinite(float(reward)):
                    raise FloatingPointError('Nonfinite reward')
                done=environment.slot>=training['arrival_steps'] and not environment.tasks
                learner.remember(environment.agent_rewards(0.),records,done)
                total_return+=float(reward)
                if done:
                    break
            else:
                raise AssertionError('Terminal cohort failed to drain within lifetime')
            if environment.arrivals!=environment.ended or environment.tasks:
                raise AssertionError('Task conservation or missing terminal costs')
            context,glob,*_=environment.observations()
            stats=learner.update(context,glob)
            if not all(math.isfinite(float(v)) for v in stats.values()):
                raise FloatingPointError('Nonfinite optimizer values')
            if any(not torch.isfinite(p).all() for module in (learner.actors,learner.critic)
                   for p in module.parameters()):
                raise FloatingPointError('Nonfinite parameters')
            expected=-environment.cost_total/(environment.u*cfg['arrival_per_user_s']*cfg['dt_s'])
            if not math.isclose(total_return,expected,rel_tol=1e-10,abs_tol=1e-8):
                raise AssertionError('Episode return lost physical cost including drainage')
            completed=episode+1
            env_steps+=environment.slot
            train_wall+=time.monotonic()-tick
            rows.append(dict(episode=completed,env_steps=env_steps,episode_return=total_return,
                arrival_steps=training['arrival_steps'],drain_steps=environment.slot-training['arrival_steps'],
                actual_steps=environment.slot,arrivals=environment.arrivals,ended=environment.ended,
                train_trace_hash=environment.arrival_hash,training_wall_s=train_wall))
            csv_write(out/'episode_returns.csv',rows)
            print('EPISODE',method,completed,'actual_steps',env_steps,flush=True)
            reason=('stop marker' if (out/'STOP_AFTER_EPISODE').exists() else
                'available RAM below .5GiB' if available_ram_gib()<.5 else
                'wall budget' if time.monotonic()-started>=max_seconds else
                'component boundary' if stop_episode and completed>=stop_episode else None)
            if (reason or completed==training['episodes'] or
                    completed-checkpoint_episode>=training['checkpoint_episodes'] or
                    time.monotonic()-last_checkpoint>=training['checkpoint_seconds']):
                checkpoint()
                checkpoint_episode,last_checkpoint=completed,time.monotonic()
            if reason and completed<training['episodes']:
                json_write(out/'paused.json',dict(update=completed,reason=reason))
                return False
        json_write(out/'complete.json',dict(updates=completed,episodes=completed,
            env_steps=env_steps,arrival_steps_per_episode=training['arrival_steps'],
            drainage_included=True,component_only=bool(component)))
        if (out/'paused.json').exists():
            (out/'paused.json').unlink()
        return True
    except BaseException as error:
        json_write(out/'failure.json',dict(message=str(error),traceback=traceback.format_exc(),
            completed_episodes=completed,recover_from='latest.pt'))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=('TMAPO','IPPO','PPO'),required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--lr',type=int,default=1)
    p.add_argument('--seed',type=int,default=87021)
    p.add_argument('--max-seconds',type=float,default=10800)
    a=p.parse_args()
    train(a.out,a.method,a.resume,a.max_seconds,lr=a.lr,seed=a.seed)


if __name__=='__main__':
    main()

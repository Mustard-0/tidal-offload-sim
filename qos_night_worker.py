"""Bounded original-task QoS sweep and two frozen-policy TMAPO sensitivities."""
import argparse
import bisect
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

from matched_pair import MODES, POPULATIONS, TRACES, ROOT, SOURCE, check, controller_environment
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import pack, seal, sha, unseal

CAMPAIGN='tmapo-qos-sensitivity-20261112'
BRANCH='experiment/'+CAMPAIGN
BASE='/repos/'+REPOSITORY
BASE_COMMIT='68f9e5082fc1fb1622f08031045a1701335a0810'
ARCHIVE=dict(campaign='nightreview-20261028-193947',run_id='40756312226',attempt='1',
    source_commit='ad141c66bde9bdd99fe627e4327dcb363377d7a7',
    plan_hash='c3e7517f1f1492427fd74c5246ea45d80fe334fbf3085447dd3edacc2e3ac1bf',
    source_hashes_hash='6d437e08251b9eaede9cb08a75b006285745dd14126126e3fc495e8bb7eacb0e')
METHODS=('TMAPO','IPPO','PPO','SD-SAC-Adapted','MARATD3-Adapted')
SEEDS=(87021,87022)
LEARNED=tuple(f'{m}_seed{s}' for m in METHODS for s in SEEDS)
CALIBRATED=.19420843653655595
FACTORS=tuple(sorted([i/100 for i in range(1,101)]+[CALIBRATED]))
VARIANTS={'epsilon05':(.5,20),'epsilon2':(2.,20),'wait0':(1.,0),'wait10':(1.,10)}
QOS_CELLS=('GeoI-Greedy',*LEARNED,*(f'NoPrivacy_{n}' for n in POPULATIONS))
SENS_CELLS=tuple(f'{v}_seed{s}' for v in VARIANTS for s in SEEDS)


class Sweep:
    """One pass: store counts by the first deadline factor a task meets."""
    def __init__(self):
        self.hist=[0]*(len(FACTORS)+1)
        self.arrivals=self.completed=0

    def add(self,failed,original,delay,deadline):
        assert deadline>0 and delay>=-1e-9 and math.isfinite(delay)
        self.arrivals+=1
        self.completed+=not failed
        if failed or original:
            self.hist[-1]+=1
            return
        i=bisect.bisect_left(FACTORS,(delay-1e-9)/deadline)
        while i>0 and delay<=FACTORS[i-1]*deadline+1e-9:i-=1
        while i<len(FACTORS) and delay>FACTORS[i]*deadline+1e-9:i+=1
        self.hist[i]+=1

    def values(self):
        count=self.arrivals;result=[]
        for v in self.hist[:-1]:
            count-=v
            result.append(count)
        return result

    def summary(self,ref):
        values=self.values()
        assert self.arrivals==int(ref['arrivals']) and self.completed==int(ref['completed'])
        assert values[-1]==round(float(ref['qos_rate'])*self.arrivals)
        if 'qos95_rate' in ref:
            assert values[FACTORS.index(.95)]==round(float(ref['qos95_rate'])*self.arrivals)
        return dict(ref,violations=values)


def archived_asset(api,release,assets,key,cell,suffix):
    name=cell+'.'+suffix
    expected=dict(ARCHIVE,cell=name)
    receipt=json.loads(download(assets[name+'.receipt.json']))
    assert all(receipt.get(k)==v for k,v in expected.items())
    asset=assets[name+'.enc']
    assert 0<asset['size']<600*1024*1024 and asset['digest'].startswith('sha256:')
    packet=download(asset)
    assert len(packet)==receipt['archive_size'] and sha(packet)==receipt['archive_sha256']
    raw,_=unseal(packet,key,expected)
    return raw


def raw_sweep(raw,refs):
    rows=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        assert len(names)==len(set(names))
        metrics={(int(r['users']),int(r['trace_seed'])):r for r in
                 csv.DictReader(io.StringIO(z.read('test_metrics.csv').decode()))}
        for ref in refs:
            n,t=int(ref['users']),int(ref['trace_seed'])
            metric=metrics[n,t]
            assert metric['full_input_hash']==ref['full_input_hash']
            assert int(metric['arrivals'])==int(ref['arrivals'])
            assert int(metric['completed'])==int(ref['completed'])
            prefix=f'cases/N{n}_t{t}/tasks/'
            chunks=sorted(k for k in names if k.startswith(prefix) and k.endswith('.csv.gz'))
            assert chunks
            sweep=Sweep();seen=set()
            for name in chunks:
                with z.open(name) as compressed, gzip.GzipFile(fileobj=compressed) as raw_csv:
                    with io.TextIOWrapper(raw_csv,encoding='utf-8',newline='') as f:
                        for r in csv.DictReader(f):
                            task=int(r['task_id'])
                            assert task not in seen
                            seen.add(task)
                            assert r['completed'] in ('0','1') and r['qos'] in ('0','1')
                            failed=r['completed']=='0';orig=r['qos']=='1'
                            delay=float(r['end_s'])-float(r['arrival_s']);deadline=float(r['deadline_s'])
                            assert orig==(failed or delay>deadline+1e-9)
                            sweep.add(failed,orig,delay,deadline)
            rows.append(sweep.summary(ref))
    return rows


def runtime():
    sys.path[:0]=[str(ROOT/'.deps'),str(SOURCE),str(SOURCE/'model')]
    from experiments.lr_screen.runtime import np,torch,seed_all
    from pressure9.environment import Environment,GeoIEnvironment
    from nightstudy.protocol import configuration
    from environment import GreedyPolicy
    return np,torch,seed_all,Environment,GeoIEnvironment,configuration,GreedyPolicy


def rollout(env,policy,torch):
    stop=30500;env.arrival_stop_slot=stop
    with torch.no_grad():
        for slot in range(stop+203):
            env.step(policy,arrivals=slot<stop)
            if env.slot>=stop and not env.tasks:break
        else:raise AssertionError('Natural drainage incomplete')
    assert env.arrivals==env.ended
    return dict(env.metrics(),users=env.u,trace_seed=env.seed,full_input_hash=env.full_input_hash,
        warmup_s=5.,measure_s=300.,phase_s=0.)


def greedy_sweep(ref):
    np,torch,_,_,geo,configuration,policy_type=runtime()
    cfg,_=configuration('Greedy',87021,ref['users'])
    base=controller_environment(geo,np)
    sweep=Sweep()
    class Recorded(base):
        def end(self,j,failed=False):
            measured=self.measured(j)
            super().end(j,failed)
            if measured:sweep.add(bool(failed),bool(j.violated),self.time-j.arrival,j.deadline)
    env=Recorded(cfg,geoi=False,seed=ref['trace_seed'],phase_s=0.,family='Greedy',
                 measure_start=5.,measure_end=305.,collect=False)
    row=rollout(env,policy_type(env.cfg),torch)
    row['method']=MODES[0]
    check(row,(ref['full_input_hash'],ref['arrivals']))
    for k in ('delay_s','energy_j_per_arrival','qos_rate','qos95_rate','mean_task_cost'):
        assert math.isclose(row[k],ref[k],rel_tol=1e-9,abs_tol=1e-12)
    return sweep.summary(ref)


def sensitivity(raw,seed,variant,refs,emit):
    np,torch,seed_all,base,_,configuration,_=runtime()
    from pressure9.policies import evaluation_policy
    from pressure9.runner import model_hash
    from privacy import amplification_threshold
    eps,cap=VARIANTS[variant]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert 'terminal_policy.pt' in z.namelist()
        saved=torch.load(io.BytesIO(z.read('terminal_policy.pt')),map_location='cpu',weights_only=False)
    assert saved['method']=='TMAPO' and saved['seed']==seed and not saved['component_only']
    assert saved['training']['updates']==1208

    class Sensitivity(base):
        def __init__(self,cfg,**kwargs):
            # The archived constructor validates its nominal contract. This
            # explicit evaluation adapter varies only the requested setting.
            super().__init__(dict(cfg),**kwargs)
            self.cfg.update(epsilon_req=eps,wait_max_s=cap/1000 if cap else cfg['wait_max_s'])
            self.min_batch=self.group_target=amplification_threshold(self.cfg)
        def wait_information(self,tasks):
            info=super().wait_information(tasks)
            if not cap:info=dict(info,allow=False,horizon_s=0.,reason='disabled_for_sensitivity')
            return info

    for ref in refs:
        trace=ref['trace_seed']
        seed_all(int(np.random.SeedSequence([trace,seed,2803]).generate_state(1)[0]))
        cfg,t=configuration('TMAPO',seed,100,variant='lr_horizon')
        env=Sensitivity(cfg,seed=trace,phase_s=0.,family='TMAPO',measure_start=5.,measure_end=305.,collect=False)
        policy=evaluation_policy('TMAPO',t,env.cfg,saved['policy'])
        policy.reset_history()
        before=model_hash(policy)
        row=rollout(env,policy,torch)
        assert model_hash(policy)==before and env.max_observed_wait<=cap/1000+1e-9
        assert row['full_input_hash']==ref['full_input_hash'] and row['arrivals']==ref['arrivals']
        assert row['number_queued_pending']==row['class_staging_left']==0
        emit(dict(row,method='TMAPO',training_seed=seed,variant=variant,epsilon=eps,
            wait_cap_ms=cap,model_identity=saved['identity_hash'],training=False))


def assets_for(api,release):
    result={}
    for page in range(1,6):
        rows=api.request('GET',BASE+f'/releases/{release["id"]}/assets?per_page=100&page={page}')
        result.update((a['name'],a) for a in rows)
        if len(rows)<100:return result
    raise ValueError('Unexpected archive asset count')


def run(cell):
    assert sys.platform=='linux' and sys.version_info[:3]==(3,9,25)
    assert os.environ['GITHUB_REPOSITORY']==REPOSITORY and os.environ['GITHUB_REF']=='refs/heads/'+BRANCH
    assert os.environ['GITHUB_RUN_ATTEMPT']=='1' and cell in QOS_CELLS+SENS_CELLS
    commit=os.environ['GITHUB_SHA'];run_id=os.environ['GITHUB_RUN_ID']
    api=GitHub(os.environ['GH_TOKEN']);key=bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    release=api.request('GET',BASE+'/releases/tags/'+CAMPAIGN)
    assert release['target_commitish']==commit
    assets=assets_for(api,release)
    packet=download(assets['inputs.enc'])
    raw,_=unseal(packet,key,dict(campaign=CAMPAIGN,source_commit=commit,purpose='night-plan-inputs'))
    plan=json.loads(raw)
    assert plan['factors']==list(FACTORS)
    name=cell+'.enc'
    assert name not in assets
    header=dict(campaign=CAMPAIGN,source_commit=commit,run_id=run_id,attempt='1',
        cell=cell,inputs_sha256=sha(packet),purpose='night-small-outcome')
    rows=[];error=None
    try:
        if cell.startswith('NoPrivacy_'):
            for ref in plan['reference'][cell]:
                rows.append(greedy_sweep(ref))
        else:
            old=api.request('GET',BASE+'/releases/tags/'+ARCHIVE['campaign'])
            assert old['target_commitish']==ARCHIVE['source_commit']
            source_assets=assets_for(api,old)
            if cell in SENS_CELLS:
                variant,seed_text=cell.split('_seed');seed=int(seed_text)
                weights=archived_asset(api,old,source_assets,key,f'TMAPO_seed{seed}','weights')
                refs=[r for r in plan['reference'][f'TMAPO_seed{seed}'] if r['users']==100]
                assert len(refs)==2
                sensitivity(weights,seed,variant,refs,rows.append)
            else:
                data=archived_asset(api,old,source_assets,key,cell,'evaluation-raw')
                rows=raw_sweep(data,plan['reference'][cell])
    except Exception as e:
        error=dict(type=type(e).__name__,message=str(e))
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        (folder/'outcome.json').write_text(json.dumps(dict(completed=error is None,
            error=error,rows=rows),allow_nan=False),encoding='utf-8')
        payload=seal(pack(folder),key,header)
    asset=api.upload(release['id'],name,payload)
    assert asset['size']==len(payload) and asset['digest']=='sha256:'+sha(payload)
    print('Encrypted outcome saved',cell,len(rows),'rows',flush=True)
    return error is None


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('cell',choices=QOS_CELLS+SENS_CELLS)
    raise SystemExit(0 if run(p.parse_args().cell) else 2)

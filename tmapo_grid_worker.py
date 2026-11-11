"""Reproduce the nominal policy before extending the epsilon x shrink grid."""
import argparse
import io
import itertools
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

from qos_night_worker import ARCHIVE, BASE, SEEDS, archived_asset, assets_for, rollout, runtime
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import pack, seal, sha, unseal

CAMPAIGN='tmapo-privacy-grid-20261112'
BRANCH='experiment/'+CAMPAIGN
BASE_COMMIT='597b4f872b399caa55ebd3a01afac535dd4c9f2b'
EPS=(.5,1.,2.)
SHRINK=(.25,.5,.75)
VARIANTS={'nominal':(1.,.5)}
VARIANTS.update({f'e{i}_s{j}':(e,s) for i,e in enumerate(EPS)
                 for j,s in enumerate(SHRINK) if s!=.5})
GATES=tuple(f'nominal_seed{s}' for s in SEEDS)
CELLS=tuple(f'{v}_seed{s}' for v,s in itertools.product(VARIANTS,SEEDS) if v!='nominal')
PARITY=('delay_s','energy_j_per_arrival','qos_rate','mean_task_cost','ql_m',
        'published','completed','gate_allowed','class_gate_wait_actions')


def evaluate(raw,seed,variant,refs,emit):
    np,torch,seed_all,base,_,configuration,_=runtime()
    from pressure9.policies import evaluation_policy
    from pressure9.runner import model_hash
    from privacy import amplification_threshold
    eps,shrink=VARIANTS[variant]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        saved=torch.load(io.BytesIO(z.read('terminal_policy.pt')),map_location='cpu',weights_only=False)
    assert saved['method']=='TMAPO' and saved['seed']==seed and not saved['component_only']
    assert saved['training']['updates']==1208
    torch.set_num_threads(1)

    class Sensitivity(base):
        def __init__(self,cfg,**kwargs):
            super().__init__(dict(cfg),**kwargs)
            self.cfg.update(epsilon_req=eps,report_shrink=shrink)
            assert {k for k in cfg if self.cfg[k]!=cfg[k]}<={'epsilon_req','report_shrink'}
            self.min_batch=self.group_target=amplification_threshold(self.cfg)
            self.max_report_radius=0.

        def release(self,tasks,reason='adaptive_region_class'):
            super().release(tasks,reason)
            for j in tasks:
                center=self.cfg['region_centers_m'][j.region]
                radius=math.hypot(float(j.report[0])-center[0],float(j.report[1])-center[1])
                assert radius<=self.cfg['region_radius_m']*shrink+1e-7
                self.max_report_radius=max(self.max_report_radius,radius)

    for ref in refs:
        trace=ref['trace_seed']
        seed_all(int(np.random.SeedSequence([trace,seed,2803]).generate_state(1)[0]))
        cfg,t=configuration('TMAPO',seed,100,variant='lr_horizon')
        env=Sensitivity(cfg,seed=trace,phase_s=0.,family='TMAPO',measure_start=5.,measure_end=305.,collect=False)
        policy=evaluation_policy('TMAPO',t,env.cfg,saved['policy'])
        policy.reset_history()
        before=model_hash(policy)
        row=rollout(env,policy,torch)
        assert model_hash(policy)==before and env.max_observed_wait<=.02+1e-9
        assert row['arrivals']==ref['arrivals'] and row['full_input_hash']==ref['full_input_hash']
        assert row['number_queued_pending']==row['class_staging_left']==0
        if variant=='nominal':
            for k in PARITY:
                assert math.isclose(row[k],ref[k],rel_tol=1e-9,abs_tol=1e-12),f'Nominal parity failed: {k}'
        emit(dict(row,method='TMAPO',training_seed=seed,variant=variant,epsilon=eps,shrink=shrink,
            model_identity=saved['identity_hash'],model_hash=before,training=False,
            max_report_radius_m=env.max_report_radius,nominal_parity_checked=variant=='nominal'))


def run(cell):
    assert sys.platform=='linux' and sys.version_info[:3]==(3,9,25)
    assert os.environ['GITHUB_REPOSITORY']==REPOSITORY and os.environ['GITHUB_REF']=='refs/heads/'+BRANCH
    assert os.environ['GITHUB_RUN_ATTEMPT']=='1' and cell in GATES+CELLS
    api=GitHub(os.environ['GH_TOKEN']);key=bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    commit=os.environ['GITHUB_SHA'];run_id=os.environ['GITHUB_RUN_ID']
    release=api.request('GET',BASE+'/releases/tags/'+CAMPAIGN)
    assert release['target_commitish']==commit
    assets=assets_for(api,release);packet=download(assets['inputs.enc'])
    raw,_=unseal(packet,key,dict(campaign=CAMPAIGN,source_commit=commit,purpose='grid-plan-inputs'))
    plan=json.loads(raw)
    assert plan['variants']=={k:list(v) for k,v in VARIANTS.items()}
    name=cell+'.enc'
    assert name not in assets
    header=dict(campaign=CAMPAIGN,source_commit=commit,run_id=run_id,attempt='1',
        cell=cell,inputs_sha256=sha(packet),purpose='grid-small-outcome')
    variant,s=cell.split('_seed');seed=int(s)
    rows=[];error=None
    try:
        old=api.request('GET',BASE+'/releases/tags/'+ARCHIVE['campaign'])
        assert old['target_commitish']==ARCHIVE['source_commit']
        weights=archived_asset(api,old,assets_for(api,old),key,f'TMAPO_seed{seed}','weights')
        evaluate(weights,seed,variant,plan['reference'][str(seed)],rows.append)
        assert len(rows)==2
    except Exception as e:
        error=dict(type=type(e).__name__,message=str(e))
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        (folder/'outcome.json').write_text(json.dumps(dict(completed=error is None,error=error,rows=rows),
            allow_nan=False),encoding='utf-8')
        payload=seal(pack(folder),key,header)
    asset=api.upload(release['id'],name,payload)
    assert asset['size']==len(payload) and asset['digest']=='sha256:'+sha(payload)
    print('Encrypted outcome saved',cell,len(rows),'rows',flush=True)
    return error is None


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('cell',choices=GATES+CELLS)
    raise SystemExit(0 if run(p.parse_args().cell) else 2)

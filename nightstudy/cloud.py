"""One finite screen and at most one validation-qualified formal campaign."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
from . import protocol as P
from pressure9 import cloud as B

BASE='/repos/'+B.REPOSITORY
ORIGINAL_CHILD=B.run_child


def campaign():
    value=os.environ['MODERATE_CAMPAIGN']
    if not re.fullmatch(r'nightreview-[a-z0-9-]{1,60}',value):raise ValueError('Unexpected finite campaign')
    assert os.environ['GITHUB_RUN_ATTEMPT']=='1'
    assert len(bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']))==32
    return value


def header(cell):
    return dict(campaign=campaign(),cell=cell,run_id=os.environ['GITHUB_RUN_ID'],attempt='1',
        source_commit=os.environ['GITHUB_SHA'],plan_hash=P.digest(P.plan()),source_hashes_hash=P.digest(P.source_hashes()))


B.campaign=campaign;B.header=header
B.plan=P.plan
ORIGINAL_CHECK_GATE=B.check_gate


def check_gate():
    ORIGINAL_CHECK_GATE()
    run=B.get('/actions/runs/'+os.environ['GITHUB_RUN_ID'])
    start=datetime.datetime.fromisoformat(run['created_at'].replace('Z','+00:00'))
    if (datetime.datetime.now(datetime.timezone.utc)-start).total_seconds()>24*3600:
        raise RuntimeError('24h campaign entry budget exhausted; no new phase')


B.check_gate=check_gate


def selection():
    folder=P.ROOT/'runs/selection'
    if not folder.exists():B.receive('selection',folder)
    value=json.loads((folder/'selection.json').read_text())
    assert value['allowed'] and value['plan_hash']==P.digest(P.plan())
    return value


def child(command,out,cell,stage,before=0):
    command=list(command)
    if command[3] in ('pressure9.runner','pressure9.evaluate'):
        method=command[command.index('--method')+1]
        variant=selection()['selected_variant'] if method=='TMAPO' else 'control'
        mode='train' if command[3].endswith('runner') else 'evaluate'
        command[3]='nightstudy.worker';command.insert(4,mode)
        command+=['--variant',variant]
    return ORIGINAL_CHILD(command,out,cell,stage,before)


B.run_child=child


def gate():
    campaign()
    assert sys.platform=='linux' and sys.version_info[:3]==(3,9,25)
    assert not B.get('')['private']
    B.api().request('POST',BASE+'/releases',dict(tag_name=campaign(),target_commitish=os.environ['GITHUB_SHA'],
        name=campaign(),prerelease=True,draft=False,body='Finite validation-only hyperparameter review. Pure code and encrypted synthetic outputs; no rank guarantee.'))
    out=P.ROOT/'runs/gate';small=P.ROOT/'runs/gate-small';small.mkdir(parents=True)
    try:
        from pressure9 import preflight
        preflight.source_hashes=P.source_hashes
        proof=preflight.run(out)
        from experiments.lr_screen.vendor.accountant import calibrate
        table={n:calibrate(n,1.,1e-6,.01,8.) for n in range(1,51)}
        assert all(table[n]==1. for n in range(1,30)) and table[30]>1.
        # All screened changes are explicit optimization settings, not physics.
        reference=P.configuration('TMAPO')[0]
        assert len(P.SCREEN)==12 and len(P.FORMAL)==10
        for spec in P.SCREEN:
            cfg,t=P.configuration(spec['method'],spec['seed'],variant=spec['variant'],screen=True)
            assert cfg==reference and t['updates']*t['rollout_steps']==30200
        proof.update(plan_hash=P.digest(P.plan()),source_hashes_hash=P.digest(P.source_hashes()),
            finite_screen_cells=12,conditional_formal_cells=10,unchanged_common_environment=True,
            accountant_eps0_table=table,first_amplification_batch=30)
        (out/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
        for name in ('verification.json','tests.log'):shutil.copyfile(out/name,small/name)
        B.encrypted(small,'preflight-private')
        B.upload('preflight.json',json.dumps(dict(header('preflight'),passed=True,tests_run=proof['tests'],
            real_updates_and_exact_resume=True)).encode())
    except BaseException as error:
        (small/'error.json').write_text(json.dumps(dict(error=repr(error))),encoding='utf-8')
        if (out/'tests.log').exists():shutil.copyfile(out/'tests.log',small/'tests.log')
        B.encrypted(small,'preflight-private');raise


def screen(cell):
    B.check_gate()
    spec=next(s for s in P.SCREEN if s['id']==cell)
    out=P.ROOT/'runs/screens'/cell;out.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','nightstudy.worker','train','--out',str(out),
        '--method',spec['method'],'--seed',str(spec['seed']),'--variant',spec['variant'],
        '--screen','--maximum-seconds','5400']
    code=ORIGINAL_CHILD(command,out,cell,'train_screen')
    state='completed' if code==0 and (out/'complete.json').exists() else (
        'paused' if code==2 and (out/'paused.json').exists() else 'failed')
    if state=='failed':
        B.upload(cell+'.halt.json',json.dumps(dict(header(cell+'.screen'),state=state,child_exited=True,exit_code=code)).encode())
    if out.exists():B.encrypted(out,cell+'.screen-raw')
    small=P.ROOT/'runs/screen-small'/cell;small.mkdir(parents=True,exist_ok=False)
    for name in ('complete.json','manifest.json','training.csv','cycles.csv','paused.json','failure.json'):
        if (out/name).exists():shutil.copyfile(out/name,small/name)
    for file in out.glob('validation_*.json'):shutil.copyfile(file,small/file.name)
    (small/'screen_spec.json').write_text(json.dumps(spec),encoding='utf-8')
    B.encrypted(small,cell+'.screen-summary')
    B.upload(cell+'.screen.outcome.json',json.dumps(dict(header(cell+'.screen'),state=state,child_exited=True,exit_code=code)).encode())
    if state=='failed':raise RuntimeError('Failed diagnostic screen, no retry')


def select():
    records=[];complete=True
    assets={a['name']:a for a in B.release()['assets']}
    for spec in P.SCREEN:
        name=spec['id']+'.screen.outcome.json'
        if name not in assets:complete=False;continue
        outcome=json.loads(B.download(assets[name]))
        assert all(outcome[k]==v for k,v in header(spec['id']+'.screen').items())
        if outcome['state']!='completed':complete=False;continue
        folder=P.ROOT/'runs/screen-received'/spec['id']
        B.receive(spec['id']+'.screen-summary',folder)
        done=json.loads((folder/'complete.json').read_text());assert done['env_steps']==30200 and done['update']==302 and done['pending_tasks']==0
        if spec['method']=='TMAPO':
            values={u:json.loads((folder/f'validation_{u:04d}.json').read_text()) for u in (0,151,302)}
            records.append(dict(**spec,initial_cost=values[0]['mean_task_cost'],
                middle_cost=values[151]['mean_task_cost'],cost=values[302]['mean_task_cost'],qos=values[302]['qos_rate']))
    from .selection import choose
    scores,best=choose(records,complete)
    value=dict(allowed=bool(best),selected_variant=best['variant'] if best else None,
        all_screens_complete=complete,records=records,scores=scores,plan_hash=P.digest(P.plan()),
        reason='predeclared independent-validation gate passed' if best else 'finite screens did not establish qualified validation improvement; no automatic formal training',
        final_test_used=False,no_more_rounds=True)
    folder=P.ROOT/'runs/selection';folder.mkdir(parents=True,exist_ok=False)
    (folder/'selection.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    B.encrypted(folder,'selection')
    B.upload('selection.completed.json',json.dumps(dict(header('selection'),completed=True,formal_allowed=bool(best))).encode())
    with open(os.environ['GITHUB_OUTPUT'],'a',encoding='utf-8') as f:f.write('allowed='+str(bool(best)).lower()+'\n')


def collect():
    selection()
    import pressure9.report as report
    report.plan=P.plan
    B.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gate','screen','select','train','evaluate','collect'))
    p.add_argument('--cell');p.add_argument('--segment',type=int,choices=(1,2));a=p.parse_args()
    if a.mode=='gate':gate()
    elif a.mode=='screen':screen(a.cell)
    elif a.mode=='select':select()
    elif a.mode=='train':selection();B.train(a.cell,a.segment)
    elif a.mode=='evaluate':selection();B.evaluate(a.cell)
    else:collect()

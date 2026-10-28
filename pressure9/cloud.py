"""Finite cloud-only owner: two bounded train segments then terminal inference."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from .protocol import ROOT,plan,digest,source_hashes
from experiments.lr_screen.github_api import GitHub,REPOSITORY,download
from experiments.lr_screen.transport import seal,unseal,pack,unpack,sha

BASE='/repos/'+REPOSITORY


def campaign():
    tag=os.environ['MODERATE_CAMPAIGN']
    if not re.fullmatch(r'pressure9-nine-[a-z0-9-]{1,60}',tag):
        raise ValueError('Unexpected finite campaign')
    assert os.environ['GITHUB_RUN_ATTEMPT']=='1'
    assert len(bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']))==32
    return tag


def header(cell):
    return dict(campaign=campaign(),cell=cell,run_id=os.environ['GITHUB_RUN_ID'],attempt='1',
        source_commit=os.environ['GITHUB_SHA'],plan_hash=digest(plan()),source_hashes_hash=digest(source_hashes()))


def api():
    return GitHub(os.environ['GH_TOKEN'])


def get(path):
    for attempt in range(3):
        try:
            return api().request('GET',BASE+path)
        except Exception:
            if attempt==2:
                raise
            time.sleep(3*(attempt+1))


def release():
    return get('/releases/tags/'+campaign())


def upload(name,data):
    # Writes are never blindly retried. An uncertain write is accepted ONLY if
    # the exact immutable name/digest/size is discoverable by read-only lookup.
    try:
        return api().upload(release()['id'],name,data)
    except Exception:
        matches=[a for a in release()['assets'] if a['name']==name]
        if len(matches)==1 and matches[0].get('digest')=='sha256:'+sha(data) and matches[0]['size']==len(data):
            return matches[0]
        raise


def encrypted(folder,cell):
    data=seal(pack(folder),bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']),header(cell))
    upload(cell+'.enc',data)
    upload(cell+'.receipt.json',json.dumps(dict(header(cell),archive_name=cell+'.enc',
        archive_sha256=sha(data),archive_size=len(data))).encode())


def receive(cell,out):
    assets={a['name']:a for a in release()['assets']}
    receipt=json.loads(download(assets[cell+'.receipt.json']))
    assert all(receipt[k]==v for k,v in header(cell).items())
    packet=download(assets[cell+'.enc'])
    assert sha(packet)==receipt['archive_sha256'] and len(packet)==receipt['archive_size']
    raw,meta=unseal(packet,bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']),header(cell))
    unpack(raw,out)


def gate():
    campaign()
    assert sys.platform=='linux' and sys.version_info[:3]==(3,9,25)
    assert not get('')['private'],'Unknown-cost private Actions forbidden'
    api().request('POST',BASE+'/releases',dict(tag_name=campaign(),target_commitish=os.environ['GITHUB_SHA'],
        name=campaign(),prerelease=True,draft=False,
        body='Finite common rate9 study. Pure code, encrypted synthetic outputs only. No superiority guarantee.'))
    from .preflight import run
    out=ROOT/'runs/preflight'
    summary=ROOT/'runs/preflight-delivery';summary.mkdir(parents=True,exist_ok=False)
    try:
        proof=run(out)
    except BaseException as error:
        (summary/'error.json').write_text(json.dumps(dict(error=repr(error))),encoding='utf-8')
        if (out/'tests.log').exists():
            shutil.copyfile(out/'tests.log',summary/'tests.log')
        if out.exists():
            encrypted(summary,'preflight-private')
        raise
    for name in ('verification.json','tests.log'):
        shutil.copyfile(out/name,summary/name)
    encrypted(summary,'preflight-private')
    upload('preflight.json',json.dumps(dict(header('preflight'),passed=True,tests_run=proof['tests'],
        all_5_real_actor_critic_updates=True,all_5_exact_midcycle_resume=True,
        paired_full_exogenous_inputs=True)).encode())
    print('Five-method source/update/exact-resume gate passed',flush=True)


def check_gate():
    assets=release()['assets']
    if any(a['name'].endswith('.halt.json') for a in assets):
        raise RuntimeError('Another finite cell reported a failure; stop expanding the queue')
    asset=next(a for a in assets if a['name']=='preflight.json')
    proof=json.loads(download(asset))
    assert proof['passed'] and all(proof[k]==v for k,v in header('preflight').items())


def run_child(command,out,cell,stage,before=0):
    log=ROOT/'runs/logs'/f'{cell}.{stage}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        started=False
        while child.poll() is None:
            status=out/'status.json'
            if stage.startswith('train') and not started and status.exists():
                try:
                    value=json.loads(status.read_text())
                except (ValueError,OSError):
                    time.sleep(2.);continue
                if value['update']>=10 and value['update']>before and value['actor_updated']:
                    upload(cell+'.'+stage+'.started.json',json.dumps(dict(header(cell+'.'+stage),
                        process_started=True,real_updates=value['update'],environment_steps=value['env_steps'],
                        actor_updated=True,preflight_passed=True,cpu_count=os.cpu_count())).encode())
                    started=True
            time.sleep(2.)
        code=child.wait()
    if out.exists():
        shutil.copyfile(log,out/(stage+'.worker.log'))
    return code


def train(cell,segment):
    check_gate()
    spec=next(c for c in plan()['cells'] if c['id']==cell)
    stage=f'train{segment}'
    out=ROOT/'runs/train'/cell
    before=0
    if segment==2:
        asset=next(a for a in release()['assets'] if a['name']==cell+'.train1.outcome.json')
        previous=json.loads(download(asset))
        assert all(previous[k]==v for k,v in header(cell+'.train1').items())
        assert previous['child_exited'] and previous['state'] in ('completed','paused')
        if previous['state']=='completed':
            upload(cell+'.train2.outcome.json',json.dumps(dict(header(cell+'.train2'),state='completed',
                no_new_training=True,archive_cell=cell+'.train1',weights_cell=cell+'.weights',
                provenance='same completed segment1, not a new training execution')).encode())
            return
        if not out.exists():
            receive(cell+'.train1',out)
        before=json.loads((out/'status.json').read_text())['update']
        assert not (out/'failure.json').exists() and not (out/'.worker.lock').exists()
    out.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','pressure9.runner','--out',str(out),
             '--method',spec['method'],'--seed',str(spec['seed']),'--maximum-seconds','9000']
    if segment==2:
        command.append('--resume')
    code=run_child(command,out,cell,stage,before)
    state='completed' if code==0 and (out/'complete.json').exists() else (
          'paused' if code==2 and (out/'paused.json').exists() and not (out/'failure.json').exists() else 'failed')
    if state=='failed':
        upload(cell+'.halt.json',json.dumps(dict(header(cell+'.'+stage),state='failed',child_exited=True,exit_code=code)).encode())
    if out.exists():
        encrypted(out,cell+'.'+stage)
    if state=='completed':
        weights=ROOT/'runs/weights'/cell;weights.mkdir(parents=True,exist_ok=False)
        for name in ('terminal_policy.pt','manifest.json','complete.json','training.csv','cycles.csv'):
            shutil.copyfile(out/name,weights/name)
        for file in out.glob('validation_*.json'):
            shutil.copyfile(file,weights/file.name)
        encrypted(weights,cell+'.weights')
    upload(cell+'.'+stage+'.outcome.json',json.dumps(dict(header(cell+'.'+stage),state=state,
        child_exited=True,exit_code=code,archive_cell=cell+'.'+stage,
        weights_cell=cell+'.weights' if state=='completed' else None)).encode())
    if state=='failed':
        raise RuntimeError('Scientific/unknown training failure; no automatic retry or further phase')
    print('Bounded training segment saved: '+state,flush=True)


def evaluate(cell):
    check_gate()
    rules=cell in ('Greedy','GeoI-Greedy')
    if rules:
        method,seed=cell,87021;weights=None
    else:
        spec=next(c for c in plan()['cells'] if c['id']==cell)
        method,seed=spec['method'],spec['seed']
        asset=next(a for a in release()['assets'] if a['name']==cell+'.train2.outcome.json')
        outcome=json.loads(download(asset))
        assert outcome['state']=='completed'
        folder=ROOT/'runs/terminal'/cell
        receive(cell+'.weights',folder)
        weights=folder/'terminal_policy.pt'
    out=ROOT/'runs/evaluation'/cell
    command=[sys.executable,'-u','-m','pressure9.evaluate','--out',str(out),'--method',method,'--seed',str(seed)]
    if weights:
        command+=['--weights',str(weights)]
    code=run_child(command,out,cell,'evaluation')
    if out.exists():
        encrypted(out,cell+'.evaluation-raw')
    summary=ROOT/'runs/evaluation-summary'/cell;summary.mkdir(parents=True,exist_ok=False)
    for name in ('test_metrics.csv','complete.json','incomplete.json'):
        if (out/name).exists():
            shutil.copyfile(out/name,summary/name)
    if not rules:
        for name in ('training.csv','cycles.csv','manifest.json'):
            source=ROOT/'runs/terminal'/cell/name
            shutil.copyfile(source,summary/name)
        for file in (ROOT/'runs/terminal'/cell).glob('validation_*.json'):
            shutil.copyfile(file,summary/file.name)
    encrypted(summary,cell+'.evaluation-summary')
    state='completed' if code==0 and (out/'complete.json').exists() else 'incomplete'
    upload(cell+'.evaluation.outcome.json',json.dumps(dict(header(cell+'.evaluation'),state=state,
        child_exited=True,exit_code=code)).encode())
    if state!='completed':
        raise RuntimeError('Finite evaluation incomplete; evidence retained')


def collect():
    folders=[]
    for cell in [c['id'] for c in plan()['cells']]+['Greedy','GeoI-Greedy']:
        folder=ROOT/'runs/received'/cell
        receive(cell+'.evaluation-summary',folder)
        assert (folder/'complete.json').exists()
        folders.append(folder)
    out=ROOT/'runs/comparison';out.mkdir(parents=True,exist_ok=False)
    from .report import report
    report(folders,out)
    encrypted(out,'comparison')
    upload('campaign.completed.json',json.dumps(dict(header('comparison'),completed=True,
        training_cells=10,environment_steps_per_cell=120800,evaluation_populations=plan()['test']['populations'])).encode())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gate','train','evaluate','collect'))
    p.add_argument('--cell');p.add_argument('--segment',type=int,choices=(1,2))
    a=p.parse_args()
    if a.mode=='gate':
        gate()
    elif a.mode=='train':
        train(a.cell,a.segment)
    elif a.mode=='evaluate':
        evaluate(a.cell)
    else:
        collect()

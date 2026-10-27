"""User-authorized ciphertext transport; bounded standard public CPU execution."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from .protocol import ROOT,plan,digest,source_hashes,ABLATIONS
from ops.github_api import GitHub,REPOSITORY,download


def request_from_tag():
    tag=os.environ['GITHUB_REF_NAME']
    if not tag.startswith(('highload-run-','highload-probe-')) or os.environ['GITHUB_RUN_ATTEMPT']!='1':
        raise ValueError('Unknown trigger or prohibited rerun')
    req=json.loads(subprocess.check_output(['git','for-each-ref','refs/tags/'+tag,'--format=%(contents)'],text=True))
    if req['plan_hash']!=digest(plan()) or not 1<=len(req['entries'])<=16:
        raise ValueError('Wrong frozen plan')
    if len({e['cell'] for e in req['entries']})!=len(req['entries']):raise ValueError('Duplicate job')
    allowed={c['id'] for c in plan()['cells']+plan()['conditional_ablation_cells']}
    for e in req['entries']:
        if not 1<=e['segment']<=96 or e['kind'] not in ('train','evaluation'):
            raise ValueError('Unknown job or segment cap')
        if e['kind']=='train' and (e['cell'] not in allowed or e['cell']!=f"{e['method']}_LR{e['lr']}_seed{e['seed']}"):
            raise ValueError('Unknown train cell')
        if e['kind']=='evaluation' and not e['cell'].startswith('eval--'):raise ValueError('Unknown evaluation job')
        if e['kind']=='train' and e['method'] in ABLATIONS and e['lr']!=req.get('selected_TMAPO_lr'):
            raise ValueError('Ablation must use full-model selected LR')
        if e['segment']>1 and not e.get('input'):raise ValueError('Missing continuation parent')
    return req


def run(entry,req):
    from ops.transport import pack,unpack,seal,unseal,sha
    from experiments.lr_screen.runtime import torch,runtime_info
    torch.set_num_threads(1)
    contract=json.loads((ROOT/'cloud/contract.json').read_text())
    if source_hashes()!=contract['source_hashes'] or digest(plan())!=contract['plan_hash']:
        raise ValueError('Frozen scientific code mismatch')
    actual=runtime_info('cpu')
    if sys.platform!='linux' or sys.version_info[:3]!=(3,9,25):
        raise ValueError('Only reviewed Ubuntu CPU/Python runtime')
    for k,v in contract['runtime'].items():
        if actual[k]!=v:raise ValueError('Wrong CPU runtime '+k)
    api=GitHub(os.environ['GH_TOKEN'])
    key=bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    out=ROOT/'runs/cloud'/entry['cell']
    parent=entry.get('input')
    prior_csv=None;prior_status=None
    if parent:
        assets=[]
        for page in range(1,1000):
            part=api.request('GET','/repos/'+REPOSITORY+'/releases/'+str(req['release_id'])+'/assets?per_page=100&page='+str(page))
            assets+=part
            if len(part)<100:break
        asset=next(a for a in assets if a['name']==parent['name'])
        packet=download(asset)
        if sha(packet)!=parent['sha256']:raise ValueError('Parent digest mismatch')
        raw,_=unseal(packet,key,dict(campaign=req['campaign'],cell=entry['cell'],direction='input',segment=entry['segment']))
        unpack(raw,out)
        if (out/'.worker.lock').exists() or (out/'failure.json').exists():raise ValueError('Unclean/failed parent')
        for name,value in parent.get('prefix_hashes',{}).items():
            if hashlib.sha256((out/name).read_bytes()).hexdigest()!=value:raise ValueError('Input prefix changed')
        if entry['kind']=='train':
            prior_status=json.loads((out/'checkpoint_status.json').read_text())
            prior_csv=(out/'training.csv').read_bytes() if (out/'training.csv').exists() else None
    shared_preflight=os.environ.get('HIGHLOAD_SHARED_PREFLIGHT_SHA')==os.environ['GITHUB_SHA']
    if (entry['segment']==1 or entry.get('local_handoff')) and not shared_preflight:
        subprocess.run([sys.executable,'-m','unittest','experiments.highload.test_protocol','-v'],cwd=ROOT,check=True,timeout=600)
    receipt=dict(campaign=req['campaign'],cell=entry['cell'],kind=entry['kind'],method=entry['method'],
        lr=entry['lr'],seed=entry['seed'],segment=entry['segment'],run_id=os.environ['GITHUB_RUN_ID'],
        attempt='1',source_commit=os.environ['GITHUB_SHA'],plan_hash=digest(plan()),preflight_passed=True)
    command=[sys.executable,'-u','-m',
        'experiments.highload.runner' if entry['kind']=='train' else 'experiments.highload.evaluate_entry',
        '--out',str(out),'--method',entry['method'],'--lr',str(entry['lr']),'--seed',str(entry['seed'])]
    if entry['kind']=='train':
        command+=['--max-seconds','10800']
        if parent:command.append('--resume')
    started=time.monotonic();log=ROOT/'runs/worker_logs'/(entry['cell']+f".segment{entry['segment']:03d}.log")
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as f:
        child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+900
        marker=out/'checkpoint_status.json' if entry['kind']=='train' else out/'evaluation/start.json'
        acknowledged=False
        while child.poll() is None and time.monotonic()<deadline:
            if marker.exists():
                try:
                    status=json.loads(marker.read_text())
                    acknowledged=(entry['kind']=='evaluation' or prior_status is None or
                        (status['identity_hash']==prior_status['identity_hash'] and status['update']>prior_status['update']))
                except (OSError,ValueError,KeyError):acknowledged=False
            if acknowledged:break
            time.sleep(.5)
        if acknowledged and child.poll() is None:
            acknowledgment=dict(receipt,process_started=True,pid=child.pid,utc_epoch=time.time())
            if entry['kind']=='train':
                cp=json.loads(marker.read_text());acknowledgment.update(update=cp['update'],env_steps=cp['env_steps'],
                    adopted_input=bool(parent),identity_hash=cp['identity_hash'])
            try:api.upload(req['release_id'],entry['cell']+f".segment{entry['segment']:03d}.started.json",json.dumps(acknowledgment).encode())
            except Exception as error:print('Startup visibility unavailable:',type(error).__name__,flush=True)
        code=child.wait(timeout=18000)
    out.mkdir(parents=True,exist_ok=True);shutil.copy2(log,out/'cloud_worker.log')
    target=out if entry['kind']=='train' else out/'evaluation'
    done=json.loads((target/'complete.json').read_text()) if (target/'complete.json').exists() else None
    paused=json.loads((target/'paused.json').read_text()) if (target/'paused.json').exists() else None
    state='failed' if code or (target/'failure.json').exists() else 'completed' if done else 'paused' if paused else 'unknown'
    prefix_ok=(prior_csv is None or ((out/'training.csv').exists() and (out/'training.csv').read_bytes().startswith(prior_csv)))
    if not prefix_ok:
        state='failed'
        (out/'prefix_failure.json').write_text(json.dumps(dict(error='Cumulative training CSV prefix changed')))
    header=dict(receipt,direction='output',parent_sha256=parent['sha256'] if parent else None)
    packet=seal(pack(out),key,header)
    name=entry['cell']+f".segment{entry['segment']:03d}.enc"
    api.upload(req['release_id'],name,packet)
    cp=json.loads((out/'checkpoint_status.json').read_text()) if (out/'checkpoint_status.json').exists() else {}
    result=dict(header,state=state,child_exited=True,exit_code=code,archive_name=name,
        archive_sha256=sha(packet),archive_size=len(packet),paused_reason=paused['reason'] if paused else None,
        update=cp.get('update'),env_steps=cp.get('env_steps'),wall_seconds=time.monotonic()-started,
        prefix_preserved=prefix_ok if prior_csv is not None else None)
    api.upload(req['release_id'],name+'.outcome.json',json.dumps(result).encode())
    if state in ('failed','unknown'):raise RuntimeError('Failure evidence retained; no blind retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--cell')
    args=p.parse_args();req=request_from_tag()
    if args.plan:
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:
            f.write('matrix='+json.dumps(dict(include=[dict(cell=e['cell']) for e in req['entries']]))+'\n')
    else:run(next(e for e in req['entries'] if e['cell']==args.cell),req)

"""One finite LOCAL owner. No networking, credential access or public upload."""
import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.highload.protocol import plan,digest,source_hashes,configuration,ABLATIONS,SEEDS


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(20):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def ram():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
            ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    s=Status();s.length=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):return None
    return s.available/2**30


def specifications():
    return {c['id']:dict(kind='train',method=c['algorithm'],lr=c['lr_index'],seed=c['seed']) for c in plan()['cells']}


def validate_result(folder,spec):
    done=json.loads((folder/'complete.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    cfg,t=configuration(spec['method'],spec['lr'],spec['seed'])
    t['ablation']=spec['method'] if spec['method'].startswith('TMAPO-') else None
    if (done['component_only'] or done['env_steps']!=300000 or done['cycles']!=10
        or done['pending_tasks'] or manifest['environment']!=cfg or manifest['training']!=t
        or manifest['source_hashes']!=source_hashes()):raise ValueError('Scientific completion mismatch')
    with (folder/'training.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)!=3000:raise ValueError('Missing update rows')
    for i,r in enumerate(rows):
        if int(r['update'])!=i+1 or int(r['env_steps'])!=(i+1)*100 or not math.isfinite(float(r['rollout_return'])):
            raise ValueError('Wrong training budget/curve')


def selection(state):
    selected={}
    scores={}
    for method in SEEDS:
        candidates={}
        for ident,spec in state['specs'].items():
            if spec['kind']!='train' or spec['method']!=method:continue
            prefix=Path(state['cells'][ident]['local_prefix'])
            val=json.loads((prefix/'validation_3000.json').read_text())
            for row in val['rows']:
                candidates.setdefault(spec['lr'],[]).append(row)
        scores[method]={lr:sum(r['mean_task_cost']*r['arrivals'] for r in rows)/sum(r['arrivals'] for r in rows)
                        for lr,rows in candidates.items()}
        selected[method]=min(scores[method],key=lambda lr:(scores[method][lr],lr))
    return dict(selected=selected,scores=scores,rule='minimum terminal validation pooled physical cost, all original seeds retained; no final-test access')


def activate_ablations(state):
    selected=selection(state)
    state['selection']=selected
    lr=selected['selected']['TMAPO']
    for method in ABLATIONS:
        for seed in SEEDS['TMAPO']:
            ident=f'{method}_LR{lr}_seed{seed}'
            spec=dict(kind='train',method=method,lr=lr,seed=seed)
            state['specs'][ident]=spec
            state['cells'][ident]=dict(state='ready',segments=0)
            alias=f'IPPO_LR{lr}_seed{seed}'
            if method=='TMAPO-LocalCritic' and alias in state['cells']:
                origin=state['cells'][alias]
                if origin['state']=='completed':
                    m=json.loads((Path(origin['local_prefix'])/'manifest.json').read_text())
                    cfg,t=configuration(method,lr,seed)
                    mt=dict(m['training']);mt.pop('ablation',None)
                    if mt!=t or m['environment']!=cfg:raise ValueError('Ablation alias configuration mismatch')
                    state['cells'][ident]=dict(state='completed',segments=0,
                        local_prefix=origin['local_prefix'],reused_exact_architecture_config_seed=alias,
                        note='IPPO is this local-critic ablation at matching LR/seed; preserve original origin, not a new training run')
    state['phase']='ablations'


def activate_evaluation(state):
    selected=state['selection']['selected']
    for ident,spec in list(state['specs'].items()):
        method=spec['method']
        if spec['kind']!='train':continue
        if method not in ABLATIONS and spec['lr']!=selected[method]:continue
        target='eval--'+ident
        state['specs'][target]=dict(kind='evaluation',method=method,lr=spec['lr'],seed=spec['seed'],
            checkpoint_folder=state['cells'][ident]['local_prefix'],
            checkpoint_alias=state['cells'][ident].get('reused_exact_architecture_config_seed'))
        state['cells'][target]=dict(state='ready',segments=0)
    for method in ('Greedy','GeoI-Greedy'):
        target='eval--'+method
        state['specs'][target]=dict(kind='evaluation',method=method,lr=0,seed=0,checkpoint_folder=None)
        state['cells'][target]=dict(state='ready',segments=0)
    state['phase']='evaluation'


def run(root,first_only=False):
    root.mkdir(parents=True,exist_ok=False)
    fd=os.open(root/'owner.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode())
    verification=json.loads((ROOT/'evidence/latest_preflight.json').read_text())
    if verification['source_hashes']!=source_hashes() or verification['plan_hash']!=digest(plan()):
        raise ValueError('Final source-locked tests required')
    state=dict(pid=os.getpid(),started_epoch=time.time(),status='running',phase='screen',
        execution='LOCAL only; public result/checkpoint transmission not authorized',
        plan_hash=digest(plan()),source_hashes=source_hashes(),
        specs=specifications(),maximum_days=7,maximum_segments=96,
        first_only=first_only,selected_first='TMAPO_LR3_seed87021',cells={})
    state['cells']={k:dict(state='ready',segments=0) for k in state['specs']}
    save(root/'plan.json',plan());save(root/'coordinator.json',state)
    try:
        while True:
            if time.time()-state['started_epoch']>7*86400:state['status']='budget_exhausted';break
            if (root/'STOP').exists():state['status']='stopped_at_saved_boundary';break
            if source_hashes()!=state['source_hashes']:raise ValueError('Scientific source changed during active experiment')
            ready=[k for k,v in state['cells'].items() if v['state']=='ready']
            if first_only:ready=[k for k in ready if k==state['selected_first']]
            if not ready:
                if first_only:state['status']='first_formal_cell_completed_waiting_cloud_authorization';break
                if state['phase']=='screen':activate_ablations(state);continue
                if state['phase']=='ablations':activate_evaluation(state);continue
                state['status']='completed';break
            ident=state['selected_first'] if state['selected_first'] in ready else ready[0]
            c=state['cells'][ident];spec=state['specs'][ident]
            if c['segments']>=96:state['status']='segment_budget_exhausted';break
            free=ram();state['available_ram_gib']=free
            if free is None or free<1.5:
                state.setdefault('resource_wait_started',time.time())
                if time.time()-state['resource_wait_started']>12*3600:state['status']='resource_wait_exhausted';break
                state['status']='waiting_for_safe_RAM';save(root/'coordinator.json',state);time.sleep(60);continue
            state.pop('resource_wait_started',None);state['status']='running'
            folder=root/'cells'/ident
            if spec['kind']=='train':
                command=[sys.executable,'-u','-m','experiments.highload.runner','--out',str(folder),
                    '--method',spec['method'],'--lr',str(spec['lr']),'--seed',str(spec['seed']),'--max-seconds','10800']
                if c['segments']:command.append('--resume')
            else:
                command=[sys.executable,'-u',str(ROOT/'ops/evaluate_local.py'),'--out',str(folder),
                    '--method',spec['method'],'--lr',str(spec['lr']),'--seed',str(spec['seed'])]
                if spec['checkpoint_folder']:command+=['--checkpoint',spec['checkpoint_folder']]
            c['segments']+=1
            log=root/f"{ident}.segment{c['segments']:03d}.log"
            with log.open('w',encoding='utf-8') as f:
                child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                c.update(state='running',pid=child.pid,log=str(log),local_prefix=str(folder))
                save(root/'coordinator.json',state)
                while child.poll() is None:
                    if (root/'STOP').exists():
                        folder.mkdir(parents=True,exist_ok=True)
                        (folder/'STOP_AFTER_UPDATE').touch()
                    time.sleep(5)
                code=child.returncode
            if code or (folder/'failure.json').exists():
                c.update(state='failed',exit_code=code);state['status']='needs_review';break
            if (folder/'complete.json').exists():
                if spec['kind']=='train':validate_result(folder,spec)
                c['state']='completed'
            elif (folder/'paused.json').exists():
                reason=json.loads((folder/'paused.json').read_text())['reason']
                if reason=='wall budget':c['state']='ready'
                else:c['state']='paused';state['status']='saved_requires_review';break
            else:raise ValueError('Worker exited without known saved outcome')
            save(root/'coordinator.json',state)
    except BaseException as error:
        state.update(status='needs_review',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        save(root/'coordinator.json',state);os.close(fd);(root/'owner.lock').unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--first-only',action='store_true')
    a=p.parse_args();run(a.out,a.first_only)

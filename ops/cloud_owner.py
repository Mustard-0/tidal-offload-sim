"""Single bounded owner; public CPU jobs exchange only authorized ciphertext."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.highload.protocol import plan,digest,source_hashes,configuration,SEEDS
from ops.local_owner import save,ram,specifications,validate_result,activate_ablations,activate_evaluation
from ops.github_api import GitHub,REPOSITORY,download
from ops.transport import pack,unpack,seal,unseal,sha

BASE='/repos/'+REPOSITORY
KEY_FILE=Path('F:/stin_project/tmapo_lr_speed_20261026/evidence/speed_20261026/transport_key.json')
HANDOFF='TMAPO_LR3_seed87021'


def hashes(folder):
    return {n:sha((folder/n).read_bytes()) for n in
        ('latest.pt','training.csv','manifest.json','checkpoint_status.json') if (folder/n).exists()}


def prefix_audit(folder,spec):
    m=json.loads((folder/'manifest.json').read_text())
    cp=json.loads((folder/'checkpoint_status.json').read_text())
    cfg,t=configuration(spec['method'],spec['lr'],spec['seed'])
    t['ablation']=spec['method'] if spec['method'].startswith('TMAPO-') else None
    identity={k:m[k] for k in ('schema','method','lr','seed','environment','training','source_hashes','component_only','runtime')}
    if (m['component_only'] or m['environment']!=cfg or m['training']!=t or
        m['source_hashes']!=source_hashes() or digest(identity)!=m['identity_hash'] or
        m['identity_hash']!=cp['identity_hash']):raise ValueError('Frozen prefix identity mismatch')
    with (folder/'training.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=cp['update'] or cp['env_steps']!=100*len(rows) or not 0<=len(rows)<=3000:
        raise ValueError('Prefix budget mismatch')
    for i,r in enumerate(rows,1):
        if int(r['update'])!=i or int(r['env_steps'])!=100*i or not math.isfinite(float(r['rollout_return'])):
            raise ValueError('Invalid cumulative return prefix')
    if (folder/'failure.json').exists() or (folder/'.worker.lock').exists():raise ValueError('Unclean prefix')
    return cp


def ready_order(state):
    ready=[k for k,c in state['cells'].items() if c['state']=='ready']
    result=[HANDOFF] if HANDOFF in ready else []
    families={}
    for k in ready:
        if k not in result:families.setdefault(state['specs'][k]['method'],[]).append(k)
    while any(families.values()):
        for values in families.values():
            if values:result.append(values.pop(0))
    return result


def assets(api,release):
    found={}
    for page in range(1,1000):
        part=api.request('GET',BASE+f'/releases/{release}/assets?per_page=100&page={page}')
        found.update({a['name']:a for a in part})
        if len(part)<100:return found
    raise ValueError('Unexpected release pagination')


def verify_asset(a,data):
    if a['size']!=len(data) or a.get('digest')!='sha256:'+sha(data):raise ValueError('Published digest/size mismatch')


def dispatch(api,state,root,key,ids):
    entries=[]
    for ident in ids:
        c=state['cells'][ident];spec=state['specs'][ident];segment=c['segments']+1
        if segment>96:raise ValueError('Cumulative segment cap reached')
        folder=Path(c['local_prefix']) if c.get('local_prefix') else None
        if folder is None and spec.get('checkpoint_folder'):folder=Path(spec['checkpoint_folder'])
        parent=None
        if folder:
            if (folder/'.worker.lock').exists():raise ValueError('Parent still owned by worker')
            metadata=dict(campaign=state['campaign'],cell=ident,direction='input',segment=segment)
            packet=seal(pack(folder),key,metadata)
            name=ident+f'.input{segment:03d}.enc'
            a=api.upload(state['release_id'],name,packet);verify_asset(a,packet)
            parent=dict(name=name,sha256=sha(packet),prefix_hashes=hashes(folder))
        entries.append(dict(cell=ident,kind=spec['kind'],method=spec['method'],lr=spec['lr'],seed=spec['seed'],
            segment=segment,input=parent,local_handoff=bool(c.get('local_handoff'))))
    number=len(state['waves'])+1
    tag=f"highload-run-{state['campaign']}-{number:04d}"
    request=dict(campaign=state['campaign'],release_id=state['release_id'],plan_hash=state['plan_hash'],entries=entries,
        selected_TMAPO_lr=state.get('selection',{}).get('selected',{}).get('TMAPO'))
    w=dict(tag=tag,entries=entries,run_id=None,received=[],created_epoch=time.time(),finalized=False)
    state['waves'].append(w)
    for e in entries:state['cells'][e['cell']].update(state='active',segments=e['segment'],wave=number)
    # Persist intent before the only dispatch. An uncertain API result never causes redispatch.
    save(root/'coordinator.json',state)
    obj=api.request('POST',BASE+'/git/tags',dict(tag=tag,message=json.dumps(request),object=state['public_commit'],type='commit'))
    api.request('POST',BASE+'/git/refs',dict(ref='refs/tags/'+tag,sha=obj['sha']))
    w['dispatched_epoch']=time.time();save(root/'coordinator.json',state)


def collect(api,state,root,key):
    listing=assets(api,state['release_id'])
    recent=None
    for w in state['waves']:
        if w['finalized']:continue
        if w['run_id'] is None:
            if recent is None:recent=api.request('GET',BASE+'/actions/runs?event=push&per_page=100')['workflow_runs']
            found=[r for r in recent if r.get('head_branch')==w['tag']]
            if len(found)>1:raise ValueError('Multiple workflows own one wave')
            if not found:
                if time.time()-w['created_epoch']>1800:raise ValueError('Dispatched wave has no visible workflow; do not redispatch')
                continue
            w['run_id']=str(found[0]['id'])
        run=api.request('GET',BASE+'/actions/runs/'+w['run_id'])
        wave_commit=w.get('public_commit',state['public_commit'])
        if run['head_sha']!=wave_commit or run['run_attempt']!=1:raise ValueError('Wrong public code/attempt')
        jobs=api.request('GET',BASE+'/actions/runs/'+w['run_id']+'/jobs?per_page=100')['jobs']
        w['status']=run['status'];w['conclusion']=run['conclusion']
        save(root/'github'/f"run{w['run_id']}.json",dict(run=run,jobs=jobs))
        for e in w['entries']:
            ident=e['cell']
            if ident in w['received']:continue
            job=next((j for j in jobs if j['name']==e.get('job_name',ident)),None)
            expected=dict(campaign=state['campaign'],cell=ident,kind=e['kind'],method=e['method'],lr=e['lr'],seed=e['seed'],
                segment=e['segment'],run_id=w['run_id'],attempt='1',source_commit=wave_commit,plan_hash=state['plan_hash'],preflight_passed=True)
            stem=ident+f".segment{e['segment']:03d}"
            started_name=stem+'.started.json'
            if started_name in listing and not state['cells'][ident].get('started_receipt')==started_name:
                data=download(listing[started_name]);verify_asset(listing[started_name],data);start=json.loads(data)
                if any(start.get(k)!=v for k,v in expected.items()) or not start.get('process_started'):
                    raise ValueError('Wrong startup receipt')
                if e.get('local_handoff') and (not start.get('adopted_input') or start.get('update',0)<=197):
                    raise ValueError('Local continuation not proven')
                save(root/'receipts'/started_name,start)
                state['cells'][ident]['started_receipt']=started_name
            receipt_name=stem+'.enc.outcome.json'
            packed_done=bool(e.get('job_name') and receipt_name in listing and job and job['status'] in ('in_progress','completed'))
            if (job is None or job['status']!='completed') and not packed_done:
                if run['status']=='completed':raise ValueError('Workflow ended without expected training job')
                continue
            if receipt_name not in listing:
                w.setdefault('missing_since',{}).setdefault(ident,time.time())
                if time.time()-w['missing_since'][ident]>180:raise ValueError('Completed job has no published outcome: '+ident)
                continue
            data=download(listing[receipt_name]);verify_asset(listing[receipt_name],data);receipt=json.loads(data)
            expected.update(direction='output',parent_sha256=e['input']['sha256'] if e.get('input') else None)
            if any(receipt.get(k)!=v for k,v in expected.items()) or receipt.get('child_exited') is not True:
                raise ValueError('Outcome ownership mismatch')
            save(root/'receipts'/receipt_name,receipt)
            if receipt['archive_name']!=stem+'.enc':raise ValueError('Wrong archive name')
            a=listing.get(receipt['archive_name'])
            if a is None:continue
            if a['digest']!='sha256:'+receipt['archive_sha256'] or a['size']!=receipt['archive_size']:
                raise ValueError('Outcome/public asset mismatch')
            free=ram();state['available_ram_gib']=free
            if free is None or free<max(1.5,3*a['size']/2**30+.5):
                state.setdefault('resource_wait_started',time.time())
                if time.time()-state['resource_wait_started']>43200:raise ValueError('Evidence collection RAM wait exhausted')
                continue
            state.pop('resource_wait_started',None)
            packet=download(a);raw,_=unseal(packet,key,expected)
            folder=root/'received'/ident/f"segment{e['segment']:03d}"
            unpack(raw,folder)
            c=state['cells'][ident];c.update(local_prefix=str(folder),local_handoff=False)
            job_failed=job['conclusion'] not in (None,'success') and not e.get('job_name')
            if receipt['state'] in ('failed','unknown') or receipt['exit_code'] or job_failed:
                c['state']='failed';state.update(stop_expansion=True,error='Cloud failure evidence retained: '+ident)
            elif e['kind']=='train':
                prefix_audit(folder,state['specs'][ident])
                if e.get('input') and receipt['prefix_preserved'] is not True:raise ValueError('Prefix preservation not verified')
                if receipt['state']=='completed':validate_result(folder,state['specs'][ident]);c['state']='completed'
                elif receipt['state']=='paused' and receipt['paused_reason']=='wall budget':c['state']='ready'
                else:raise ValueError('Unrecognized training boundary')
            else:
                if receipt['state']=='completed':
                    done=json.loads((folder/'evaluation/complete.json').read_text())
                    if done['scenes']!=16 or done['all_raw_audits_passed'] is not True:raise ValueError('Incomplete evaluation')
                    c['state']='completed'
                elif receipt['state']=='paused' and receipt['paused_reason']=='wall budget':c['state']='ready'
                else:raise ValueError('Unrecognized evaluation boundary')
            w['received'].append(ident);save(root/'coordinator.json',state)
        if len(w['received'])==len(w['entries']) and run['status']=='completed':w['finalized']=True


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    fd=os.open(root/'owner.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    state={}
    try:
        export=json.loads((ROOT/'evidence/cloud_export.json').read_text())
        handoff=json.loads((ROOT/'evidence/handoff_verified.json').read_text())
        if not handoff['local_processes_exited'] or handoff['cell']!=HANDOFF:raise ValueError('No clean handoff')
        if source_hashes()!=export['source_hashes'] or digest(plan())!=export['plan_hash']:raise ValueError('Frozen science changed')
        folder=Path(handoff['prefix']);prefix_audit(folder,specifications()[HANDOFF])
        if hashes(folder)!=handoff['prefix_hashes']:raise ValueError('Handoff prefix changed')
        key_value=json.loads(KEY_FILE.read_text());key=bytes.fromhex(key_value['key'])
        api=GitHub();campaign='20261027-'+time.strftime('%H%M%S',time.gmtime())
        state=dict(pid=os.getpid(),campaign=campaign,started_epoch=handoff['original_started_epoch'],status='running',phase='screen',
            plan_hash=export['plan_hash'],source_hashes=export['source_hashes'],public_commit=export['commit'],
            specs=specifications(),cells={k:dict(state='ready',segments=0) for k in specifications()},waves=[],
            stop_expansion=False,maximum_parallel=16,maximum_segments=96,maximum_days=7,
            execution='standard public CPU; authorized AES-GCM simulation transfer; no plaintext results or paid compute')
        state['cells'][HANDOFF].update(segments=1,local_prefix=str(folder),local_handoff=True)
        save(root/'coordinator.json',state);save(root/'plan.json',plan())
        release=api.request('POST',BASE+'/releases',dict(tag_name='highload-store-'+campaign,target_commitish=export['commit'],
            name='Encrypted high-load campaign '+campaign,body='Authorized AES-GCM simulation packets only. No plaintext results, papers, user files or keys.',draft=False,prerelease=True))
        state.update(release_id=release['id'],release_url=release['html_url']);save(root/'coordinator.json',state)
        while True:
            try:
                if source_hashes()!=state['source_hashes']:raise ValueError('Frozen scientific source changed')
                if time.time()-state['started_epoch']>7*86400 or (root/'STOP').exists():state['stop_expansion']=True
                collect(api,state,root,key);state.pop('network_error',None)
                active=sum(c['state']=='active' for c in state['cells'].values())
                if state['stop_expansion']:
                    if not active:state['status']='needs_review';break
                else:
                    ready=ready_order(state)
                    if ready and active<16:dispatch(api,state,root,key,ready[:16-active])
                    elif not ready and not active:
                        if state['phase']=='screen':activate_ablations(state)
                        elif state['phase']=='ablations':activate_evaluation(state)
                        else:state['status']='completed';break
                save(root/'coordinator.json',state);time.sleep(45)
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as error:
                if isinstance(error,urllib.error.HTTPError) and error.code not in (429,500,502,503,504):raise
                state['network_error']=type(error).__name__;save(root/'coordinator.json',state);time.sleep(60)
    except BaseException as error:
        state.update(status='needs_review',stop_expansion=True,error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        save(root/'coordinator.json',state);os.close(fd);(root/'owner.lock').unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)

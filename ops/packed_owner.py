"""Audited single-owner adoption and up to20 standard four-process runners."""
import argparse
import copy
import http.client
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
from ops.cloud_owner import BASE,KEY_FILE,collect,ready_order,verify_asset,hashes
from ops.cloud_pack import groups
from ops.github_api import GitHub
from ops.local_owner import save,activate_ablations,activate_evaluation
from ops.transport import pack,seal,sha
from experiments.highload.protocol import source_hashes,digest,plan


class ReadRetryGitHub(GitHub):
    def request(self,method,path,value=None):
        for attempt in range(5):
            try:return super().request(method,path,value)
            except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.HTTPException) as error:
                transient=not isinstance(error,urllib.error.HTTPError) or error.code in (429,500,502,503,504)
                if method!='GET' or not transient or attempt==4:raise
                time.sleep(min(2**attempt,8))


def worker_groups(state,root):
    count=0
    for w in state['waves']:
        if w['finalized']:continue
        names={e.get('job_name',e['cell']) for e in w['entries']}
        p=root/'github'/f"run{w.get('run_id')}.json"
        known={j['name']:j for j in json.loads(p.read_text())['jobs']} if p.exists() else {}
        for name in names:
            if name in known:
                count+=known[name]['status']!='completed'
            else:count+=any(e['cell'] not in w['received'] for e in w['entries'] if e.get('job_name',e['cell'])==name)
    return count


def build_request(state,ids):
    if len(ids)>16 or len(ids)!=len(set(ids)):raise ValueError('Bounded unique dispatch required')
    entries=[]
    for i,ident in enumerate(ids):
        c=state['cells'][ident];s=state['specs'][ident]
        if c['state']!='ready' or c['segments']>=96:raise ValueError('Not uniquely ready or cumulative cap reached')
        entries.append(dict(cell=ident,kind=s['kind'],method=s['method'],lr=s['lr'],seed=s['seed'],segment=c['segments']+1,
            input=None,local_handoff=bool(c.get('local_handoff')),job_name=f'pack-{i//4:02d}'))
    return dict(campaign=state['campaign'],release_id=state['release_id'],plan_hash=state['plan_hash'],entries=entries,
        packing=4,selected_TMAPO_lr=state.get('selection',{}).get('selected',{}).get('TMAPO'))


def dispatch(api,state,root,key,ids):
    req=build_request(state,ids)
    for e in req['entries']:
        c=state['cells'][e['cell']];spec=state['specs'][e['cell']]
        folder=Path(c['local_prefix']) if c.get('local_prefix') else Path(spec['checkpoint_folder']) if spec.get('checkpoint_folder') else None
        if folder:
            if (folder/'.worker.lock').exists():raise ValueError('Parent training still alive')
            packet=seal(pack(folder),key,dict(campaign=state['campaign'],cell=e['cell'],direction='input',segment=e['segment']))
            name=e['cell']+f".input{e['segment']:03d}.enc"
            asset=api.upload(state['release_id'],name,packet);verify_asset(asset,packet)
            e['input']=dict(name=name,sha256=sha(packet),prefix_hashes=hashes(folder))
    tag=f"highload-run-{state['campaign']}-{len(state['waves'])+1:04d}"
    w=dict(tag=tag,entries=req['entries'],run_id=None,received=[],created_epoch=time.time(),finalized=False,
        public_commit=state['execution_commit'],packing=4,execution_profile='standard-CPU-four-independent-processes-v1')
    state['waves'].append(w)
    for e in req['entries']:state['cells'][e['cell']].update(state='active',segments=e['segment'],wave=len(state['waves']))
    save(root/'coordinator.json',state)
    obj=api.request('POST',BASE+'/git/tags',dict(tag=tag,message=json.dumps(req),object=w['public_commit'],type='commit'))
    api.request('POST',BASE+'/git/refs',dict(ref='refs/tags/'+tag,sha=obj['sha']))
    w['dispatched_epoch']=time.time();save(root/'coordinator.json',state)


def adoption(root,export,gate):
    audit=json.loads((ROOT/'evidence/optimization_handoff_20261027/audit.json').read_text())
    s=json.loads((root/'coordinator.json').read_text())
    if s!=audit['old_state']:raise ValueError('Old saved state changed after handoff audit')
    if (root/'STOP').exists() or s['error']!="RemoteDisconnected('Remote end closed connection without response')":
        raise ValueError('This is not the audited read-only connection failure')
    if audit['unclaimed_received_dirs'] or not audit['no_unknown_dispatch']:raise ValueError('Unresolved old ownership')
    if not gate['passed'] or not gate['equivalence_passed'] or gate['source_commit']!=export['commit']:
        raise ValueError('No actual matching cloud packing gate')
    if source_hashes()!=s['source_hashes'] or digest(plan())!=s['plan_hash']:raise ValueError('Frozen scientific source changed')
    s=copy.deepcopy(s)
    s.setdefault('owner_history',[]).append(dict(pid=s['pid'],status=s['status'],error=s['error'],handoff='audited read-only network disconnect; all existing cloud jobs adopted without rerun'))
    s.update(pid=os.getpid(),status='running',stop_expansion=False,execution_commit=export['commit'],
        maximum_parallel_jobs=20,maximum_processes_per_runner=4,
        execution_override=dict(reason='Human explicitly requested all free resource acceleration',base_plan_parallel=16,
            actual_job_cap=20,processes_per_standard_four_vCPU_runner=4,
            scientific_budget_seed_reward_model_unchanged=True,account_plan_not_returned_by_API=True,
            limit_basis='conservative documented Free20 job cap; no paid larger/GPU runners',
            packing_probe_run=gate['run_id'],packing_throughput_speedup=gate['throughput_speedup']))
    for k in ('error','traceback','network_error'):s.pop(k,None)
    return s


def run(root):
    # The lock is never removed automatically if another process owns it.
    fd=os.open(root/'owner.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    state={}
    try:
        export=json.loads((ROOT/'evidence/cloud_export.json').read_text())
        gate=json.loads((ROOT/'evidence/packing_probe_verified.json').read_text())
        state=adoption(root,export,gate)
        execution_files=('ops/packed_owner.py','ops/cloud_owner.py','ops/cloud_pack.py','experiments/highload/cloud_worker.py','.github/workflows/highload.yml')
        state['execution_source_hashes']={name:sha((ROOT/name).read_bytes().replace(b'\r\n',b'\n')) for name in execution_files}
        save(root/'coordinator.json',state)
        key=bytes.fromhex(json.loads(KEY_FILE.read_text())['key']);api=ReadRetryGitHub()
        while True:
            try:
                if source_hashes()!=state['source_hashes']:raise ValueError('Frozen scientific source changed')
                if any(sha((ROOT/n).read_bytes().replace(b'\r\n',b'\n'))!=h for n,h in state['execution_source_hashes'].items()):
                    raise ValueError('Active execution code changed')
                control=root/'execution_control.json'
                if control.exists() and json.loads(control.read_text()).get('action')=='quiesce':
                    state['status']='saved_for_owner_handoff';break
                if time.time()-state['started_epoch']>7*86400 or (root/'STOP').exists():state['stop_expansion']=True
                collect(api,state,root,key);state.pop('network_error',None)
                active=sum(c['state']=='active' for c in state['cells'].values())
                live_jobs=worker_groups(state,root);state['live_runner_jobs']=live_jobs
                if state['stop_expansion']:
                    if not active:state['status']='needs_review';break
                else:
                    ready=ready_order(state)
                    if ready and live_jobs<20:
                        # No extra cells: only the already-declared ready queue is packed.
                        while ready and live_jobs<20:
                            take=ready[:min(16,(20-live_jobs)*4)]
                            dispatch(api,state,root,key,take);live_jobs+=math.ceil(len(take)/4);ready=ready[len(take):]
                    elif not ready and not active:
                        if state['phase']=='screen':activate_ablations(state)
                        elif state['phase']=='ablations':activate_evaluation(state)
                        else:state['status']='completed';break
                save(root/'coordinator.json',state);time.sleep(30)
            except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.HTTPException) as error:
                if isinstance(error,urllib.error.HTTPError) and error.code not in (429,500,502,503,504):raise
                # Writes are not blindly retried: an unregistered dispatch must be reconciled.
                if any(w.get('dispatched_epoch') is None for w in state['waves']):raise
                state['network_error']=type(error).__name__;save(root/'coordinator.json',state);time.sleep(30)
    except BaseException as error:
        state.update(status='needs_review',stop_expansion=True,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if state:save(root/'coordinator.json',state)
        os.close(fd);(root/'owner.lock').unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)

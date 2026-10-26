"""Lightweight finite parallel coordinator. No torch or plaintext public assets."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.error
from .protocol import ROOT, plan, digest
from .github_api import GitHub, REPOSITORY, download
from .transport import sha, seal, unseal, pack, unpack

PARENT = Path('F:/stin_project/tmapo_lr_screen_20261025/runs/lr_screen_20261026_01/TMAPO_LR1_seed87021')


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def git(repo,*args):
    env = dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    return subprocess.check_output(['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo),*args],
                                    text=True,env=env,stderr=subprocess.STDOUT).strip()


def dispatch(api, state, root, key, ids):
    entries = []
    release_id = state['release_id']
    for cell_id in ids:
        c = state['cells'][cell_id]
        if c['segments'] >= 96:
            raise RuntimeError('Cumulative segment cap reached')
        segment = c['segments']+1
        parent = c.get('local_prefix')
        input_asset = None
        if parent:
            header = dict(campaign=state['campaign'],cell=cell_id,direction='input',segment=segment)
            packet = seal(pack(Path(parent)),key,header)
            name = cell_id+f'.input{segment:03d}.enc'
            api.upload(release_id,name,packet)
            input_asset = dict(name=name,sha256=sha(packet))
        entries.append(dict(cell=cell_id,segment=segment,input=input_asset))
    wave = len(state['waves'])+1
    request = dict(campaign=state['campaign'],release_id=release_id,plan_hash=digest(plan()),entries=entries)
    tag = f"lr-fast-run-{state['campaign']}-{wave:04d}"
    repo = Path(state['public_checkout'])
    git(repo,'tag','-a',tag,state['public_commit'],'-m',json.dumps(request,separators=(',',':')))
    # Persist intent first: a network error is not permission to duplicate a tag.
    state['waves'].append(dict(tag=tag,entries=entries,run_id=None,state='dispatching'))
    for e in entries:
        state['cells'][e['cell']].update(state='dispatching',segments=e['segment'],tag=tag)
    save(root/'coordinator.json',state)
    git(repo,'push','origin','refs/tags/'+tag)
    state['waves'][-1]['state'] = 'pending_registration'
    for e in entries:
        state['cells'][e['cell']]['state'] = 'active'
    save(root/'coordinator.json',state)


def collect(api, state, root, key):
    release = api.request('GET','/repos/'+REPOSITORY+'/releases/'+str(state['release_id']))
    assets = {a['name']:a for a in release['assets']}
    runs = api.request('GET','/repos/'+REPOSITORY+'/actions/runs?per_page=100')['workflow_runs']
    for wave in state['waves']:
        if wave['state'] == 'finalized':
            continue
        found = [r for r in runs if r['head_branch']==wave['tag']]
        if len(found)>1:
            raise RuntimeError('Ambiguous run registration')
        if not found:
            continue
        run = found[0]
        wave['run_id'] = run['id']
        if run['run_attempt'] != 1 or run['head_sha'] != state['public_commit']:
            raise RuntimeError('Wrong GitHub run revision or attempt')
        jobs = api.request('GET','/repos/'+REPOSITORY+'/actions/runs/'+str(run['id'])+'/jobs?per_page=100')['jobs']
        for entry in wave['entries']:
            c = state['cells'][entry['cell']]
            if c.get('received_segment',0) >= entry['segment']:
                continue
            name = entry['cell']+f".segment{entry['segment']:03d}.enc"
            receipt_name = name+'.outcome.json'
            job = next((j for j in jobs if j['name']==entry['cell']),None)
            if not job or job['status'] != 'completed':
                continue
            if receipt_name not in assets or name not in assets:
                if job['conclusion'] not in ('success',None):
                    c.update(state='failed',reason='Completed cloud job without verified result',job_id=job['id'])
                    state['stop_expansion'] = True
                continue
            receipt = json.loads(download(assets[receipt_name]))
            expected = dict(campaign=state['campaign'],cell=entry['cell'],segment=entry['segment'],
                direction='output',run_id=str(run['id']),attempt='1',source_commit=state['public_commit'],
                plan_hash=digest(plan()),parent_sha256=entry['input']['sha256'] if entry['input'] else None)
            if any(receipt.get(k) != v for k,v in expected.items()) or not receipt['child_exited']:
                raise RuntimeError('Receipt origin or parent mismatch')
            packet = download(assets[name])
            if sha(packet) != receipt['archive_sha256'] or len(packet) != receipt['archive_size']:
                raise RuntimeError('Encrypted result digest mismatch')
            raw, _ = unseal(packet,key,expected)
            folder = root/'received'/entry['cell']/f"segment{entry['segment']:03d}"
            unpack(raw,folder)
            if receipt['state']=='failed' or receipt['exit_code'] != 0:
                c.update(state='failed',reason='Cloud training failed',local_prefix=str(folder),
                         received_segment=entry['segment'])
                state['stop_expansion'] = True
                save(folder/'verified_receipt.json',receipt)
                continue
            manifest = json.loads((folder/'manifest.json').read_text())
            contract = json.loads((ROOT/'cloud/contract.json').read_text())
            cfg = next(x for x in plan()['cells'] if x['id']==entry['cell'])
            if (manifest['component_only'] or manifest['source_hashes'] != contract['source_hashes']
                    or manifest['environment'] != plan()['environment']
                    or manifest['training']['updates']*manifest['training']['rollout_steps'] != cfg['env_steps']):
                raise RuntimeError('Result scientific source/config mismatch')
            checkpoint = json.loads((folder/'checkpoint_status.json').read_text())
            if checkpoint['env_steps'] > cfg['env_steps'] or checkpoint['env_steps'] != checkpoint['update']*128:
                raise RuntimeError('Budget exceeded or malformed update boundary')
            if receipt['state']=='completed':
                done = json.loads((folder/'complete.json').read_text())
                if done['env_steps'] != cfg['env_steps'] or done['component_only']:
                    raise RuntimeError('Incomplete budget marked complete')
                c['state'] = 'completed'
            elif receipt['state']=='paused' and receipt['paused_reason']=='wall budget' and receipt['exit_code']==0:
                c['state'] = 'ready'
            else:
                c.update(state='failed',reason=receipt.get('paused_reason') or receipt['state'])
                state['stop_expansion'] = True
            c.update(local_prefix=str(folder),received_segment=entry['segment'],
                     env_steps=checkpoint['env_steps'],update=checkpoint['update'])
            save(folder/'verified_receipt.json',receipt)
        if run['status']=='completed':
            for entry in wave['entries']:
                c = state['cells'][entry['cell']]
                if c.get('received_segment',0) < entry['segment']:
                    c.update(state='failed',reason='Finalized workflow lacks verified receipt')
                    state['stop_expansion'] = True
            wave.update(state='finalized',conclusion=run['conclusion'])
    return state


def run(root, public_checkout, public_commit, key_file):
    if root.exists():
        raise FileExistsError('Existing coordinator requires manual owner/state review')
    root.mkdir(parents=True)
    key = bytes.fromhex(json.loads(key_file.read_text())['key'])
    api = GitHub()
    campaign = time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    store_tag = 'lr-fast-store-'+campaign
    release = api.request('POST','/repos/'+REPOSITORY+'/releases',dict(tag_name=store_tag,
        target_commitish=public_commit,name='Encrypted LR screen segments '+campaign,
        body='Encrypted simulation checkpoints/results only; pure code on the execution branch.',draft=False))
    cells = {c['id']:dict(state='ready',segments=0,env_steps=0) for c in plan()['cells']}
    cells['TMAPO_LR1_seed87021']['local_prefix'] = str(PARENT)
    state = dict(pid=os.getpid(),campaign=campaign,started_epoch=time.time(),state='running',
        public_checkout=str(public_checkout),public_commit=public_commit,release_id=release['id'],
        cells=cells,waves=[],stop_expansion=False,maximum_parallel=11,maximum_days=7)
    save(root/'plan.json',plan())
    save(root/'coordinator.json',state)
    try:
        while True:
            if time.time()-state['started_epoch'] > 7*86400 or (root/'STOP').exists():
                state['stop_expansion'] = True
            state = collect(api,state,root,key)
            ready = [k for k,v in state['cells'].items() if v['state']=='ready']
            if ready and not state['stop_expansion']:
                dispatch(api,state,root,key,ready)
            active = any(c['state'] in ('active','dispatching') for c in state['cells'].values())
            if all(c['state']=='completed' for c in state['cells'].values()):
                state['state']='completed'
                import shutil
                for name,cell in state['cells'].items():
                    shutil.copytree(cell['local_prefix'],root/name)
                # Plotting is a separate short-lived CPU process; the persistent
                # coordinator does not hold torch/checkpoints in memory.
                subprocess.run([os.environ.get('LR_SCIENTIFIC_PYTHON',sys.executable),'-m',
                    'experiments.lr_screen.runner','report','--out',str(root)],cwd=ROOT,check=True)
                break
            if state['stop_expansion'] and not active:
                state['state']='needs_review'
                break
            state['updated_epoch']=time.time()
            save(root/'coordinator.json',state)
            time.sleep(60)
    except BaseException as e:
        state.update(state='needs_review',error=str(e),traceback=traceback.format_exc())
        raise
    finally:
        save(root/'coordinator.json',state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--public-checkout',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--key-file',type=Path,required=True)
    a = p.parse_args()
    run(a.out,a.public_checkout,a.commit,a.key_file)

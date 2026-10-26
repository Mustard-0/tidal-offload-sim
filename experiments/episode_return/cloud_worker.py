"""Free CPU segment; encrypted full state; only three fixed episode-return runs."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from .protocol import ROOT, plan, digest, source_hashes
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download


def request_from_tag():
    tag = os.environ['GITHUB_REF_NAME']
    if not tag.startswith('lr-fast-run-short100-') or os.environ['GITHUB_RUN_ATTEMPT'] != '1':
        raise ValueError('Wrong run tag or retry; no blind reruns')
    request = json.loads(subprocess.check_output(['git','for-each-ref','refs/tags/'+tag,
        '--format=%(contents)'],text=True))
    expected = {c['id'] for c in plan()['cells']}
    ids = [e['cell'] for e in request['entries']]
    if (request['plan_hash'] != digest(plan()) or len(ids) != len(set(ids))
            or not 1<=len(ids)<=16 or not set(ids) <= expected):
        raise ValueError('Unknown/duplicated cell or changed plan')
    if any(not 2 <= e['segment'] <= 96 or not e.get('input') for e in request['entries']):
        raise ValueError('Segment cap')
    return request


def run(entry, request):
    from experiments.lr_screen.runtime import runtime_info, torch
    from experiments.lr_screen.transport import sha, seal, unseal, pack, unpack
    torch.set_num_threads(1)
    contract = json.loads((ROOT/'cloud/contract.json').read_text())
    if contract['source_hashes'] != source_hashes() or contract['plan_hash'] != digest(plan()):
        raise ValueError('Source contract mismatch')
    runtime = runtime_info('cpu')
    for k, value in contract['cloud_runtime'].items():
        actual = '.'.join(map(str,sys.version_info[:3])) if k == 'python_version' else runtime[k]
        if actual != value:
            raise ValueError('Wrong CPU runtime '+k)
    if sys.platform != 'linux':
        raise ValueError('Only reviewed free Ubuntu CPU runtime')
    key = bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    api = GitHub(os.environ['GH_TOKEN'])
    cell = next(c for c in plan()['cells'] if c['id'] == entry['cell'])
    out = ROOT/'runs/episode_cloud'/cell['id']
    parent = entry.get('input')
    initial_extension=entry['segment']==2
    input_folder=ROOT/'runs/completed2000_input'/cell['id'] if initial_extension else out
    if parent:
        release = api.request('GET','/repos/'+REPOSITORY+'/releases/'+str(request['release_id']))
        asset = next(a for a in release['assets'] if a['name'] == parent['name'])
        packet = download(asset)
        if sha(packet) != parent['sha256']:
            raise ValueError('Parent digest mismatch')
        raw, _ = unseal(packet,key,dict(campaign=request['campaign'],cell=cell['id'],direction='input'))
        unpack(raw,input_folder)
        if (input_folder/'.worker.lock').exists() or (input_folder/'failure.json').exists():
            raise ValueError('Unclean worker or unknown failure; no automatic resume')
    # Execute only the minimal new-path test on first segments. The mechanism,
    # shared environment and PPO core retain the already-reviewed source hashes.
    if initial_extension:
        checks = [sys.executable,'-m','unittest','experiments.episode_return.test_runner','-v']
        subprocess.run(checks,cwd=ROOT,check=True,timeout=600)
    command = [sys.executable,'-u','-m','experiments.episode_return.runner',
        '--method',cell['algorithm'],'--lr',str(cell['lr_index']),'--seed',str(cell['seed']),
        '--out',str(out),'--max-seconds','10800']
    if initial_extension:
        command.extend(['--extend-from',str(input_folder)])
    elif parent:
        command.append('--resume')
    started = time.monotonic()
    log = ROOT/'episode_worker.log'
    with log.open('w',encoding='utf-8') as stream:
        child = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        # A startup acknowledgment follows explicit adoption, never merely Popen.
        origin_file=out/'continuation_origin.json'
        deadline=time.monotonic()+120
        while not origin_file.exists() and child.poll() is None and time.monotonic()<deadline:
            time.sleep(.25)
        started_receipt = dict(campaign=request['campaign'],cell=cell['id'],segment=entry['segment'],
            run_id=os.environ['GITHUB_RUN_ID'],attempt='1',source_commit=os.environ['GITHUB_SHA'],
            plan_hash=digest(plan()),preflight_passed=True,formal_process_started=True,
            child_pid=child.pid,resumed=bool(parent),explicit_budget_extension=initial_extension,
            parent_episodes=2000,target_episodes=3000,lr_schedule='terminal10percent retained',utc_epoch=time.time(),
            actual_region_population=[34,33,33],arrival_steps=100,drainage_included=True)
        started_receipt.update(algorithm=cell['algorithm'],lr_index=cell['lr_index'],seed=cell['seed'],
            actor_lr=cell['actor_lr'],critic_lr=cell['critic_lr'])
        if origin_file.exists():
            origin=json.loads(origin_file.read_text())
            started_receipt.update(parent_adoption_verified=True,
                parent_checkpoint_sha256=origin['parent_checkpoint_sha256'],
                parent_csv_sha256=origin['parent_csv_sha256'])
            try:
                api.upload(request['release_id'],cell['id']+f".segment{entry['segment']:03d}.started.json",
                           json.dumps(started_receipt).encode())
            except Exception as error:
                # Optional observability cannot cancel a healthy bounded child.
                print('Startup receipt upload unavailable:',type(error).__name__,flush=True)
        result_code = child.wait(timeout=18000)
    done = json.loads((out/'complete.json').read_text()) if (out/'complete.json').exists() else None
    paused = json.loads((out/'paused.json').read_text()) if (out/'paused.json').exists() else None
    failed = result_code != 0 or (out/'failure.json').exists()
    state = 'failed' if failed else 'completed' if done else 'paused' if paused else 'unknown'
    if state=='completed':
        try:
            from .report import plot_cell
            plot_cell(out,cell['id'])
        except Exception as error:
            (out/'plot_failure.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error))))
    shutil.copy2(log,out/'cloud_worker.log')
    checkpoint = json.loads((out/'checkpoint_status.json').read_text())
    header = dict(campaign=request['campaign'],cell=cell['id'],segment=entry['segment'],
        direction='output',run_id=os.environ['GITHUB_RUN_ID'],attempt='1',
        source_commit=os.environ['GITHUB_SHA'],plan_hash=digest(plan()),
        parent_sha256=parent['sha256'] if parent else None)
    header.update(algorithm=cell['algorithm'],lr_index=cell['lr_index'],seed=cell['seed'],
        actor_lr=cell['actor_lr'],critic_lr=cell['critic_lr'])
    packet = seal(pack(out),key,header)
    name = cell['id']+f".segment{entry['segment']:03d}.enc"
    api.upload(request['release_id'],name,packet)
    receipt = dict(header,state=state,child_exited=True,exit_code=result_code,
        archive_name=name,archive_sha256=sha(packet),archive_size=len(packet),
        paused_reason=paused['reason'] if paused else None,
        env_steps=checkpoint['env_steps'],episodes=checkpoint['completed_episodes'],
        training_wall_s=checkpoint['training_wall_s'],wall_s=time.monotonic()-started)
    api.upload(request['release_id'],name+'.outcome.json',json.dumps(receipt).encode())
    if failed or state == 'unknown':
        raise RuntimeError('Failed/unknown training; encrypted evidence retained')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan',action='store_true')
    p.add_argument('--cell')
    args = p.parse_args()
    request = request_from_tag()
    if args.plan:
        with open(os.environ['GITHUB_OUTPUT'],'a') as stream:
            stream.write('matrix='+json.dumps(dict(include=[dict(cell=e['cell'])
                for e in request['entries']]),separators=(',',':'))+'\n')
    else:
        run(next(e for e in request['entries'] if e['cell'] == args.cell),request)


if __name__ == '__main__':
    main()

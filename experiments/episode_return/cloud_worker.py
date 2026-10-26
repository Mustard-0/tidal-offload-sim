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
    if not tag.startswith('lr-fast-run-episode-') or os.environ['GITHUB_RUN_ATTEMPT'] != '1':
        raise ValueError('Wrong run tag or retry; no blind reruns')
    request = json.loads(subprocess.check_output(['git','for-each-ref','refs/tags/'+tag,
        '--format=%(contents)'],text=True))
    expected = {c['id'] for c in plan()['cells']}
    ids = [e['cell'] for e in request['entries']]
    if request['plan_hash'] != digest(plan()) or len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError('Unknown/duplicated cell or changed plan')
    if any(not 1 <= e['segment'] <= 96 for e in request['entries']):
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
    if parent:
        release = api.request('GET','/repos/'+REPOSITORY+'/releases/'+str(request['release_id']))
        asset = next(a for a in release['assets'] if a['name'] == parent['name'])
        packet = download(asset)
        if sha(packet) != parent['sha256']:
            raise ValueError('Parent digest mismatch')
        raw, _ = unseal(packet,key,dict(campaign=request['campaign'],cell=cell['id'],direction='input'))
        unpack(raw,out)
        if (out/'.worker.lock').exists() or (out/'failure.json').exists():
            raise ValueError('Unclean worker or unknown failure; no automatic resume')
    # Execute only the minimal new-path test on first segments. The mechanism,
    # shared environment and PPO core retain the already-reviewed source hashes.
    if not parent:
        checks = [sys.executable,'-m','unittest','experiments.episode_return.test_runner','-v']
        subprocess.run(checks,cwd=ROOT,check=True,timeout=600)
    command = [sys.executable,'-u','-m','experiments.episode_return.runner',
        '--method',cell['algorithm'],'--out',str(out),'--max-seconds','10800']
    if parent:
        command.append('--resume')
    started = time.monotonic()
    log = ROOT/'episode_worker.log'
    with log.open('w',encoding='utf-8') as stream:
        result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=18000)
    done = json.loads((out/'complete.json').read_text()) if (out/'complete.json').exists() else None
    paused = json.loads((out/'paused.json').read_text()) if (out/'paused.json').exists() else None
    failed = result.returncode != 0 or (out/'failure.json').exists()
    state = 'failed' if failed else 'completed' if done else 'paused' if paused else 'unknown'
    shutil.copy2(log,out/'cloud_worker.log')
    checkpoint = json.loads((out/'checkpoint_status.json').read_text())
    header = dict(campaign=request['campaign'],cell=cell['id'],segment=entry['segment'],
        direction='output',run_id=os.environ['GITHUB_RUN_ID'],attempt='1',
        source_commit=os.environ['GITHUB_SHA'],plan_hash=digest(plan()),
        parent_sha256=parent['sha256'] if parent else None)
    packet = seal(pack(out),key,header)
    name = cell['id']+f".segment{entry['segment']:03d}.enc"
    api.upload(request['release_id'],name,packet)
    receipt = dict(header,state=state,child_exited=True,exit_code=result.returncode,
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

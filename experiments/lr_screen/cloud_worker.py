"""One bounded standard-CPU segment, encrypted input/output, no other studies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from .protocol import ROOT, plan, digest
from .github_api import GitHub, REPOSITORY, download


def request_from_tag():
    tag = os.environ['GITHUB_REF_NAME']
    if not tag.startswith('lr-fast-run-') or os.environ.get('GITHUB_RUN_ATTEMPT') != '1':
        raise ValueError('Only first attempts of exact finite campaign tags are accepted')
    request = json.loads(subprocess.check_output(['git','for-each-ref','refs/tags/'+tag,
                                                  '--format=%(contents)'],text=True))
    if request['plan_hash'] != digest(plan()) or len(request['entries']) > 11:
        raise ValueError('Changed scientific plan or expanded matrix')
    expected = {c['id'] for c in plan()['cells']}
    ids = [c['cell'] for c in request['entries']]
    if len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError('Unknown or duplicated task')
    if any(not 1 <= c['segment'] <= 96 for c in request['entries']):
        raise ValueError('Segment budget exceeded')
    return request


def prepare_gate():
    from .runtime import source_hashes, runtime_info, json_write, torch
    from .readiness import APPROVAL
    torch.set_num_threads(1)
    contract = json.loads((ROOT/'cloud/contract.json').read_text())
    if source_hashes() != contract['source_hashes'] or digest(plan()) != contract['plan_hash']:
        raise ValueError('Unreviewed cloud source or plan')
    info = runtime_info('cpu')
    for name,value in contract['cloud_runtime'].items():
        actual = info[name] if name != 'python_version' else '.'.join(map(str,sys.version_info[:3]))
        if actual != value:
            raise ValueError('Unexpected cloud dependency '+name)
    if sys.platform != 'linux':
        raise ValueError('This gate is only for the reviewed Ubuntu CPU runner')
    command = [sys.executable,'-m','unittest','experiments.lr_screen.test_components',
               'experiments.lr_screen.test_campaign','experiments.lr_screen.test_transport',
               'model.test_protocol','model.test_review_fixes','-v']
    subprocess.run(command,cwd=ROOT,check=True,timeout=600)
    json_write(APPROVAL,dict(passed=True,source_hashes=source_hashes(),plan_hash=digest(plan()),
                            runtime=info,scope='Reviewed free standard Ubuntu CPU segment'))


def run(entry, request):
    from .transport import sha, seal, unseal, pack, unpack
    key = bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    if len(key) != 32:
        raise ValueError('Missing encryption key; never publish plaintext result')
    api = GitHub(os.environ['GH_TOKEN'])
    cell = next(c for c in plan()['cells'] if c['id'] == entry['cell'])
    out = ROOT/'runs/cloud'/cell['id']
    if request.get('mode') == 'benchmark':
        prepare_gate()
        command = [sys.executable,'-m','experiments.lr_screen.profile_speed','--device','cpu',
                   '--out','evidence/cloud_benchmark']
        subprocess.run(command,cwd=ROOT,check=True,timeout=600)
        data = (ROOT/'evidence/cloud_benchmark/timing.json').read_bytes()
        header = dict(campaign=request['campaign'],cell=cell['id'],direction='benchmark',
                      run_id=os.environ['GITHUB_RUN_ID'],source_commit=os.environ['GITHUB_SHA'])
        packet = seal(data,key,header)
        name = 'benchmark.enc'
        api.upload(request['release_id'],name,packet)
        api.upload(request['release_id'],name+'.outcome.json',json.dumps(dict(header,
            child_exited=True,exit_code=0,archive_name=name,archive_sha256=sha(packet),archive_size=len(packet))).encode())
        return
    parent = entry.get('input')
    if parent:
        release = api.request('GET','/repos/'+REPOSITORY+'/releases/'+str(request['release_id']))
        asset = next(a for a in release['assets'] if a['name'] == parent['name'])
        packet = download(asset)
        if sha(packet) != parent['sha256']:
            raise ValueError('Input asset does not match dispatch digest')
        raw, _ = unseal(packet,key,dict(campaign=request['campaign'],cell=cell['id'],direction='input'))
        unpack(raw,out)
    prepare_gate()
    if parent:
        from .portable import adopt
        adopt(out,cell['algorithm'],cell['lr_index'])
    started = time.monotonic()
    command = [sys.executable,'-u','-m','experiments.lr_screen.runner','run','--out',str(out.parent),
               '--method',cell['algorithm'],'--lr',str(cell['lr_index']),'--device','cpu',
               '--max-seconds','10800']
    if parent:
        command.append('--resume')
    log = ROOT/'worker.log'
    with log.open('w',encoding='utf-8') as f:
        result = subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=18000)
    # Publish only after the training process has actually exited.
    done = json.loads((out/'complete.json').read_text()) if (out/'complete.json').exists() else None
    paused = json.loads((out/'paused.json').read_text()) if (out/'paused.json').exists() else None
    failed = (out/'failure.json').exists() or result.returncode != 0
    state = 'failed' if failed else 'completed' if done else 'paused' if paused else 'unknown'
    header = dict(campaign=request['campaign'],cell=cell['id'],segment=entry['segment'],
                  direction='output',run_id=os.environ['GITHUB_RUN_ID'],attempt='1',
                  source_commit=os.environ['GITHUB_SHA'],plan_hash=digest(plan()),
                  parent_sha256=parent['sha256'] if parent else None)
    # Include worker diagnostics inside the encrypted prefix, not public logs.
    import shutil
    shutil.copy2(log,out/'cloud_worker.log')
    packet = seal(pack(out),key,header)
    name = cell['id']+f".segment{entry['segment']:03d}.enc"
    api.upload(request['release_id'],name,packet)
    receipt = dict(header,state=state,child_exited=True,exit_code=result.returncode,
        archive_name=name,archive_sha256=sha(packet),archive_size=len(packet),
        paused_reason=paused['reason'] if paused else None,
        env_steps=done['env_steps'] if done else None,wall_s=time.monotonic()-started)
    api.upload(request['release_id'],name+'.outcome.json',json.dumps(receipt).encode())
    if failed or state == 'unknown':
        raise RuntimeError('Scientific/code failure; encrypted evidence preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--cell')
    a = parser.parse_args()
    request = request_from_tag()
    if a.plan:
        matrix = dict(include=[dict(cell=e['cell']) for e in request['entries']])
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:
            f.write('matrix='+json.dumps(matrix,separators=(',',':'))+'\n')
        return
    run(next(e for e in request['entries'] if e['cell'] == a.cell),request)


if __name__ == '__main__':
    main()

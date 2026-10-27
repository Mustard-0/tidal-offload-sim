"""One bounded standard-public-CPU workflow; encrypted simulation outputs only."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest
import urllib.error

from .protocol import ROOT,plan,digest,source_hashes,METHODS,SEEDS
from experiments.lr_screen.github_api import GitHub,REPOSITORY,download
from experiments.lr_screen.transport import seal,unseal,pack,unpack,sha

BASE='/repos/'+REPOSITORY


def campaign():
    value=os.environ['STUDY_CAMPAIGN']
    if not re.fullmatch(r'basic-n100-[a-z0-9-]{1,60}',value):raise ValueError('Invalid finite campaign tag')
    if os.environ['GITHUB_RUN_ATTEMPT']!='1':raise ValueError('No automatic or manual rerun adoption')
    if len(bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']))!=32:raise ValueError('Missing encryption key')
    return value


def header(cell,direction='output'):
    return dict(campaign=campaign(),cell=cell,direction=direction,
        run_id=os.environ['GITHUB_RUN_ID'],attempt='1',source_commit=os.environ['GITHUB_SHA'],
        plan_hash=digest(plan()),source_hashes_hash=digest(source_hashes()))


def api():return GitHub(os.environ['GH_TOKEN'])


def release():
    return api().request('GET',BASE+'/releases/tags/'+campaign())


def upload_file(name,data):
    return api().upload(release()['id'],name,data)


def encrypt_folder(folder,cell):
    packet=seal(pack(folder),bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']),header(cell))
    upload_file(cell+'.enc',packet)
    receipt=dict(header(cell),archive_name=cell+'.enc',archive_sha256=sha(packet),archive_size=len(packet))
    upload_file(cell+'.receipt.json',json.dumps(receipt).encode())


def gate():
    campaign()
    if sys.platform!='linux':raise RuntimeError('Only the authorized standard Linux CPU workflow')
    from experiments.lr_screen.runtime import torch,np,runtime_info,json_write
    import scipy
    if (sys.version_info[:3]!=(3,9,25) or torch.__version__!='2.5.1+cpu'
        or np.__version__!='2.0.2' or scipy.__version__!='1.13.1'):
        raise RuntimeError('Unexpected frozen scientific runtime')
    a=api();repo=a.request('GET',BASE)
    if repo['private']:raise RuntimeError('Refuse unknown-cost private Actions execution')
    # A unique campaign release is created once. Never blindly retry writes.
    a.request('POST',BASE+'/releases',dict(tag_name=campaign(),target_commitish=os.environ['GITHUB_SHA'],
        name=campaign(),body='Encrypted finite simulation campaign; code and non-performance execution receipts only.',draft=False,prerelease=True))
    folder=ROOT/'runs/preflight';folder.mkdir(parents=True,exist_ok=False)
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
        suite=unittest.defaultTestLoader.loadTestsFromNames(['study.test_gate','study.test_core'])
        result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    (folder/'tests.log').write_text(buffer.getvalue(),encoding='utf-8')
    value=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
               failures=len(result.failures),errors=len(result.errors),runtime=runtime_info('cpu'))
    json_write(folder/'result.json',value)
    encrypt_folder(folder,'preflight-private')
    upload_file('preflight.json',json.dumps(dict(header('preflight'),
        passed=value['passed'],tests_run=value['tests_run'],
        source_contract=True,checkpoint_resume_tested=True)).encode())
    if not result.wasSuccessful():raise RuntimeError('Preflight failed; encrypted test evidence retained; no formal jobs allowed')
    print('All source-locked contract and real learning/resume checks passed',flush=True)


def worker(cell):
    if cell not in {c['id'] for c in plan()['cells']}:raise ValueError('Outside declared six cells')
    gate_asset=next(a for a in release()['assets'] if a['name']=='preflight.json')
    proof=json.loads(download(gate_asset))
    if not proof['passed'] or any(proof[k]!=v for k,v in header('preflight').items()):
        raise ValueError('Wrong or failed source-locked gate')
    spec=next(c for c in plan()['cells'] if c['id']==cell)
    out=ROOT/'runs/formal'/cell
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u','-m','study.runner','--out',str(out),
             '--method',spec['method'],'--seed',str(spec['seed'])]
    with (out/'worker.log').open('w',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        started=False
        while child.poll() is None:
            status=out/'status.json'
            if not started and status.exists():
                try:s=json.loads(status.read_text())
                except (ValueError,OSError):time.sleep(1.);continue
                if s['episodes']>=10:
                    upload_file(cell+'.started.json',json.dumps(dict(header(cell),
                        process_started=True,real_updates=s['episodes'],
                        initial_validation_saved=(out/'initial_validation_metrics.csv').exists(),
                        preflight_passed=True,cpu_count=os.cpu_count())).encode())
                    started=True
            time.sleep(2.)
        rc=child.wait()
    if rc==0 and (out/'complete.json').exists() and cell=='TMAPO_seed87021':
        from .rules import evaluate_rules
        evaluate_rules(out)
    encrypt_folder(out,cell)
    state='completed' if rc==0 and (out/'complete.json').exists() else 'paused' if (out/'paused.json').exists() else 'failed'
    upload_file(cell+'.outcome.json',json.dumps(dict(header(cell),state=state,
        child_exited=True,exit_code=rc,archive_name=cell+'.enc')).encode())
    if state!='completed':raise RuntimeError('Finite cell not complete; encrypted prefix retained, no rerun')
    print('Finite cell completed; private checkpoint/log/metrics encrypted',flush=True)


def collect():
    assets=release()['assets'];folders=[];missing=[]
    for cell in (c['id'] for c in plan()['cells']):
        match=[a for a in assets if a['name']==cell+'.enc']
        if len(match)!=1:missing.append(cell);continue
        packet=download(match[0])
        raw,meta=unseal(packet,bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']),header(cell))
        folder=ROOT/'runs/received'/cell;unpack(raw,folder);folders.append(folder)
    result=ROOT/'runs/comparison';result.mkdir(parents=True,exist_ok=False)
    if missing or len(folders)!=6 or any(not (f/'complete.json').exists() for f in folders):
        (result/'incomplete.json').write_text(json.dumps(dict(missing=missing,received=len(folders))))
        encrypt_folder(result,'comparison');raise RuntimeError('Incomplete finite study; evidence preserved, no complete figure claimed')
    from .report import report
    report(folders,result)
    encrypt_folder(result,'comparison')
    upload_file('campaign.completed.json',json.dumps(dict(header('comparison'),
        completed=True,training_cells=6,evaluation_populations=list(plan()['evaluation_populations']))).encode())
    print('Six-cell campaign and20-150 user figures completed; encrypted delivery published')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gate','worker','collect'))
    p.add_argument('--cell');args=p.parse_args()
    if args.mode=='gate':gate()
    elif args.mode=='worker':worker(args.cell)
    else:collect()

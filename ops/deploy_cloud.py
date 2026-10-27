"""Publish allowlisted pure code only after explicit ciphertext-transfer consent."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.highload.protocol import source_hashes,plan,digest
from ops.github_api import GitHub,REPOSITORY

BRANCH='experiment/highload-ablation-20261027'


def git(*args,api=None):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    if api:
        encoded=base64.b64encode(('x-access-token:'+api.token).encode()).decode()
        env.update(GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='credential.helper',GIT_CONFIG_VALUE_0='',
            GIT_CONFIG_KEY_1='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_1='Authorization: Basic '+encoded)
        proxy=urllib.request.getproxies().get('https')
        if proxy:env.update(GIT_CONFIG_COUNT='3',GIT_CONFIG_KEY_2='http.proxy',GIT_CONFIG_VALUE_2=proxy)
    r=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),*args],
        capture_output=True,text=True,env=env,timeout=180)
    if r.returncode:raise RuntimeError('Git operation failed; credentials and header diagnostics suppressed')
    return r.stdout.strip()


def prepare():
    frozen=json.loads((ROOT/'evidence/latest_preflight.json').read_text())
    if frozen['source_hashes']!=source_hashes() or frozen['plan_hash']!=digest(plan()):
        raise ValueError('Do not deploy changed scientific source')
    if git('branch','--show-current')!=BRANCH:raise ValueError('Wrong independent branch')
    contract=dict(source_hashes=source_hashes(),plan_hash=digest(plan()),
        runtime=dict(torch='2.5.1+cpu',numpy='2.0.2',scipy='1.13.1',device='cpu',torch_threads=1),
        platform='ubuntu-22.04 standard public CPU; no paid runner',
        authorization='Human explicitly approved public pure code and encrypted simulation checkpoints/logs/results on this repository; no plaintext results/PDFs/keys/user files')
    (ROOT/'cloud').mkdir(exist_ok=True)
    (ROOT/'cloud/contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    return contract


def publish():
    contract=prepare()
    if git('status','--porcelain'):raise ValueError('Commit reviewed code/contract before publication')
    files=git('ls-files').splitlines()
    allowed_prefixes=('model/','experiments/','ops/','.github/workflows/')
    for name in files:
        if name not in ('.gitignore','README.md','cloud/contract.json') and not name.startswith(allowed_prefixes):
            raise ValueError('File not allowlisted: '+name)
        if Path(name).suffix.lower() not in ('.py','.ps1','.md','.json','.yml',''):
            raise ValueError('Unexpected non-code file')
        data=(ROOT/name).read_bytes()
        if re.search(rb'(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|(?m:^-----BEGIN [A-Z ]*PRIVATE KEY-----\r?$))',data):
            raise ValueError('Potential embedded credential')
    api=GitHub()
    repo=api.request('GET','/repos/'+REPOSITORY)
    if repo['visibility']!='public':raise ValueError('Only public standard-free-CPU target authorized')
    if not api.request('GET','/repos/'+REPOSITORY+'/actions/permissions')['enabled']:
        raise ValueError('Actions disabled')
    api.request('GET','/repos/'+REPOSITORY+'/actions/secrets/LR_SCREEN_TRANSPORT_KEY')
    remote=git('remote').splitlines()
    if 'origin' not in remote:git('remote','add','origin','https://github.com/'+REPOSITORY+'.git')
    if git('remote','get-url','origin')!='https://github.com/'+REPOSITORY+'.git':raise ValueError('Unexpected remote')
    commit=git('rev-parse','HEAD')
    git('push','origin','HEAD:refs/heads/'+BRANCH,api=api)
    meta=dict(published=True,commit=commit,branch=BRANCH,repository=REPOSITORY,
        plan_hash=contract['plan_hash'],source_hashes=contract['source_hashes'],files=files,
        only_pure_code_uploaded=True,private_data_uploads_here=0)
    (ROOT/'evidence/cloud_export.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('source_hashes','files')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','publish'))
    a=p.parse_args();prepare() if a.mode=='prepare' else publish()

"""Independent one-thread processes share a standard four-vCPU runner."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.highload.cloud_worker import request_from_tag


def groups(entries,workers=4):
    if not 1<=workers<=4:raise ValueError('Only reviewed standard CPU process count')
    return [entries[i:i+workers] for i in range(0,len(entries),workers)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--job')
    a=p.parse_args();req=request_from_tag()
    if req.get('packing')!=4:raise ValueError('Missing reviewed execution profile')
    packs=groups(req['entries'])
    for i,entries in enumerate(packs):
        if any(e['job_name']!=f'pack-{i:02d}' for e in entries):raise ValueError('Pack ownership mismatch')
    if a.plan:
        matrix=dict(include=[dict(job=f'pack-{i:02d}') for i in range(len(packs))])
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('matrix='+json.dumps(matrix)+'\n')
        return
    entries=next(es for es in packs if es[0]['job_name']==a.job)
    if len(os.sched_getaffinity(0))<4:raise ValueError('Expected standard four-vCPU runner')
    subprocess.run([sys.executable,'-m','unittest','experiments.highload.test_protocol','-v'],cwd=ROOT,check=True,timeout=600)
    env=dict(os.environ,HIGHLOAD_SHARED_PREFLIGHT_SHA=os.environ['GITHUB_SHA'])
    children=[]
    for e in entries:
        children.append(subprocess.Popen([sys.executable,'-u','-m','experiments.highload.cloud_worker','--cell',e['cell']],cwd=ROOT,env=env))
    codes=[c.wait(timeout=19000) for c in children]
    if any(codes):raise RuntimeError('A worker failed; all bounded siblings were allowed to save and exit')


if __name__=='__main__':main()

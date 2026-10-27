"""Publish only execution helpers; frozen scientific identity stays unchanged."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops import deploy_cloud
from ops.local_owner import save
from ops.github_api import GitHub,REPOSITORY
from experiments.highload.protocol import plan,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('publish','probe'));a=p.parse_args()
    deploy_cloud.BRANCH='experiment/highload-packed-cpu-20261027'
    if a.mode=='publish':
        old=ROOT/'evidence/cloud_export.json'
        backup=ROOT/'evidence/optimization_handoff_20261027/original_cloud_export.json'
        if not backup.exists():save(backup,json.loads(old.read_text()))
        deploy_cloud.publish();return
    api=GitHub();s=json.loads((ROOT/'runs/highload_cloud_01/coordinator.json').read_text())
    export=json.loads((ROOT/'evidence/cloud_export.json').read_text())
    target=ROOT/'evidence/packing_probe_dispatch.json'
    if target.exists():raise ValueError('Never repeat an uncertain probe dispatch')
    tag='highload-probe-'+s['campaign']
    entry=dict(cell='TMAPO_LR1_seed87021',kind='train',method='TMAPO',lr=1,seed=87021,segment=1,input=None)
    req=dict(campaign=s['campaign'],release_id=s['release_id'],plan_hash=digest(plan()),entries=[entry],component_only=True)
    save(target,dict(tag=tag,source_commit=export['commit'],request=req,status='intent_recorded'))
    obj=api.request('POST','/repos/'+REPOSITORY+'/git/tags',dict(tag=tag,message=json.dumps(req),object=export['commit'],type='commit'))
    api.request('POST','/repos/'+REPOSITORY+'/git/refs',dict(ref='refs/tags/'+tag,sha=obj['sha']))
    print(json.dumps(dict(tag=tag,dispatched=True,component_only=True,source_commit=export['commit'])))


if __name__=='__main__':main()

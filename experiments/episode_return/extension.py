"""Explicit budget extension of completed prefixes, never an implicit resume."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from .protocol import ROOT,parent_configuration,source_hashes,digest
from experiments.lr_screen.runtime import torch,checkpoint_write,json_write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_extension(parent,out,identity,actual,component=False):
    parent,out=Path(parent).resolve(),Path(out).resolve()
    if parent==out or parent in out.parents or out in parent.parents:
        raise ValueError('Extension must use a separate output directory')
    manifest=json.loads((parent/'manifest.json').read_text())
    done=json.loads((parent/'complete.json').read_text())
    status=json.loads((parent/'checkpoint_status.json').read_text())
    method,lr,seed=identity['method'],identity['lr'],identity['training']['seed']
    if (parent/'.worker.lock').exists() or (parent/'failure.json').exists():
        raise ValueError('Parent is active or failed')
    expected_cfg,expected_training=parent_configuration(method,lr,seed)
    if component:
        if not manifest['component_only']:
            raise ValueError('Component extension cannot adopt formal data')
        parent_n=manifest['training']['episodes']
        expected_training=dict(manifest['training'])
        if manifest['source_hashes']!=source_hashes():
            raise ValueError('Wrong component parent sources')
    else:
        parent_n=2000
        contracts=json.loads((ROOT/'experiments/episode_return/parent_contracts.json').read_text())
        accepted=contracts['contracts'].get(manifest['source_commit'])
        if accepted is None or manifest['source_hashes']!=accepted['source_hashes']:
            raise ValueError('Unrecognized frozen episode2000 origin')
        if manifest['component_only'] or manifest['schema']!='short100-natural-drain-v1':
            raise ValueError('Not a completed formal parent')
        current=source_hashes()
        for name,value in accepted['source_hashes'].items():
            if name.startswith(('model/','experiments/lr_screen/')) and current.get(name)!=value:
                raise ValueError('Model, physical environment, mechanism or PPO core changed')
    if (manifest['method']!=method or manifest['lr']!=lr or manifest['environment']!=expected_cfg
            or manifest['training']!=expected_training or manifest['runtime']!=identity['runtime']):
        raise ValueError('Parent algorithm/LR/seed/config/runtime differs')
    old_identity={k:manifest[k] for k in ('schema','method','lr','environment','training','runtime',
                                       'source_hashes','component_only','transfer_integrated')}
    if digest(old_identity)!=manifest['identity_hash']:
        raise ValueError('Parent identity hash mismatch')
    if (done['episodes']!=parent_n or status['completed_episodes']!=parent_n
            or status['pending_tasks']!=0 or not done['drainage_included']
            or bool(done['component_only'])!=bool(component)):
        raise ValueError('Parent is incomplete, undrained or mislabeled')
    saved=torch.load(parent/'latest.pt',map_location='cpu',weights_only=False)
    if (saved['identity_hash']!=manifest['identity_hash'] or saved['completed_episodes']!=parent_n
            or saved['update']!=parent_n or saved['env_steps']!=done['env_steps']
            or saved['env_steps']!=status['env_steps'] or saved['environment'].tasks):
        raise ValueError('Parent checkpoint boundary mismatch')
    with (parent/'episode_returns.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=parent_n or len(saved['rows'])!=parent_n:
        raise ValueError('Missing parent episode prefix')
    for i,(row,stored) in enumerate(zip(rows,saved['rows']),1):
        if (row!={k:str(v) for k,v in stored.items()} or int(row['episode'])!=i
                or int(row['arrivals'])!=int(row['ended'])):
            raise ValueError('Parent CSV and saved prefix disagree')
    for optimizer,key in (('aopt','actor_lr'),('copt','critic_lr')):
        opt=saved['learner'][optimizer]
        if not opt['state']:
            raise ValueError('Optimizer state is missing')
        for group in opt['param_groups']:
            expected=expected_training[key]*expected_training['lr_final_fraction']
            if not math.isclose(group['lr'],expected,rel_tol=1e-12,abs_tol=1e-15):
                raise ValueError('Parent has not reached the recorded terminal learning rate')
    origin=dict(parent_source_commit=manifest['source_commit'],parent_identity_hash=manifest['identity_hash'],
        parent_episodes=parent_n,target_episodes=identity['training']['episodes'],
        parent_checkpoint_sha256=sha(parent/'latest.pt'),parent_csv_sha256=sha(parent/'episode_returns.csv'),
        parent_env_steps=saved['env_steps'],lr_decay_episode_horizon=identity['training']['lr_decay_episode_horizon'],
        lr_rule='retain terminal LR, no schedule rewind',state_reset=False,
        learner_optimizer_normalizer_rng_retained=True,post_hoc_budget_extension=True)
    shutil.copytree(parent,out/'origin_prefix')
    shutil.copy2(parent/'episode_returns.csv',out/'episode_returns.csv')
    new_hash=digest(identity)
    json_write(out/'manifest.json',dict(identity,identity_hash=new_hash,execution_runtimes=[actual],
        source_commit=os.environ.get('GITHUB_SHA','local-component'),test_set_used=False,
        episode_definition='100 arrival slots plus full natural drainage',continuation=origin))
    converted=dict(saved,identity_hash=new_hash)
    checkpoint_write(out/'latest.pt',converted)
    json_write(out/'checkpoint_status.json',status)
    json_write(out/'continuation_origin.json',origin)
    return origin

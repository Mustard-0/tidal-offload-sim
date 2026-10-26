"""Explicit CPU platform/source adoption, with immutable parent provenance."""
import json
import shutil
import subprocess
from .runtime import ROOT, torch, source_hashes, runtime_info, json_write, checkpoint_write
from .protocol import configuration, digest
from .transport import sha


def adopt(folder, method, lr):
    folder = ROOT/folder
    old = json.loads((folder/'manifest.json').read_text())
    cfg, training = configuration(method,lr)
    current = dict(schema='lr-cell-20261025',method=method,lr=lr,environment=cfg,training=training,
        runtime=runtime_info('cpu'),source_hashes=source_hashes(),component_only=False)
    identity = digest(current)
    if old['identity_hash'] == identity:
        return
    contract = json.loads((ROOT/'cloud/contract.json').read_text())
    allowed_sources = [current['source_hashes'], contract['parent_source_hashes']]
    if old['source_hashes'] not in allowed_sources:
        raise ValueError('Unreviewed parent source')
    if (old['environment'] != cfg or old['training'] != training or old['method'] != method
            or old['lr'] != lr or old['component_only'] or old['runtime']['device'] != 'cpu'):
        raise ValueError('Parent scientific configuration or device differs')
    if (folder/'complete.json').exists():
        raise ValueError('Completed cell must not resume')
    raw = (folder/'latest.pt').read_bytes()
    checkpoint = torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    if checkpoint['identity_hash'] != old['identity_hash'] or checkpoint['rng']['cuda']:
        raise ValueError('Unknown checkpoint identity or CUDA RNG cannot be adopted as CPU')
    if checkpoint['update']*training['rollout_steps'] != checkpoint['rows'][-1]['env_steps']:
        raise ValueError('Checkpoint does not contain a whole update prefix')
    origin = folder/'origins'/sha(raw)
    origin.mkdir(parents=True,exist_ok=False)
    shutil.copy2(folder/'manifest.json',origin/'manifest.json')
    shutil.copy2(folder/'latest.pt',origin/'parent_checkpoint.pt')
    if (folder/'source').exists():
        shutil.move(str(folder/'source'),str(origin/'source'))
    # No model, optimizer, environment, CSV prefix or RNG field is replaced.
    checkpoint['identity_hash'] = identity
    checkpoint_write(folder/'latest.pt',checkpoint)
    for rel in current['source_hashes']:
        target = folder/'source'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/rel).read_bytes())
    commit = subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    json_write(folder/'manifest.json',dict(current,identity_hash=identity,source_commit=commit,
        transfer_integrated=True,test_set_used=False,status='formal',parent_checkpoint_sha256=sha(raw)))
    json_write(origin/'adoption.json',dict(parent_identity=old['identity_hash'],new_identity=identity,
        parent_checkpoint_sha256=sha(raw),parent_runtime=old['runtime'],actual_runtime=current['runtime'],
        update=checkpoint['update'],env_steps=checkpoint['update']*training['rollout_steps'],
        model_optimizer_environment_rng_csv_preserved=True,
        source_change='Hoist identical observation reductions; explicit portable transport only'))

"""One bounded Greedy pair per Linux runner; encrypted small summaries only."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from matched_pair import MODES, POPULATIONS, TRACES, simulate
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import pack, seal, sha, unseal

CAMPAIGN = 'greedy-paired-overlay-20261112'
BRANCH = 'experiment/'+CAMPAIGN
BASE = '/repos/'+REPOSITORY


def run(n, trace):
    assert sys.platform == 'linux' and sys.version_info[:3] == (3,9,25)
    assert os.environ['GITHUB_REPOSITORY'] == REPOSITORY
    assert os.environ['GITHUB_REF'] == 'refs/heads/'+BRANCH
    assert os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    assert n in POPULATIONS and trace in TRACES
    commit, run_id = os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID']
    key = bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    assert len(key)==32
    api = GitHub(os.environ['GH_TOKEN'])
    release = api.request('GET',BASE+'/releases/tags/'+CAMPAIGN)
    assert release['target_commitish'] == commit
    assets = {a['name']:a for a in release['assets']}
    packet = download(assets['inputs.enc'])
    raw,_ = unseal(packet,key,dict(campaign=CAMPAIGN,source_commit=commit,purpose='expected-inputs'))
    inputs = json.loads(raw)
    signature = tuple(inputs[f'{n}_{trace}'])
    assert len(inputs)==12 and len(signature)==2
    ident = f'N{n}_{trace}'
    name = ident+'.enc'
    assert name not in assets, 'No rerun or overwrite'
    header = dict(campaign=CAMPAIGN,source_commit=commit,run_id=run_id,attempt='1',
        cell=ident,inputs_sha256=sha(packet),purpose='matched-greedy-evaluation')
    rows=[]
    error=None
    for mode in MODES:
        try:
            rows.append(simulate((mode,n,trace,signature)))
        except Exception as e:
            error=dict(type=type(e).__name__,message=str(e))
            break
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        (folder/'outcome.json').write_text(json.dumps(dict(completed=error is None,
            error=error,scenes=rows),allow_nan=False),encoding='utf-8')
        result=seal(pack(folder),key,header)
    asset=api.upload(release['id'],name,result)
    assert asset['size']==len(result) and asset.get('digest')=='sha256:'+sha(result)
    print('Encrypted outcome saved:',ident,'complete' if error is None else 'failed',flush=True)
    return error is None


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--users',type=int,required=True)
    p.add_argument('--trace',type=int,required=True)
    a=p.parse_args()
    raise SystemExit(0 if run(a.users,a.trace) else 2)

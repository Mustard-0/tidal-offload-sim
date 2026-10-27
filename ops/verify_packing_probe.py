"""Verify a small encrypted component report, never a model checkpoint."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.packed_owner import ReadRetryGitHub
from ops.cloud_owner import assets,verify_asset,KEY_FILE,BASE
from ops.github_api import download
from ops.transport import unseal
from ops.local_owner import save


def main():
    d=json.loads((ROOT/'evidence/packing_probe_dispatch.json').read_text());api=ReadRetryGitHub()
    runs=api.request('GET',BASE+'/actions/runs?event=push&per_page=100')['workflow_runs']
    matches=[r for r in runs if r.get('head_branch')==d['tag']]
    if len(matches)!=1:print(json.dumps(dict(status='workflow_not_yet_registered')));return
    run=matches[0]
    if run['head_sha']!=d['source_commit'] or run['run_attempt']!=1:raise ValueError('Probe code or attempt mismatch')
    if run['status']!='completed':print(json.dumps(dict(status=run['status'],run_id=run['id'],url=run['html_url'])));return
    listing=assets(api,d['request']['release_id']);name='packing_benchmark.'+str(run['id'])+'.enc'
    if name+'.outcome.json' not in listing:raise ValueError('Probe ended without evidence; do not rerun blindly')
    data=download(listing[name+'.outcome.json']);verify_asset(listing[name+'.outcome.json'],data);r=json.loads(data)
    expected=dict(campaign=d['request']['campaign'],direction='packing_benchmark',source_commit=d['source_commit'],run_id=str(run['id']),attempt='1')
    if any(r[k]!=v for k,v in expected.items()):raise ValueError('Wrong probe origin')
    a=listing[name]
    if a['size']!=r['archive_size'] or a['digest']!='sha256:'+r['archive_sha256']:raise ValueError('Probe receipt/digest mismatch')
    if a['size']>1000000:raise ValueError('Unexpected large benchmark report')
    raw,_=unseal(download(a),bytes.fromhex(json.loads(KEY_FILE.read_text())['key']),expected)
    result=json.loads(raw);result.update(expected,github_conclusion=run['conclusion'],url=run['html_url'])
    save(ROOT/'evidence/packing_probe_report.json',result)
    if not result['passed'] or run['conclusion']!='success':raise ValueError('Actual performance/equivalence gate failed; retained report')
    save(ROOT/'evidence/packing_probe_verified.json',result)
    print(json.dumps({k:result[k] for k in ('passed','equivalence_passed','cpu_count','processes','throughput_speedup','timings','run_id','url')}))


if __name__=='__main__':main()

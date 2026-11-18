"""One CPU regression job; encrypted small evidence, no formal scale-up."""
import io
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

from review_protocol import BASE_COMMIT, BRANCH, CAMPAIGN, digest, plan
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import seal, sha, unseal

BASE = '/repos/' + REPOSITORY


def run():
    assert os.environ['GITHUB_REPOSITORY'] == REPOSITORY
    assert os.environ['GITHUB_REF'] == 'refs/heads/' + BRANCH
    assert os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    api = GitHub(os.environ['GH_TOKEN'])
    key = bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    commit, run_id = os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID']
    release = api.request('GET', BASE + '/releases/tags/' + CAMPAIGN)
    assert release['target_commitish'] == commit
    assets = {a['name']: a for a in release['assets']}
    assert 'preflight.enc' not in assets
    packet = download(assets['inputs.enc'])
    raw, _ = unseal(packet, key, dict(campaign=CAMPAIGN, source_commit=commit, purpose='review-inputs'))
    expected = json.loads(raw)
    assert expected['plan'] == json.loads(json.dumps(plan()))
    assert expected['source_base'] == BASE_COMMIT
    for path, h in expected['inherited_sha256'].items():
        assert sha(Path(path).read_bytes().replace(b'\r\n', b'\n')) == h, path
    for path, h in expected['new_sha256'].items():
        assert sha(Path(path).read_bytes().replace(b'\r\n', b'\n')) == h, path
    started = time.monotonic()
    value = dict(completed=False, performance_result=False, plan_hash=digest(expected['plan']))
    try:
        from review_checks import environment_checks, gradient_checks
        value.update(environment_checks())
        value.update(gradient_checks())
        value['completed'] = True
    except Exception as error:
        value['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    value['wall_seconds'] = time.monotonic() - started
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('preflight.json', json.dumps(value, allow_nan=False))
    header = dict(campaign=CAMPAIGN, source_commit=commit, run_id=run_id, attempt='1',
        inputs_sha256=sha(packet), purpose='review-small-outcome')
    result = seal(data.getvalue(), key, header)
    assert len(result) < 1024 * 1024
    asset = api.upload(release['id'], 'preflight.enc', result)
    assert asset['size'] == len(result) and asset['digest'] == 'sha256:' + sha(result)
    print('Regression evidence encrypted and saved. Completed:', value['completed'])
    return value['completed']


if __name__ == '__main__':
    raise SystemExit(0 if run() else 2)

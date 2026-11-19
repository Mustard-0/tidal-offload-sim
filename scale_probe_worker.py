"""Run bounded cloud-only pressure probes over authenticated pilot policies."""
import argparse
from collections import Counter, defaultdict
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import seal, sha, unseal
from pilot_protocol import settings
from pilot_worker import module_hash
from review_checks import bootstrap, check_metrics
from scale_diagnostics_environment import diagnostic_environment_types
import scale_probe_protocol as spec

BASE = '/repos/' + REPOSITORY


def packet(asset, cap):
    assert 0 < asset['size'] < cap
    raw = download(asset)
    assert len(raw) == asset['size'] and 'sha256:' + sha(raw) == asset['digest']
    return raw


def load_prior(api, key, cell):
    from experiments.lr_screen.runtime import torch
    release = api.request('GET', BASE + '/releases/tags/' + spec.PRIOR_CAMPAIGN)
    assert release['id'] == spec.PRIOR_RELEASE and release['target_commitish'] == spec.BASE_COMMIT
    assets = {a['name']: a for a in api.request('GET', BASE + f"/releases/{release['id']}/assets?per_page=100")}
    name = cell + '.weights.enc'
    data = packet(assets[name], 20 * 1024**2)
    raw, _ = unseal(data, key, dict(campaign=spec.PRIOR_CAMPAIGN, source_commit=spec.BASE_COMMIT,
        run_id=str(spec.PRIOR_RUN), attempt='1', inputs_sha256=spec.PRIOR_INPUTS,
        cell=cell, purpose='pilot-terminal-weights'))
    weights = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    assert weights['label'] == cell.rsplit('_seed', 1)[0] and weights['training_complete']
    assert str(weights['seed']) == cell.rsplit('_seed', 1)[1]
    return weights


def execute(api, key, cell, value, deadline):
    bootstrap()
    from experiments.lr_screen.runtime import torch, seed_all
    from pressure9.policies import evaluation_policy
    torch.set_num_threads(1)
    _, plain, tt, greedy, local = diagnostic_environment_types()
    rule = cell in spec.RULES
    name, seed = (cell, spec.DEVELOPMENT_SEEDS[0]) if rule else (cell.rsplit('_seed', 1)[0], int(cell.rsplit('_seed', 1)[1]))
    prior = None if rule else load_prior(api, key, cell)
    if prior:
        _, _, cfg, train = settings(name, seed, 150)
        assert prior['configuration'] == dict(environment=cfg, training=train)
    model_hash = None
    for scenario, (arrival, remote) in spec.FACTORS.items():
        for trace in spec.SEEDS:
            for n in spec.POPULATIONS:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Diagnostic budget exhausted before next scene')
                method, variant, cfg, train = settings(name, seed, n)
                assert not variant
                cfg['arrival_per_user_s'] *= arrival
                cfg['cpu_bs'] *= remote
                cfg['cpu_sat'] *= remote
                seed_all(trace)
                opts = dict(seed=trace, phase_s=spec.PHASE, measure_start=spec.WARMUP,
                    measure_end=spec.WARMUP + spec.MEASURE, collect=True)
                env = tt(cfg, **opts) if method == 'TMAPO' else plain(cfg, family=method, **opts)
                policy = ((local() if method == 'Local-Only' else greedy(cfg)) if rule else
                    evaluation_policy(method, train, cfg, prior['policy']))
                before = None if rule else module_hash(policy)
                if not rule:
                    if model_hash is None:
                        model_hash = before
                    assert model_hash == before
                stop = round((spec.WARMUP + spec.MEASURE) / cfg['dt_s'])
                with torch.no_grad():
                    for slot in range(stop + math.ceil(cfg['lifetime_s']/cfg['dt_s']) + 3):
                        env.step(policy, arrivals=slot < stop)
                        if env.slot >= stop and not env.tasks:
                            break
                m = check_metrics(env)
                assert len(env.rows) == m['arrivals'] and env.pressure_samples == round(spec.MEASURE/cfg['dt_s'])
                assert rule or module_hash(policy) == before
                routes = Counter('local' if r['node'] < n else 'remote' for r in env.rows if r['completed'])
                profiles = defaultdict(lambda: [0, 0])
                for r in env.rows:
                    profile = profiles[r['task_profile']]
                    profile[0] += 1
                    profile[1] += r['completed']
                resolutions = sorted(r['end_s'] - r['arrival_s'] for r in env.rows)
                value['scenes'].append(dict(scenario=scenario, method=method, training_seed=None if rule else seed,
                    trace_seed=trace, users=n, full_input_hash=env.full_input_hash, measure_s=spec.MEASURE,
                    model_hash=before, successful_routes=dict(routes), profile_arrivals_and_completed=dict(profiles),
                    p90_resolution_s=resolutions[math.ceil(.9*len(resolutions))-1], **m))
        print(f'{cell} finished scenario {scenario}', flush=True)


def run(cell):
    assert cell in spec.CELLS and os.environ['GITHUB_REPOSITORY'] == REPOSITORY
    assert os.environ['GITHUB_REF'] == 'refs/heads/' + spec.BRANCH and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    api, key = GitHub(os.environ['GH_TOKEN']), bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    commit, run_id = os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID']
    release = api.request('GET', BASE + '/releases/tags/' + spec.CAMPAIGN)
    assert release['target_commitish'] == commit
    assets = {a['name']: a for a in api.request('GET', BASE + f"/releases/{release['id']}/assets?per_page=100")}
    data = packet(assets['inputs.enc'], 2 * 1024**2)
    raw, _ = unseal(data, key, dict(campaign=spec.CAMPAIGN, source_commit=commit, purpose='review-inputs'))
    inputs = json.loads(raw)
    assert inputs['plan'] == json.loads(json.dumps(spec.plan()))
    paths = {**inputs['inherited_sha256'], **inputs['new_sha256'], **inputs['plan']['inherited_code']}
    for name, h in paths.items():
        assert sha(Path(name).read_bytes().replace(b'\r\n', b'\n')) == h, name
    value = dict(cell=cell, completed=False, scenes=[], plan_hash=spec.digest(inputs['plan']))
    try:
        execute(api, key, cell, value, time.monotonic() + spec.WALL_S)
        assert len(value['scenes']) == 36
        value['completed'] = True
    except Exception as error:
        value['error'] = dict(type=type(error).__name__, message=str(error))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('outcome.json', json.dumps(value, allow_nan=False))
    packet_out = seal(stream.getvalue(), key, dict(campaign=spec.CAMPAIGN, source_commit=commit,
        run_id=run_id, attempt='1', inputs_sha256=sha(data), cell=cell, purpose='scale-diagnostic-small'))
    asset = api.upload(release['id'], cell + '.enc', packet_out)
    assert asset['size'] == len(packet_out) and asset['digest'] == 'sha256:' + sha(packet_out)
    return value['completed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cell', choices=spec.CELLS)
    raise SystemExit(0 if run(parser.parse_args().cell) else 2)

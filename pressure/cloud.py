"""Bounded free CPU evaluation and authorized encrypted result delivery."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import sys

from .experiment import ROOT, run_scene, preflight, csv_write, json_write, source_hashes
from .protocol import plan, digest
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import seal, unseal, pack, unpack, sha

BASE = '/repos/' + REPOSITORY


def campaign():
    tag = os.environ['PRESSURE_CAMPAIGN']
    if not re.fullmatch(r'greedy-pressure-[a-z0-9-]{1,60}', tag):
        raise ValueError('Unexpected campaign')
    if os.environ['GITHUB_RUN_ATTEMPT'] != '1':
        raise ValueError('No reruns adopted')
    if len(bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])) != 32:
        raise ValueError('Missing existing key')
    return tag


def header(cell):
    return dict(campaign=campaign(), cell=cell, run_id=os.environ['GITHUB_RUN_ID'], attempt='1',
        source_commit=os.environ['GITHUB_SHA'], plan_hash=digest(plan()),
        source_hashes_hash=digest(source_hashes()))


def api():
    return GitHub(os.environ['GH_TOKEN'])


def release():
    return api().request('GET', BASE + '/releases/tags/' + campaign())


def upload(name, data):
    return api().upload(release()['id'], name, data)


def encrypted(folder, cell):
    packet = seal(pack(folder), bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']), header(cell))
    upload(cell + '.enc', packet)
    receipt = dict(header(cell), archive_name=cell+'.enc', archive_sha256=sha(packet), archive_size=len(packet))
    upload(cell+'.receipt.json', json.dumps(receipt).encode())


def gate():
    campaign()
    if sys.platform != 'linux' or sys.version_info[:3] != (3,9,25):
        raise RuntimeError('Expected standard Linux CPU Python3.9.25')
    repo = api().request('GET', BASE)
    if repo['private']:
        raise RuntimeError('Refuse potentially paid private Actions')
    api().request('POST', BASE+'/releases', dict(tag_name=campaign(), target_commitish=os.environ['GITHUB_SHA'],
        name=campaign(), prerelease=True, draft=False,
        body='Finite rule-only common-environment pressure probe. Pure code; encrypted simulation outputs only.'))
    folder = ROOT/'runs/preflight'
    proof = preflight(folder)
    encrypted(folder, 'preflight-private')
    upload('preflight.json', json.dumps(dict(header('preflight'), passed=proof['passed'],
        tests_run=proof['tests_run'], torch_imported=False, training=False)).encode())
    print('Rule-only source and streaming contracts passed', flush=True)


def worker(cell):
    spec = next(c for c in plan()['cells'] if c['id'] == cell)
    asset = next(a for a in release()['assets'] if a['name'] == 'preflight.json')
    proof = json.loads(download(asset))
    assert proof['passed'] and all(proof[k] == v for k,v in header('preflight').items())
    out = ROOT/'runs/raw'/cell
    out.mkdir(parents=True, exist_ok=False)
    summary = ROOT/'runs/summaries'/cell
    summary.mkdir(parents=True, exist_ok=False)
    rows = []
    upload(cell+'.started.json', json.dumps(dict(header(cell), process_started=True,
        preflight_passed=True, training=False, cpu_count=os.cpu_count())).encode())
    failure = None
    try:
        for trace in plan()['traces']:
            for scene in plan()['scenes']:
                folder = out/f"t{trace}_{scene['name']}"
                value = run_scene(folder, spec['method'], spec['users'], spec['rate'], trace,
                                  scene['warmup_s'], scene['measure_s'], maximum_seconds=1800)
                rows.append(value)
                csv_write(summary/'metrics.csv', rows)
                print('Bounded scene saved', flush=True)
        json_write(summary/'complete.json', dict(completed=True, cell=cell, scenes=len(rows), training=False))
    except BaseException as error:
        failure = error
        json_write(out/'failure.json', dict(type=type(error).__name__, message=str(error)))
    encrypted(out, cell+'-raw')
    encrypted(summary, cell+'-summary')
    upload(cell+'.outcome.json', json.dumps(dict(header(cell), state='failed' if failure else 'completed',
        child_exited=True, raw_archive=cell+'-raw.enc', summary_archive=cell+'-summary.enc', training=False)).encode())
    if failure:
        raise RuntimeError('Finite worker failed; encrypted evidence retained; no rerun') from failure


def collect():
    assets = release()['assets']
    rows = []
    for cell in (c['id'] for c in plan()['cells']):
        asset = next(a for a in assets if a['name'] == cell+'-summary.enc')
        packet = download(asset)
        raw, meta = unseal(packet, bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']), header(cell+'-summary'))
        folder = ROOT/'runs/received'/cell
        unpack(raw, folder)
        if not (folder/'complete.json').exists():
            raise RuntimeError('Incomplete probe grid; no complete results claimed')
        with (folder/'metrics.csv').open(newline='') as f:
            rows.extend(csv.DictReader(f))
    assert len(rows) == 48
    paired = {}
    for row in rows:
        key = tuple(row[k] for k in ('users', 'arrival_rate_per_user_s', 'trace', 'scene'))
        sig = row['full_input_hash'], row['trace_hash'], row['arrivals']
        assert paired.setdefault(key, sig) == sig, 'Methods did not receive the same task input'
        assert row['all_tasks_resolved'] == 'True'
    out = ROOT/'runs/comparison'
    out.mkdir(parents=True, exist_ok=False)
    csv_write(out/'scene_metrics.csv', rows)
    from .report import report
    report(rows, out)
    json_write(out/'complete.json', dict(completed=True, cells=12, scenes=48, training=False,
        paired_full_inputs=True, plan=plan(), plan_hash=digest(plan()), source_commit=os.environ['GITHUB_SHA'],
        runtime='Python3.9.25/NumPy2.0.2/SciPy1.13.1, original environment, no Torch'))
    encrypted(out, 'comparison')
    upload('campaign.completed.json', json.dumps(dict(header('comparison'), completed=True,
        cells=12, scenes=48, training=False)).encode())
    print('Finite common pressure probe complete; encrypted figures published', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['gate','worker','collect'])
    parser.add_argument('--cell')
    args = parser.parse_args()
    if args.mode == 'gate':
        gate()
    elif args.mode == 'worker':
        worker(args.cell)
    else:
        collect()

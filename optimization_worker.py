"""One source-frozen cloud gate and at most two normal training parts."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

from optimization_protocol import (ADMISSION_STEPS, BASE_COMMIT, BRANCH, CAMPAIGN, CELLS,
    CHECKPOINT_UPDATES, TOTAL_STEPS, digest, plan)
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import seal, sha, unseal

BASE = '/repos/' + REPOSITORY


class Transport:
    def __init__(self):
        assert os.environ['GITHUB_REPOSITORY'] == REPOSITORY
        assert os.environ['GITHUB_REF'] == 'refs/heads/' + BRANCH
        assert os.environ['GITHUB_RUN_ATTEMPT'] == '1'
        self.api = GitHub(os.environ['GH_TOKEN'])
        self.key = bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
        self.commit, self.run_id = os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID']
        self.release = self.api.request('GET', BASE + '/releases/tags/' + CAMPAIGN)
        assert self.release['target_commitish'] == self.commit
        self.assets = {a['name']: a for a in self.api.request('GET', BASE + '/releases/' + str(self.release['id']) + '/assets?per_page=100')}
        packet = download(self.assets['inputs.enc'])
        self.inputs_sha256 = sha(packet)
        raw, _ = unseal(packet, self.key, dict(campaign=CAMPAIGN, source_commit=self.commit, purpose='review-inputs'))
        self.inputs = json.loads(raw)
        assert self.inputs['plan'] == json.loads(json.dumps(plan()))
        for path, h in {**self.inputs['inherited_sha256'], **self.inputs['new_sha256'], **self.inputs['plan']['inherited_review_hashes']}.items():
            assert sha(Path(path).read_bytes().replace(b'\r\n', b'\n')) == h, path
        self.plan_hash = digest(self.inputs['plan'])

    def header(self, cell, purpose):
        return dict(campaign=CAMPAIGN, source_commit=self.commit, run_id=self.run_id,
            attempt='1', inputs_sha256=self.inputs_sha256, cell=cell, purpose=purpose)

    def upload(self, name, raw, cell, purpose, limit):
        assert name not in self.assets and len(raw) < limit
        packet = seal(raw, self.key, self.header(cell, purpose))
        asset = self.api.upload(self.release['id'], name, packet)
        assert asset['size'] == len(packet) and asset['digest'] == 'sha256:' + sha(packet)
        self.assets[name] = asset
        return dict(name=name, size=asset['size'], digest=asset['digest'])

    def get(self, name, cell, purpose, limit):
        a = self.assets[name]
        assert 0 < a['size'] < limit
        packet = download(a)
        assert a['digest'] == 'sha256:' + sha(packet)
        raw, _ = unseal(packet, self.key, self.header(cell, purpose))
        return raw

    def small(self, name, value, cell):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('outcome.json', json.dumps(dict(value, plan_hash=self.plan_hash), allow_nan=False))
        return self.upload(name, data.getvalue(), cell, 'optimization-small-outcome', 2 * 1024**2)

    def read_small(self, name, cell):
        raw = self.get(name, cell, 'optimization-small-outcome', 2 * 1024**2)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            assert z.namelist() == ['outcome.json'] and z.testzip() is None
            assert z.getinfo('outcome.json').file_size < 2 * 1024**2
            value = json.loads(z.read('outcome.json'))
        assert value['plan_hash'] == self.plan_hash
        return value


def within_dispatch_window(t):
    from datetime import datetime, timezone
    created = datetime.fromisoformat(t.release['created_at'].replace('Z', '+00:00'))
    return (datetime.now(timezone.utc) - created).total_seconds() <= 24 * 3600


def train_part(t, cell, part):
    from optimization_core import (runtime, make, restore, serialize, deserialize,
                                   train_block, validate, snapshot)
    np, torch, seed_all, rng_state, rng_load, info, learner, weights, set_progress, tm = runtime()
    variant, seed = cell.rsplit('_seed', 1)
    seed = int(seed)
    actual = info('cpu')
    identity = dict(cell=cell, source_commit=t.commit, plan_hash=t.plan_hash,
        runtime={k:actual[k] for k in ('python','numpy','scipy','torch','device','torch_threads')})
    value = dict(cell=cell, part=part, status='failed', completed=False, formal=False,
        updates=0, env_steps=0, training_rows=[], validation_rows=[])
    if not within_dispatch_window(t):
        value.update(status='not_started_budget_window', reason='24-hour stage-entry limit')
        return value
    if part == 2:
        prior = t.read_small(cell + '.part1.enc', cell + '.part1')
        assert prior['status'] in ('completed', 'paused')
        if prior['completed']:
            return dict(prior, part=2, status='no_new_training', referenced_part=1,
                referenced_part_wall_seconds=prior['wall_seconds'], wall_seconds=0.)
        checkpoint = prior['checkpoint']
        assert t.assets[checkpoint['name']]['digest'] == checkpoint['digest']
        saved = deserialize(t.get(checkpoint['name'], cell + '.part1', 'optimization-checkpoint', 30 * 1024**2))
        e, p, rows, validation = restore(saved, variant, seed, identity)
    else:
        e, p = make(variant, seed)
        rows, validation = [], []
    started = time.monotonic()
    try:
        deadline = started + 8700  # Reserve time to upload the normal boundary snapshot.
        def validate_here():
            current = e.slot // 100
            if current not in CHECKPOINT_UPDATES:
                return True
            existing = [r for r in validation if r['update']==current]
            if len(existing)==8:
                return True
            fresh = validate(p, variant, seed, current, existing, deadline)
            validation[:] = [r for r in validation if r['update']!=current] + fresh
            return len(fresh)==8
        can_continue = validate_here()
        while can_continue and e.slot < TOTAL_STEPS and time.monotonic() < deadline:
            rows.append(train_block(e, p))
            update = e.slot // 100
            if update in CHECKPOINT_UPDATES:
                can_continue = validate_here()
            if update in (1, 151):
                name = cell + '.u' + str(update) + '.enc'
                if name not in t.assets:
                    t.small(name, dict(cell=cell, part=part, update=update, env_steps=e.slot,
                        real_training=True, component_only=False), cell + '.u' + str(update))
            if time.monotonic() >= deadline:
                break
        done = e.slot == TOTAL_STEPS and len(validation)==24
        if done:
            from review_checks import check_metrics
            check_metrics(e)
        value.update(status='completed' if done else 'paused', completed=done,
            updates=e.slot // 100, env_steps=e.slot, training_rows=rows,
            validation_rows=validation, training_input_hash=e.full_input_hash,
            wall_seconds=time.monotonic()-started,
            pending_tasks=len(e.tasks), maximum_wait_s=e.max_observed_wait)
    except Exception as error:
        value.update(error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()),
            updates=e.slot // 100, env_steps=e.slot, training_rows=rows, validation_rows=validation)
    # Failed states are preserved but never eligible for continuation.
    checkpoint = serialize(snapshot(e, p, identity, rows, validation)) if e.slot % 100 == 0 and not p.buffer else None
    if checkpoint is not None:
        value['checkpoint'] = t.upload(cell + '.part' + str(part) + '.checkpoint.enc', checkpoint,
            cell + '.part' + str(part), 'optimization-checkpoint', 30 * 1024**2)
    return value


def run(action, cell=None, part=None):
    t = Transport()
    if action == 'prepare':
        pending = []
        for name in CELLS:
            prior = t.read_small(name + '.part1.enc', name + '.part1')
            assert prior['status'] in ('completed','paused')
            if not prior['completed']:
                assert 'checkpoint' in prior
                pending.append(name)
        t.small('resume_plan.enc', dict(paused_cells=pending, completed_cells_referenced=[c for c in CELLS if c not in pending]), 'resume-plan')
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
            f.write('has_paused=' + ('true' if pending else 'false') + '\n')
            f.write('matrix=' + json.dumps(dict(cell=pending),separators=(',',':')) + '\n')
        print('Normal paused states eligible for one continuation:', len(pending))
        return True
    if action == 'gate':
        try:
            from optimization_core import runtime, gate
            runtime()
            value = gate()
        except Exception as error:
            value = dict(passed=False, error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        t.small('gate.enc', value, 'gate')
        print('Source-frozen gate evidence saved:', value['passed'])
        return value['passed']
    assert cell in CELLS and part in (1, 2)
    assert t.read_small('gate.enc', 'gate')['passed']
    value = train_part(t, cell, part)
    t.small(cell + '.part' + str(part) + '.enc', value, cell + '.part' + str(part))
    print('Bounded development part saved:', cell, part, value['status'])
    return value['status'] in ('completed', 'paused', 'no_new_training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gate','train','prepare'))
    parser.add_argument('--cell', choices=CELLS)
    parser.add_argument('--part', type=int, choices=(1,2))
    args = parser.parse_args()
    raise SystemExit(0 if run(args.action,args.cell,args.part) else 2)

"""Fresh bounded pilot cells with encrypted summaries and terminal weights."""
import io
import json
import math
import os
from pathlib import Path
import time
import traceback
import zipfile

from pilot_protocol import BRANCH, CAMPAIGN, CELLS, DEVELOPMENT_SEEDS, DEVELOPMENT_TRACES, POPULATIONS, RULES, plan, settings
from review_protocol import digest
from review_checks import bootstrap, check_metrics
from experiments.lr_screen.github_api import GitHub, REPOSITORY, download
from experiments.lr_screen.transport import seal, sha, unseal

BASE = '/repos/' + REPOSITORY


def module_hash(p):
    nets = (p.actors, p.critic) if hasattr(p, 'actors') else (p.actor, p.q)
    return sha(b''.join(v.detach().cpu().numpy().tobytes() for net in nets for v in net.parameters()))


def execute(cell, started, value, save_weights):
    bootstrap()
    from experiments.lr_screen.runtime import np, torch, seed_all
    from pressure9.policies import learner, update, weights_only, set_progress
    from review_environment import environment_types
    Plain, TM, Greedy, Local = environment_types()
    torch.set_num_threads(1)
    rule = cell in RULES
    label, seed = (cell, DEVELOPMENT_SEEDS[0]) if rule else (cell.rsplit('_seed', 1)[0], int(cell.rsplit('_seed', 1)[1]))
    method, variant, c, t = settings(label, seed, 150)
    seed_all(seed)
    p = (Local() if method == 'Local-Only' else Greedy(c)) if rule else learner(method, t, c)
    value.update(method=label, training_seed=None if rule else seed, configuration=dict(environment=c, training=t),
        training_complete=rule, training_rows=[], evaluation_rows=[], missing_scenes=[])
    if not rule:
        e = TM(c, seed=1191900 + seed, ablation=variant) if method == 'TMAPO' else Plain(c, family=method, seed=1191900 + seed)
        before = module_hash(p)
        block_return, update_number = 0., 0
        try:
            for slot in range(5003):
                reward, records, *_ = e.step(p, arrivals=slot < 4800)
                block_return += float(reward)
                done = e.slot >= 4800 and not e.tasks
                p.remember(e.agent_rewards(t['local_credit_fraction']), records, done)
                if e.slot % 100 == 0 or done:
                    update_number += 1
                    set_progress(p, update_number, 1208)
                    context, glob, *_ = e.observations()
                    stats = update(p, context, glob)
                    assert all(math.isfinite(float(v)) for v in stats.values())
                    value['training_rows'].append(dict(update=update_number, env_steps=e.slot,
                        block_return=block_return, **stats))
                    block_return = 0.
                    if time.monotonic() - started > 5100:
                        value['stop_reason'] = 'bounded pilot wall budget during training'
                        return
                if done:
                    check_metrics(e)
                    value.update(training_complete=True, training_env_steps=e.slot,
                        training_input_hash=e.full_input_hash, model_hash=module_hash(p))
                    assert value['model_hash'] != before
                    break
            else:
                raise AssertionError('Training cohort failed to drain')
        finally:
            data = io.BytesIO()
            torch.save(dict(method=method, label=label, seed=seed, policy=weights_only(p),
                training_complete=value['training_complete'], configuration=value['configuration']), data)
            save_weights(data.getvalue())
        p.deterministic = True
        if hasattr(p, 'collect_transitions'):
            p.collect_transitions = False
    expected = {(n, trace) for n in POPULATIONS for trace in DEVELOPMENT_TRACES}
    for n in POPULATIONS:
        for trace in DEVELOPMENT_TRACES:
            method, variant, cfg, _ = settings(label, seed, n)
            e = TM(cfg, seed=trace, ablation=variant, measure_start=5., measure_end=35.) if method == 'TMAPO' else Plain(cfg, family=method, seed=trace, measure_start=5., measure_end=35.)
            if not rule:
                p.reset_history()
                before = module_hash(p)
            else:
                p = Local() if method == 'Local-Only' else Greedy(cfg)
            with torch.no_grad():
                for slot in range(3703):
                    e.step(p, arrivals=slot < 3500)
                    if e.slot >= 3500 and not e.tasks:
                        m = check_metrics(e)
                        break
                else:
                    raise AssertionError('Evaluation cohort failed to drain')
            if not rule:
                assert module_hash(p) == before
            value['evaluation_rows'].append(dict(method=label, training_seed=None if rule else seed,
                users=n, trace_seed=trace, full_input_hash=e.full_input_hash,
                warmup_s=5., measure_s=30., phase_s=0., **m))
            expected.remove((n, trace))
            if time.monotonic() - started > 5100 and expected:
                value['stop_reason'] = 'bounded pilot wall budget during evaluation'
                value['missing_scenes'] = sorted(expected)
                return
    value['completed'] = True


def run(cell):
    assert cell in CELLS and os.environ['GITHUB_REPOSITORY'] == REPOSITORY
    assert os.environ['GITHUB_REF'] == 'refs/heads/' + BRANCH and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    api = GitHub(os.environ['GH_TOKEN'])
    key = bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY'])
    commit, run_id = os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID']
    release = api.request('GET', BASE + '/releases/tags/' + CAMPAIGN)
    assert release['target_commitish'] == commit
    assets = {a['name']: a for a in release['assets']}
    assert cell + '.enc' not in assets
    packet = download(assets['inputs.enc'])
    raw, _ = unseal(packet, key, dict(campaign=CAMPAIGN, source_commit=commit, purpose='review-inputs'))
    inputs = json.loads(raw)
    assert inputs['plan'] == json.loads(json.dumps(plan()))
    for path, h in {**inputs['inherited_sha256'], **inputs['new_sha256'], **inputs['plan']['review_code_sha256']}.items():
        assert sha(Path(path).read_bytes().replace(b'\r\n', b'\n')) == h
    header = dict(campaign=CAMPAIGN, source_commit=commit, run_id=run_id, attempt='1', cell=cell,
        inputs_sha256=sha(packet), purpose='pilot-small-outcome')
    def weights(raw_weights):
        assert len(raw_weights) < 20 * 1024 * 1024
        data = seal(raw_weights, key, dict(header, purpose='pilot-terminal-weights'))
        a = api.upload(release['id'], cell + '.weights.enc', data)
        assert a['size'] == len(data) and a['digest'] == 'sha256:' + sha(data)
    started = time.monotonic()
    value = dict(completed=False, exploratory=True, formal=False, plan_hash=digest(inputs['plan']))
    try:
        execute(cell, started, value, weights)
    except Exception as error:
        value['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    value['wall_seconds'] = time.monotonic() - started
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('outcome.json', json.dumps(value, allow_nan=False))
    data = seal(output.getvalue(), key, header)
    assert len(data) < 2 * 1024 * 1024
    a = api.upload(release['id'], cell + '.enc', data)
    assert a['size'] == len(data) and a['digest'] == 'sha256:' + sha(data)
    print('Finite pilot evidence encrypted and saved:', cell, value['completed'])
    return value['completed']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cell', choices=CELLS)
    raise SystemExit(0 if run(parser.parse_args().cell) else 2)

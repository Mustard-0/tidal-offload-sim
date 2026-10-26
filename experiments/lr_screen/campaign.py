"""Finite serial CPU queue: 5 TMAPO, 3 IPPO, 3 PPO; no other studies.

Each child exits after at most one hour at a saved update boundary. Only known
wall-budget pauses resume automatically. Failures and explicit stops never do.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .runtime import ROOT, json_write, source_hashes, available_ram_gib, torch
from .protocol import plan, digest
from .readiness import require_transfer


def run(root):
    torch.set_num_threads(1)
    approval = require_transfer()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'campaign_status.json').exists():
        raise RuntimeError('Existing campaign requires manual state review; no blind second owner')
    lock = root/'campaign.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    started = time.time()
    p = plan()
    frozen_hashes = source_hashes()
    state = dict(pid=os.getpid(), state='starting', started_epoch=started,
                 max_wall_hours=168, completed_cells=[], current=None, child_pid=None,
                 plan_hash=digest(p), source_hashes=frozen_hashes, device='cpu', worker_concurrency=1)
    try:
        os.write(descriptor, json.dumps(dict(pid=os.getpid(), started_epoch=started)).encode())
        json_write(root/'plan.json', p)
        json_write(root/'accepted_preflight.json', approval)
        json_write(root/'campaign_status.json', state)
        for cell in p['cells']:
            out = root/cell['id']
            state.update(current=cell['id'], state='waiting_for_memory')
            # Three spaced samples before a new model; no killing user programs.
            stable, waited = 0, time.monotonic()
            while stable < 3:
                if (root/'STOP_AFTER_UPDATE').exists():
                    state['state'] = 'paused_explicit_stop'
                    return
                free = available_ram_gib()
                stable = stable+1 if free >= 1.5 else 0
                state['available_ram_gib'] = free
                json_write(root/'campaign_status.json', state)
                if time.monotonic()-waited > 12*3600:
                    raise RuntimeError('12-hour resource wait exhausted')
                time.sleep(10)
            attempt = 0
            while True:
                if source_hashes() != frozen_hashes or digest(plan()) != state['plan_hash']:
                    raise RuntimeError('Source or protocol changed during campaign')
                if (root/'STOP_AFTER_UPDATE').exists():
                    state['state'] = 'paused_explicit_stop'
                    return
                if time.time()-started > 168*3600:
                    state['state'] = 'paused_campaign_wall_budget'
                    return
                attempt += 1
                if attempt > 96:
                    raise RuntimeError('Finite per-cell segment limit reached')
                cmd = [sys.executable, '-u', '-m', 'experiments.lr_screen.runner', 'run',
                       '--out', str(root), '--method', cell['algorithm'], '--lr', str(cell['lr_index']),
                       '--device', 'cpu', '--max-seconds', '3600']
                if attempt > 1:
                    cmd.append('--resume')
                state.update(state='running', segment=attempt)
                logdir = root/'logs'
                logdir.mkdir(exist_ok=True)
                with (logdir/f"{cell['id']}.{attempt:03d}.log").open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    state['child_pid'] = child.pid
                    json_write(root/'campaign_status.json', state)
                    code = child.wait()
                state['child_pid'] = None
                state['child_exit_code'] = code
                if code != 0 or (out/'failure.json').exists():
                    raise RuntimeError(f"Cell failed: {cell['id']}, exit {code}; no automatic retry")
                if (out/'complete.json').exists():
                    done = json.loads((out/'complete.json').read_text())
                    if done['env_steps'] != cell['env_steps'] or done['component_only']:
                        raise RuntimeError('Invalid complete receipt')
                    state['completed_cells'].append(cell['id'])
                    from .report import report
                    report(root)
                    break
                paused = json.loads((out/'paused.json').read_text())
                if paused['reason'] != 'wall budget':
                    state.update(state='paused', reason=paused['reason'])
                    return
        state.update(state='completed', current=None)
    except BaseException as exc:
        state.update(state='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        state['updated_epoch'] = time.time()
        json_write(root/'campaign_status.json', state)
        os.close(descriptor)
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)

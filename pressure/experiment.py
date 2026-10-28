"""Execute original NumPy physics without importing Torch or altering its body."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT if (ROOT / 'model/environment.py').exists() else ROOT.parent / 'tmapo_credit_review_20261028'
sys.path[:0] = [str(SOURCE), str(SOURCE / 'model'),
               'F:/stin_project/tmapo_lr_speed_20261026/.runtime']
stub = types.ModuleType('experiments.lr_screen.runtime')
stub.np = np
sys.modules[stub.__name__] = stub
from study.environment import Environment
from study.geoi import GeoIEnvironment
from study.protocol import configuration, source_hashes
from environment import GreedyPolicy
from .protocol import plan, digest


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def configured(users, rate):
    cfg, _ = configuration('TMAPO', n=users)
    cfg['arrival_per_user_s'] = float(rate)
    return cfg


class StreamingMixin:
    """Archive completed raw records incrementally instead of accumulating RAM."""
    def open_streams(self, folder):
        self.folder = folder
        self.streams = {}
        self.input_fingerprint = hashlib.sha256()
        self.raw_cost = self.raw_energy = self.raw_delay = self.queue_delay = 0.
        self.raw_qos = self.raw_completed = self.raw_count = 0
        self.paths = dict(local=0, bs=0, satellite=0)
        self.energy_parts = dict(compute_energy_j=0., transmit_energy_j=0., report_energy_j=0.)

    def generate(self):
        first = self.serial
        super().generate()
        for task_id in range(first, self.serial):
            j = self.tasks[task_id]
            fields = [j.id, j.user, j.region, j.arrival_slot, j.arrival,
                      j.bits, j.cycles, j.deadline, j.profile]
            self.input_fingerprint.update(json.dumps(fields, separators=(',', ':')).encode() + b'\n')

    def flush_rows(self):
        for name, values in (('tasks', self.rows), ('batches', self.batch_rows), ('waits', self.wait_decision_rows)):
            if not values:
                continue
            if name not in self.streams:
                f = (self.folder / (name + '.csv')).open('w', newline='', encoding='utf-8')
                writer = csv.DictWriter(f, list(values[0]))
                writer.writeheader()
                self.streams[name] = f, writer
            f, writer = self.streams[name]
            writer.writerows(values)
            f.flush()
            if name == 'tasks':
                for r in values:
                    duration = r['end_s'] - r['arrival_s']
                    energy = sum(r[k] for k in self.energy_parts)
                    assert math.isclose(energy, r['energy_j'], abs_tol=1e-9)
                    assert r['qos'] == int(not r['completed'] or duration > r['deadline_s'] + 1e-9)
                    assert r['active_wait_s'] == 0.
                    self.raw_count += 1
                    self.raw_energy += energy
                    self.raw_delay += duration * r['completed']
                    self.raw_completed += r['completed']
                    self.raw_qos += r['qos']
                    self.raw_cost += .25 * duration / .5 + .25 * energy / .24 + r['qos']
                    self.queue_delay += r['compute_queue_s']
                    for key in self.energy_parts:
                        self.energy_parts[key] += r[key]
                    kind = 'local' if r['node'] < self.u else 'bs' if r['node'] < self.u + 8 else 'satellite'
                    self.paths[kind] += 1
            elif name == 'batches':
                for r in values:
                    users = list(map(int, r['user_ids'].split(';')))
                    assert len(users) == len(set(users)) == r['n_distinct'] <= self.region_counts[r['region']]
            values.clear()

    def close_streams(self):
        for f, writer in self.streams.values():
            f.close()


class StreamingEnvironment(StreamingMixin, Environment):
    pass


class StreamingGeoIEnvironment(StreamingMixin, GeoIEnvironment):
    pass


def run_scene(folder, method, users, rate, trace, warmup, measure, maximum_seconds=1800, streaming=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    cfg = configured(users, rate)
    cls = (StreamingEnvironment if method == 'Greedy' else StreamingGeoIEnvironment) if streaming else (
           Environment if method == 'Greedy' else GeoIEnvironment)
    env = cls(cfg, seed=trace, phase_s=0., measure_start=warmup, measure_end=warmup+measure,
              family=method, collect=True)
    if streaming:
        env.open_streams(folder)
    stop = round((warmup + measure) / cfg['dt_s'])
    env.arrival_stop_slot = stop
    policy = GreedyPolicy(cfg)
    started = time.monotonic()
    try:
        for slot in range(stop + 203):
            reward, _, _, _ = env.step(policy, arrivals=slot < stop)
            assert math.isfinite(reward)
            if streaming and slot % 100 == 99:
                env.flush_rows()
            if slot % 1000 == 999:
                json_write(folder / 'status.json', dict(slot=env.slot, admission_slots=stop,
                    pending_tasks=len(env.tasks), elapsed_s=time.monotonic()-started))
            if time.monotonic() - started > maximum_seconds:
                raise TimeoutError('Finite scene time budget reached; no automatic rerun')
            if env.slot >= stop and not env.tasks:
                break
        else:
            raise AssertionError('Natural settlement exceeded original lifetime')
        assert env.arrivals == env.ended and not env.tasks
        if streaming:
            env.flush_rows()
        metrics = env.metrics()
        assert metrics['qos_rate'] == metrics['late_completed_rate'] + metrics['rejected_rate'] + metrics['expired_rate'] or math.isclose(
            metrics['qos_rate'], metrics['late_completed_rate']+metrics['rejected_rate']+metrics['expired_rate'], abs_tol=1e-10)
        extra = {}
        if streaming:
            assert env.raw_count == metrics['arrivals']
            for a, b in ((env.raw_energy/env.raw_count, metrics['energy_j_per_arrival']),
                         (env.raw_cost/env.raw_count, metrics['mean_task_cost']),
                         (env.raw_qos/env.raw_count, metrics['qos_rate'])):
                assert math.isclose(a, b, abs_tol=1e-8)
            if env.raw_completed:
                assert math.isclose(env.raw_delay/env.raw_completed, metrics['delay_s'], abs_tol=1e-8)
            extra = dict(full_input_hash=env.input_fingerprint.hexdigest(),
                queue_ms=env.queue_delay/env.raw_count*1000,
                total_cohort_energy_j=env.raw_energy,
                **{key+'_fraction': value/env.raw_count for key, value in env.paths.items()},
                **{'raw_'+key+'_per_arrival': value/env.raw_count for key, value in env.energy_parts.items()})
        else:
            csv_write(folder / 'tasks.csv', env.rows)
        value = dict(method=method, users=users, arrival_rate_per_user_s=rate, trace=trace,
            warmup_s=warmup, measure_s=measure, drain_s=env.slot*cfg['dt_s']-(warmup+measure),
            phase_s=0., elapsed_s=time.monotonic()-started, all_tasks_resolved=True,
            scene='short_empty' if measure==1. else 'long_cycle', **metrics, **extra)
        json_write(folder / 'metrics.json', value)
        return value
    finally:
        if streaming:
            env.close_streams()


class Contracts(unittest.TestCase):
    def test_only_load_config_changes(self):
        base, _ = configuration('TMAPO', n=100)
        changed = configured(100, 9)
        self.assertEqual([k for k in base if base[k] != changed[k]], ['arrival_per_user_s'])

    def test_fixed_source(self):
        self.assertEqual(digest(source_hashes()), plan()['source_hashes_hash'])

    def test_actual_population(self):
        self.assertEqual(configured(100, 6)['region_user_counts'], [34,33,33])
        self.assertEqual(configured(150, 6)['region_user_counts'], [50,50,50])

    def test_streamed_physics_equal(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            for method in ('Greedy', 'GeoI-Greedy'):
                a = run_scene(p/(method+'_stream'), method, 100, 3, 7298101, 0, .1, streaming=True)
                b = run_scene(p/(method+'_original'), method, 100, 3, 7298101, 0, .1, streaming=False)
                for key in ('arrivals','completed','qos_rate','delay_s','energy_j_per_arrival','trace_hash'):
                    self.assertEqual(a[key], b[key])


def preflight(folder):
    assert 'torch' not in sys.modules
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Stress contracts failed')
    speed = []
    for method in ('Greedy', 'GeoI-Greedy'):
        row = run_scene(Path(folder)/method, method, 150, 9, 7298199, 0., 2., maximum_seconds=120)
        # Component-only trace, not a result-selection mechanism.
        speed.append(dict(method=method, simulated_s=2., wall_s=row['elapsed_s']))
    value = dict(passed=True, tests_run=result.testsRun, speed=speed, torch_imported=False,
                 source_hashes_hash=digest(source_hashes()), plan_hash=digest(plan()))
    json_write(Path(folder)/'preflight_private.json', value)
    return value

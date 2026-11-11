"""Re-evaluate the Greedy pair on the archived night-review inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from qos95 import StrictQoSMixin

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT if (ROOT/'model/config.json').is_file() else ROOT.parent/'tmapo_night_review_20261029'

POPULATIONS = (20, 40, 60, 80, 100, 125)
TRACES = (9792101, 9792109)
MODES = ('Greedy-NoPrivacy', 'Greedy-GeoI')
REFERENCE = SOURCE / 'results_final_partial'
OUT = ROOT / 'runs/matched_original_20261112'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def source_hashes():
    paths = {Path(__file__), ROOT/'qos95.py', ROOT/'evaluate_pair.py', SOURCE/'model/config.json'}
    for folder in ('model', 'study', 'pressure9', 'experiments/lr_screen'):
        paths.update((SOURCE/folder).glob('*.py'))
    paths.update(REFERENCE/f for f in ('scene_metrics_all_saved.csv', 'physical_metrics_common.csv'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def controller_environment(base, np):
    class SameGreedy(StrictQoSMixin, base):
        def __init__(self, cfg, *, geoi, **kwargs):
            self.geoi = geoi
            cfg = dict(cfg)
            if not geoi:
                cfg.update(privacy_processing_s=0., report_energy_j=0.)
            super().__init__(cfg, **kwargs)

        def q_estimate(self, n):
            return super().q_estimate(n) if self.geoi else 0.

        def release(self, tasks, reason='immediate_greedy'):
            if not tasks:
                return
            n, region = len(tasks), tasks[0].region
            if (len({(j.region, j.profile) for j in tasks}) != 1
                    or len({j.user for j in tasks}) != n
                    or len({j.id for j in tasks}) != n
                    or n > self.region_counts[region]
                    or any(int(self.region[j.user]) != region for j in tasks)):
                raise ValueError('Invalid actual region/class group')
            cfg = self.cfg
            beta = self.eps0(n)/(2*cfg['region_radius_m']) if self.geoi else 0.
            overhead = max(cfg['dt_s'], math.ceil(cfg['privacy_processing_s']/cfg['dt_s']-1e-9)*cfg['dt_s']) if self.geoi else 0.
            self.max_class_group = max(self.max_class_group, n)
            self.release_serial += 1
            if not hasattr(self, '_slot_admissions'):
                self._slot_admissions = []
            for j in tasks:
                if j.id not in getattr(self, '_staged_at_slot_start', set()):
                    self._slot_admissions.append((j.region, j.profile, j.user))
                truth = self.xy[j.user]
                # Only the reporting mechanism and its physical costs differ.
                if self.geoi:
                    rng = np.random.default_rng(np.random.SeedSequence([self.seed, 3, j.id]))
                    radius = rng.gamma(2., 1/beta)
                    angle = rng.uniform(0., 2*np.pi)
                    j.report = truth + radius*np.array([np.cos(angle), np.sin(angle)])
                    j.ql = float(np.linalg.norm(j.report-truth))
                    self.energy(j, cfg['report_energy_j'], 'report')
                    self.noise_publications[j.user] += 1
                else:
                    j.report, j.ql = truth.copy(), 0.
                j.epsilon, j.batch_n = beta, n
                j.status, j.release = 'report', self.time
                j.report_ready = self.time + overhead
                self.sum_class_group_wait += j.active_wait
                self.max_observed_wait = max(self.max_observed_wait, j.active_wait)
                if j.active_wait > cfg['wait_max_s']+1e-9 or cfg['weight_quality'] != 0:
                    raise AssertionError('Waiting or common reward contract changed')
                if self.measured(j):
                    self.measure_published += 1
                    self.sum_ql += j.ql
                    self.sum_ql_theory += self.q_estimate(n)
                    self.sum_k += n
    return SameGreedy


def simulate(job):
    mode, n, trace, signature = job
    started = time.monotonic()
    sys.path[:0] = [str(ROOT/'.deps'), str(SOURCE), str(SOURCE/'model')]
    from pressure9.environment import GeoIEnvironment
    from pressure9.protocol import configuration
    from experiments.lr_screen.runtime import np
    from environment import GreedyPolicy

    cfg, _ = configuration('Greedy', 87021, n)
    env = controller_environment(GeoIEnvironment, np)(
        cfg, geoi=mode == MODES[1], seed=trace, phase_s=0., family='Greedy',
        measure_start=5., measure_end=305., collect=False)
    assert env.u == n
    stop = round(305./cfg['dt_s'])
    env.arrival_stop_slot = stop
    policy = GreedyPolicy(env.cfg)
    print('START', mode, n, trace, flush=True)
    for slot in range(stop+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
        env.step(policy, arrivals=slot < stop)
        if env.slot >= stop and not env.tasks:
            break
    else:
        raise RuntimeError('Natural drainage incomplete')
    row = dict(env.metrics(), method=mode, users=n, trace_seed=trace,
               full_input_hash=env.full_input_hash, phase_s=0., warmup_s=5., measure_s=300.,
               all_tasks_resolved=True, elapsed_wall_seconds=time.monotonic()-started)
    check(row, signature)
    return row


def check(row, signature):
    if (row['full_input_hash'], int(row['arrivals'])) != signature:
        raise AssertionError('Exogenous input differs from the original plot')
    n = int(row['arrivals'])
    assert n > 0 and row['completed']+row['failed'] == n
    assert row['number_queued_pending'] == row['class_staging_left'] == 0
    assert row['phase_s'] == 0 and row['warmup_s'] == 5 and row['measure_s'] == 300
    assert row['class_gate_wait_actions'] == 0
    assert 0 <= row['qos_rate'] <= row['qos95_rate'] <= 1
    energy = sum(row[f'{k}_energy_j_per_arrival'] for k in ('transmit','compute','report'))
    assert math.isclose(row['energy_j_per_arrival'], energy, rel_tol=1e-8)
    assert math.isclose(row['qos_rate'], row['failed']/n+row['late_completed_rate'], abs_tol=1e-12)
    if row['method'] == MODES[0]:
        assert row['report_energy_j_per_arrival'] == 0
    else:
        assert row['report_energy_j_per_arrival'] > 0


def reference_signatures():
    sig = {}
    for row in read_csv(REFERENCE/'scene_metrics_all_saved.csv'):
        n, t = int(row['users']), int(row['trace_seed'])
        if n not in POPULATIONS or t not in TRACES:
            continue
        assert float(row['phase_s']) == 0 and float(row['warmup_s']) == 5 and float(row['measure_s']) == 300
        value = row['full_input_hash'], int(row['arrivals'])
        assert sig.setdefault((n,t),value) == value
    assert set(sig) == {(n,t) for n in POPULATIONS for t in TRACES}
    return sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, choices=(1,2), default=2)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sig = reference_signatures()
    jobs = [(m,n,t,sig[n,t]) for n in POPULATIONS for m in MODES for t in TRACES]
    manifest = dict(source_sha256=source_hashes(), populations=list(POPULATIONS),
        traces=list(TRACES), methods=list(MODES), expected_scenes=len(jobs),
        warmup_s=5, measure_s=300, identical_exogenous_inputs_required=True,
        source_plot=str(REFERENCE/'scalability_common.png'), training=False,
        note='Paired descriptive re-evaluation on previously seen inputs. Original QoS in overlay.')
    if args.resume:
        assert json.loads((OUT/'manifest.json').read_text()) == manifest, 'Frozen inputs changed'
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        write_json(OUT/'manifest.json', manifest)
    rows = {}
    missing = []
    for job in jobs:
        m,n,t,s = job
        path = OUT/f'{m}_{n}_{t}.json'
        if path.exists():
            row = json.loads(path.read_text())
            assert (row['method'],row['users'],row['trace_seed']) == (m,n,t)
            check(row,s); rows[m,n,t] = row
        else:
            missing.append(job)
    print('Scenes saved:', len(rows), '/', len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(simulate, job): job for job in missing}
        try:
            for future in as_completed(pending):
                m,n,t,_ = pending[future]
                row = future.result()
                write_json(OUT/f'{m}_{n}_{t}.json', row)
                rows[m,n,t] = row
                print('SAVED', len(rows), '/', len(jobs), m,n,t,round(row['elapsed_wall_seconds'],1), flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    assert len(rows) == len(jobs) and source_hashes() == manifest['source_sha256']
    with (OUT/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(next(iter(rows.values()))))
        w.writeheader()
        w.writerows(rows[m,n,t] for m,n,t,_ in jobs)
    write_json(OUT/'complete.json', dict(completed=True, scenes=len(rows),
        original_input_fingerprint_groups=len(sig), populations=list(POPULATIONS),
        training=False, deadlines_changed=False))


if __name__ == '__main__':
    main()

"""Streaming task cohort accounting; no raw task export."""

from collections import defaultdict
from math import isclose


ENERGY = ('transmit', 'compute', 'report')


def with_task_telemetry(base):
    class Measured(base):
        def __init__(self, *args, **kwargs):
            self._task_energy = defaultdict(lambda: [0, 0., 0., 0.])
            super().__init__(*args, **kwargs)

        def end(self, task, failed=False):
            if self.measured(task):
                stage = task.status
                if stage in ('upload', 'propagation'):
                    stage += '_bs' if task.node < self.b else '_sat'
                elif stage in ('compute', 'return'):
                    stage += ('_ue', '_bs', '_sat')[self.node_kind(task.node)]
                if failed:
                    expired = self.time >= task.arrival + self.cfg['lifetime_s'] - 1e-9
                    outcome = 'expired' if expired else 'rejected'
                else:
                    late = self.time-task.arrival > task.deadline + 1e-9
                    outcome = 'late' if late else 'on_time'
                group = self._task_energy[task.profile, outcome, stage]
                group[0] += 1
                for i, name in enumerate(ENERGY, 1):
                    group[i] += getattr(task, name+'_energy')
            return super().end(task, failed=failed)

        def accounting(self, m):
            rows = [dict(profile=profile, outcome=outcome, stage=stage, tasks=group[0],
                         **{k+'_j': v for k, v in zip(ENERGY, group[1:])})
                    for (profile, outcome, stage), group in sorted(self._task_energy.items())]
            arrivals = m['arrivals']
            assert sum(r['tasks'] for r in rows) == arrivals
            assert sum(r['tasks'] for r in rows if r['outcome'] in ('on_time', 'late')) == m['completed']
            assert sum(r['tasks'] for r in rows if r['outcome'] in ('late', 'expired', 'rejected')) == round(m['qos_rate']*arrivals)
            for k in ENERGY:
                assert isclose(sum(r[k+'_j'] for r in rows) / arrivals,
                               m[k+'_energy_j_per_arrival'], rel_tol=1e-8, abs_tol=1e-12)
            return rows

    return Measured

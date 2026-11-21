"""Record completed-task tail latency without exporting individual tasks."""

from collections import Counter
from math import ceil, isfinite
import os

from deadline_rescue import with_deadline_rescue
import lowload_worker as worker
from paper_load_protocol import BRANCHES, stage


def with_latency_bins(base):
    class Hist(base):
        def __init__(self, *args, **kwargs):
            self.completed_latency_bins = Counter()
            super().__init__(*args, **kwargs)

        def end(self, task, failed=False):
            if not failed and self.measured(task):
                ms = 1000 * (self.time - task.arrival)
                assert isfinite(ms) and 0 <= ms <= 1000 * (self.cfg['lifetime_s'] + self.cfg['dt_s']) + 1e-8
                self.completed_latency_bins[ceil(ms)] += 1
            return super().end(task, failed=failed)

        def metrics(self):
            m = super().metrics()
            assert sum(self.completed_latency_bins.values()) == m['completed']
            return dict(m, completed_latency_ms_bins=list(sorted(self.completed_latency_bins.items())))
    return Hist


if __name__ == '__main__':
    [name] = [k for k, branch in BRANCHES.items() if os.environ['GITHUB_REF'] == 'refs/heads/' + branch]
    worker.spec = stage(name)
    telemetry = worker.with_deadline_sweep
    worker.with_deadline_sweep = lambda env: with_latency_bins(with_deadline_rescue(telemetry(env)))
    raise SystemExit(0 if worker.main() else 2)

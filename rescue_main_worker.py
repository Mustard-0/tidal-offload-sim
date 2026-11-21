"""Run a fixed workload without training or exporting raw tasks."""

import os

from deadline_rescue import with_deadline_rescue
import lowload_worker as worker
from rescue_main_protocol import BRANCHES, stage


if __name__=='__main__':
    [name]=[k for k,b in BRANCHES.items() if os.environ['GITHUB_REF']=='refs/heads/'+b]
    worker.spec=stage(name)
    telemetry=worker.with_deadline_sweep
    worker.with_deadline_sweep=lambda env:with_deadline_rescue(telemetry(env))
    raise SystemExit(0 if worker.main() else 2)

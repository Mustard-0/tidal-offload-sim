"""Evaluate an opt-in dispatcher without rewriting any frozen method or simulator."""

import os

import lowload_worker as worker
from dispatch_trial_protocol import BRANCHES, stage
from physical_dispatch import with_dispatch


def main():
    [name]=[k for k,b in BRANCHES.items() if os.environ['GITHUB_REF']=='refs/heads/'+b]
    worker.spec=stage(name)
    telemetry=worker.with_deadline_sweep
    worker.with_deadline_sweep=lambda env:with_dispatch(telemetry(env))
    return worker.main()


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

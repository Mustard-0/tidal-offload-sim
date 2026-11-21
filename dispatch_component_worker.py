"""Revert exactly one of the previous trial's two interventions."""

import os

import lowload_worker as worker
from dispatch_component_protocol import BRANCHES, stage
from physical_dispatch import with_dispatch


def with_component(base,part):
    combined=with_dispatch(base)
    if part=='route':
        class RouteOnly(combined):
            enqueue_compute=base.enqueue_compute
        return RouteOnly
    if part=='queue':
        class QueueOnly(combined):
            route_features=base.route_features
            step=base.step
        return QueueOnly
    raise ValueError('Unknown dispatcher component')


def main():
    [(part,span)]=[k for k,b in BRANCHES.items() if os.environ['GITHUB_REF']=='refs/heads/'+b]
    worker.spec=stage(part,span)
    telemetry=worker.with_deadline_sweep
    worker.with_deadline_sweep=lambda env:with_component(telemetry(env),part)
    return worker.main()


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

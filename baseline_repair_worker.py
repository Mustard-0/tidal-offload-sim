"""Run the frozen workload with repaired observation and measurement wiring."""

import argparse

import seven_scale_worker as old
import baseline_repair_protocol as spec
from baseline_queue_stats import with_queue_sampling
from task_aware_td3 import TaskAwareTD3


_controller = old.controller
_env_types = old.frozen.evaluation_environment_types
_initialize = old.initialize_eval


def repaired_controller(method):
    return TaskAwareTD3() if method == "MARATD3-Adapted" else _controller(method)


def repaired_env_types(**kwargs):
    plain, *others = _env_types(**kwargs)
    return with_queue_sampling(plain), *others


def initialize_eval(*args):
    configure()
    _initialize(*args)


def configure():
    old.spec = spec
    old.controller = repaired_controller
    old.frozen.evaluation_environment_types = repaired_env_types
    old.initialize_eval = initialize_eval


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cell", type=int, choices=range(len(spec.CELLS)))
    a = p.parse_args()
    configure()
    old.setup()
    raise SystemExit(0 if old.run(old.frozen.transport.Transport(), a.cell) else 2)

"""Reuse the verified timing, policies and per-population receipt path."""

import os

import lowload_worker as worker
from population_grid_protocol import BRANCHES, stage


if __name__=='__main__':
    [name]=[k for k,b in BRANCHES.items() if os.environ['GITHUB_REF']=='refs/heads/'+b]
    worker.spec=stage(name)
    raise SystemExit(0 if worker.main() else 2)

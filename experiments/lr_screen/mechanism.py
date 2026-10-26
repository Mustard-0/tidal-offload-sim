"""Privacy-only adapter of supplied transfer.py; never imports its greedy policy.

The task-keyed sampler is identical to report_for(..., 'Shrink05'). Public
membership, actual distinct users and the accountant are checked separately.
"""
import numpy as np
from .vendor.location_methods import bounded_exponential, contract
from .vendor.accountant import calibrate, delta_bound

ARCHIVE_SHA256 = "c563deecf810a5e65218cf480976582112d5a95452282bb3b5c6fbaac0f4d2ec"


def report_for(truth, center, radius, eps0, seed, task_id, shrink=.5):
    truth, center = np.asarray(truth, float), np.asarray(center, float)
    if (truth.shape != (2,) or center.shape != (2,) or
            not np.isfinite(truth).all() or not np.isfinite(center).all() or
            not np.isfinite(radius) or radius <= 0 or not 0 < eps0 <= 8 or
            not 0 <= shrink <= 1 or np.linalg.norm(truth-center) > radius+1e-7):
        raise ValueError("Invalid fixed public disk, privacy budget or shrink")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3, task_id]))
    raw = bounded_exponential(rng, truth[None], center, radius, eps0, 'tight', 0.)[0]
    return raw, contract(raw, center, shrink)


def actual_batch(tasks, cfg):
    if not tasks or len({(j.region, j.profile) for j in tasks}) != 1:
        raise ValueError("Expected one nonempty public region/class batch")
    if len({j.user for j in tasks}) != len(tasks) or len({j.id for j in tasks}) != len(tasks):
        raise ValueError("Batch users and task IDs must be distinct")
    n = len(tasks)
    from population import region_counts, user_region
    size = region_counts(cfg)[tasks[0].region]
    if n > size or any(j.region != user_region(cfg,j.user) for j in tasks):
        raise ValueError("User does not belong to this fixed public region")
    eps0 = calibrate(n, cfg['epsilon_req'], cfg['delta_req'],
                     cfg['accountant_grid'], cfg['accountant_max_eps0'])
    bound = delta_bound(eps0, n, cfg['epsilon_req'])
    if bound > cfg['delta_req']+1e-12:
        raise ValueError("Actual-batch accountant failed")
    return n, eps0, bound

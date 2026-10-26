"""Generic finite-n variation-ratio bound (Wang et al., PVLDB 2024).

p=q=exp(eps0), beta=tanh(eps0/2), alpha=r=1/(p+1).
The no-clone term is zero for these generic parameters. Condition on C,
then sum the exact binomial tail of P-exp(eps_req)*Q. No asymptotic n bound.
The result is conditional on public region/class/membership, not a
user-trajectory guarantee for all adaptive releases and side channels.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.stats import binom


def delta_bound(eps0, n, eps_req):
    if int(n) != n or n < 1 or not 0 < eps_req <= 8 or not 0 < eps0 <= 8:
        raise ValueError("Expected n>=1 and 0<epsilon<=8")
    if eps0 <= eps_req:
        return 0.0
    p, e = math.exp(eps0), math.exp(eps_req)
    alpha = 1.0 / (p + 1.0)
    c = np.arange(n)
    threshold = (e * p - 1) * (c + 1) / ((p - 1) * (1 + e))
    first = np.floor(threshold).astype(int) + 1
    tail = alpha * ((p - e) * binom.sf(first - 2, c, 0.5)
                    + (1 - e * p) * binom.sf(first - 1, c, 0.5))
    return float(np.maximum(tail, 0) @ binom.pmf(c, n - 1, 2 * alpha))


@lru_cache(maxsize=8192)
def calibrate(n, eps_req=1.0, delta_req=1e-6, step=0.01, maximum=8.0):
    if n < 1 or int(n) != n or not 0 < delta_req < 1 or not 0 < step <= maximum or not 0 < eps_req <= maximum <= 8:
        raise ValueError("Invalid privacy configuration")
    # The pure local value is always feasible, including off-grid eps_req.
    lo, hi = math.floor(eps_req / step + 1e-9), math.floor(maximum / step + 1e-9)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = round(mid * step, 12)
        if delta_bound(candidate, n, eps_req) + 1e-12 <= delta_req:
            lo = mid
        else:
            hi = mid - 1
    return max(eps_req, round(lo * step, 12))


def planar_noise(rng, epsilon_per_m, size=None):
    if epsilon_per_m <= 0:
        raise ValueError("epsilon_per_m must be positive")
    radius = rng.gamma(2.0, 1.0 / epsilon_per_m, size=size)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=size)
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def local_epsilon(cfg, n, no_amp=False):
    if n<1 or int(n)!=n:
        raise ValueError("Batch size must be an actual positive distinct-user count")
    eps0 = cfg["epsilon_req"] if no_amp else calibrate(
        int(n), cfg["epsilon_req"], cfg["delta_req"], cfg["accountant_grid"], cfg["accountant_max_eps0"])
    return eps0 / (2.0 * cfg["region_radius_m"])


@lru_cache(maxsize=128)
def _first_amplifying_batch(max_distinct, eps_req, delta_req, step, maximum):
    for n in range(2, max_distinct + 1):
        if calibrate(n, eps_req, delta_req, step, maximum) > eps_req + 1e-12:
            return n
    return None


def amplification_threshold(cfg):
    """Smallest attainable distinct-user batch with a nonzero accountant gain."""
    return _first_amplifying_batch(
        int(cfg["users_per_region"]), float(cfg["epsilon_req"]),
        float(cfg["delta_req"]), float(cfg["accountant_grid"]),
        float(cfg["accountant_max_eps0"]),
    )

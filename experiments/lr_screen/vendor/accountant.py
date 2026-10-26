"""Frozen finite-n conditional shuffle accountant copied from model/privacy.py.

IMPORTANT: applies to a single genuinely unlinkable fixed-public-membership batch,
not to a user's repeated releases or full communication transcript.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.stats import binom


def delta_bound(eps0, n, eps_req):
    if int(n) != n or n < 1 or not 0 < eps_req <= 8 or not 0 < eps0 <= 8:
        raise ValueError('Expected n>=1 and 0<epsilon<=8')
    if eps0 <= eps_req:
        return 0.0
    p,e=math.exp(eps0),math.exp(eps_req)
    alpha=1.0/(p+1.0)
    c=np.arange(n)
    threshold=(e*p-1)*(c+1)/((p-1)*(1+e))
    first=np.floor(threshold).astype(int)+1
    tail=alpha*((p-e)*binom.sf(first-2,c,0.5)+(1-e*p)*binom.sf(first-1,c,0.5))
    return float(np.maximum(tail,0)@binom.pmf(c,n-1,2*alpha))

@lru_cache(maxsize=8192)
def calibrate(n,eps_req=1.0,delta_req=1e-6,step=.01,maximum=8.):
    if n<1 or int(n)!=n or not 0<delta_req<1 or not 0<step<=maximum or not 0<eps_req<=maximum<=8:
        raise ValueError('Invalid privacy configuration')
    lo,hi=math.floor(eps_req/step+1e-9),math.floor(maximum/step+1e-9)
    while lo<hi:
        mid=(lo+hi+1)//2
        candidate=round(mid*step,12)
        if delta_bound(candidate,n,eps_req)+1e-12<=delta_req:
            lo=mid
        else:
            hi=mid-1
    return max(eps_req,round(lo*step,12))

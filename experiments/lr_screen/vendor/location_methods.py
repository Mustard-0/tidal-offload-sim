"""Four location mechanisms. All public disk operations use PUBLIC center/radius.

Fixed public S=B(c,R), true x in S, report y in S:
  p_beta,eta(y|x) = 1_S(y) exp(-beta |x-y| - eta |y-c|/R)/Z_beta,eta(x)
  original: beta=eps0/(4R), eta=0;
  tight:    beta=eps0/(2R), eta=0;
  prior:    beta=eps0/(2R), eta>=0 (PUBLIC radial output bias).
For any eta>=0, Z(x) only depends on |x-c|, so
 |log Z(x)-log Z(x')| <= beta*||x-c|-|x'-c||.
The full log likelihood ratio is bounded by
 beta*(|x-x'|+||x-c|-|x'-c||) <= 2 beta R = eps0.
Thus 'tight' and 'radial prior' satisfy eps0-LDP for fixed public disk.
This is full-disk local DP; do NOT conflate with an eps0/(2R)-slope metric Geo-I.
"""
import numpy as np

def disk_points(rng,center,R,n):
    u=rng.random((n,2))
    rr=R*np.sqrt(u[:,0]); aa=2*np.pi*u[:,1]
    return np.asarray(center)+np.column_stack((rr*np.cos(aa),rr*np.sin(aa)))

def bounded_exponential(rng,truth,center,R,eps0,variant='old',eta=0.0):
    if variant not in ('old','tight') or eta<0:raise ValueError('bad variant or eta')
    beta=eps0/(4*R) if variant=='old' else eps0/(2*R)
    truth=np.asarray(truth,float); result=np.empty_like(truth)
    pending=np.arange(len(truth)); loops=0
    while pending.size:
        loops+=1
        if loops>1000:raise RuntimeError('Rejection sampler stalled')
        proposed=disk_points(rng,center,R,len(pending))
        dist=np.linalg.norm(proposed-truth[pending],axis=1)
        prior=np.linalg.norm(proposed-np.asarray(center),axis=1)/R
        accept=rng.random(len(pending)) < np.exp(-beta*dist-eta*prior)
        result[pending[accept]]=proposed[accept]
        pending=pending[~accept]
    return result

def contract(report,center,gain):
    if not 0<=gain<=1:raise ValueError('gain must lie in [0,1]')
    return np.asarray(center)+gain*(np.asarray(report)-np.asarray(center))

def planar_laplace(rng,truth,R,epsilon):
    rad=rng.gamma(2.0,2*R/epsilon,size=len(truth))
    ang=rng.uniform(0,2*np.pi,size=len(truth))
    return np.asarray(truth)+np.column_stack((rad*np.cos(ang),rad*np.sin(ang)))

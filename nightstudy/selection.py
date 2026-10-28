"""Pure validation-only selector, independently testable without training."""
import math
from .protocol import VARIANTS,SEEDS


def choose(records,complete):
    if not complete:return [],None
    assert {(r['variant'],r['seed']) for r in records}=={(v,s) for v in VARIANTS for s in SEEDS}
    assert len(records)==8
    scores=[]
    for variant in VARIANTS:
        rows=[r for r in records if r['variant']==variant]
        for r in rows:
            assert all(math.isfinite(r[k]) and r[k]>=0 for k in ('initial_cost','middle_cost','cost','qos'))
            assert 0<=r['qos']<=1
        scores.append(dict(variant=variant,cost=sum(r['cost'] for r in rows)/2,
            qos=sum(r['qos'] for r in rows)/2,
            per_seed_learning=all(r['cost']<=.9*r['initial_cost'] for r in rows),
            late_stable=all(r['cost']<=1.05*r['middle_cost'] for r in rows)))
    control=next(r for r in scores if r['variant']=='control')
    eligible=[r for r in scores if r['per_seed_learning'] and r['late_stable'] and
        r['cost']<=.98*control['cost'] and r['qos']<=control['qos']+.005]
    return scores,min(eligible,key=lambda r:(r['cost'],r['variant'])) if eligible else None

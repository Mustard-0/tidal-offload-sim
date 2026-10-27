"""Mean-conserving physical-cost attribution, including unequal region sizes."""
import math


def mean_preserving_credit(costs,counts,slot_scale,alpha):
    if (not 0<=alpha<=1 or not math.isfinite(slot_scale) or slot_scale<=0
        or not counts or any(int(n)!=n or n<=0 for n in counts)
        or sum(counts)!=len(costs)
        or any(not math.isfinite(c) or c<0 for c in costs)):
        raise ValueError('Invalid actual physical credit inputs')
    n=len(costs);g=len(counts)
    team=-math.fsum(costs)/(n*slot_scale)
    user=[-float(c)/slot_scale for c in costs]
    regional=[];start=0
    for count in counts:
        end=start+int(count)
        regional.append(-g*math.fsum(costs[start:end])/(n*slot_scale));start=end
    rewards=[(1-alpha)*team+alpha*r for r in user+regional]
    return rewards,team

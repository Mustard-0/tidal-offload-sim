"""Reuse reviewed update engines; never insert analytic hints or TM modules."""
import copy
from .recent import MaskedDiscreteSAC,RecurrentAttentionTD3,defaults
from experiments.lr_screen.algorithms import learner_for


def learner(method,t,cfg):
    if method in ('SD-SAC-Adapted','MARATD3-Adapted'):
        settings=defaults()
        settings.update(t)
        return (MaskedDiscreteSAC if method=='SD-SAC-Adapted' else RecurrentAttentionTD3)(settings,cfg,'cpu')
    return learner_for(method,t,cfg,'cpu')


def weights_only(p):
    if hasattr(p,'actors'):
        return copy.deepcopy(p.state(for_evaluation=True))
    return copy.deepcopy(dict(actor=p.actor.state_dict(),q=p.q.state_dict()))


def evaluation_policy(method,t,cfg,state):
    p=learner(method,t,cfg)
    if hasattr(p,'actors'):
        p.load(state)
    else:
        p.actor.load_state_dict(state['actor'])
        p.q.load_state_dict(state['q'])
    p.deterministic=method=='MARATD3-Adapted'
    if hasattr(p,'collect_transitions'):
        p.collect_transitions=False
    return p


def set_progress(p,step,total):
    if hasattr(p,'set_progress'):
        p.set_progress(step,total)
    else:
        scale=1-.9*(step-1)/max(total-1,1)
        for opt,key in ((p.aopt,'actor_lr'),(p.copt,'critic_lr')):
            for group in opt.param_groups:
                group['lr']=p.cfg[key]*scale*group.get('ratio',1.)


def update(p,context,glob):
    if hasattr(p,'actors'):
        return p.update(context,glob)
    stats=[p.update() for _ in range(p.cfg['offpolicy_updates_per_block'])]
    keys=set().union(*(r.keys() for r in stats))
    return {k:sum(float(r[k]) for r in stats if k in r)/sum(k in r for r in stats) for k in keys}

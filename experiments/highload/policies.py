import copy
from .protocol import configuration, ABLATIONS
from .baselines import MaskedDiscreteSAC, RecurrentAttentionTD3, defaults
from experiments.lr_screen.algorithms import learner_for


def learner(method,training,cfg,device='cpu'):
    if method in ('SD-SAC-Adapted','MARATD3-Adapted'):
        c=defaults()
        c.update(training)
        c.update(batch=64 if method=='SD-SAC-Adapted' else 8,
                 replay_capacity=20000 if method=='SD-SAC-Adapted' else 1024,
                 history=8)
        return (MaskedDiscreteSAC if method=='SD-SAC-Adapted' else RecurrentAttentionTD3)(c,cfg,device)
    underlying='IPPO' if method=='TMAPO-LocalCritic' else 'TMAPO' if method in ABLATIONS else method
    return learner_for(underlying,training,cfg,device)


def weights_only(p):
    if hasattr(p,'actors'):
        return copy.deepcopy(p.state(for_evaluation=True))
    return copy.deepcopy(dict(actor=p.actor.state_dict(),q=p.q.state_dict()))


def load_weights(p,state):
    if hasattr(p,'actors'):p.load(state)
    else:
        p.actor.load_state_dict(state['actor'])
        p.q.load_state_dict(state['q'])


def evaluation_policy(method,t,cfg,state):
    p=learner(method,t,cfg)
    load_weights(p,state)
    p.deterministic=method=='MARATD3-Adapted'
    if hasattr(p,'collect_transitions'):p.collect_transitions=False
    return p


def set_progress(p,block,total):
    if hasattr(p,'set_progress'):
        p.set_progress(block,total)
    else:
        scale=1-.9*(block-1)/max(1,total-1)
        for opt,key in ((p.aopt,'actor_lr'),(p.copt,'critic_lr')):
            for group in opt.param_groups:
                group['lr']=p.cfg[key]*scale*group.get('ratio',1.)


def update(p,context,glob):
    if hasattr(p,'actors'):
        return p.update(context,glob)
    rows=[p.update() for _ in range(6)]
    keys=set().union(*(r.keys() for r in rows))
    return {key:sum(float(r[key]) for r in rows if key in r)/sum(key in r for r in rows)
            for key in keys}

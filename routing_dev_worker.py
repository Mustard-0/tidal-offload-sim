"""Run the frozen six-cell development screen on cloud CPU."""
import argparse
import time

import optimization2_worker as inherited
import routing_dev_protocol as spec
from review_checks import bootstrap, check_metrics


def transport():
    inherited.BRANCH, inherited.CAMPAIGN, inherited.plan = spec.BRANCH, spec.CAMPAIGN, spec.plan
    return inherited.Transport()


def attach(policy, scale, public_prior):
    if not scale:
        return
    original = policy.policy_logits

    def logits(stage, obs):
        result = original(stage,obs)
        return result + public_prior('route',obs,policy.env_cfg,scale) if stage=='route' else result

    policy.policy_logits = logits


def execute(t, cell, result):
    bootstrap()
    from experiments.lr_screen.runtime import torch, seed_all
    from learners import cost_prior
    from pilot_worker import module_hash
    from pressure9.policies import evaluation_policy
    from review_environment import environment_types
    from scalability_protocol import settings
    from scalability_worker import prior_policy

    torch.set_num_threads(1)
    _, env_type, _, _ = environment_types()
    variant, seed = cell.rsplit('_seed',1)
    seed=int(seed)
    weights,_ = prior_policy(t,seed)
    deadline=time.monotonic()+spec.LIMIT
    for n in spec.POPULATIONS:
        cfg, train = settings('TMAPO',seed,n)
        for trace in spec.TRACES:
            for phase in spec.PHASES:
                if time.monotonic()>=deadline:
                    raise TimeoutError('Development wall budget before next scene')
                seed_all(trace)
                policy = evaluation_policy('TMAPO',train,cfg,weights)
                before=module_hash(policy)
                if result['model_hash'] is None:
                    result['model_hash']=before
                assert result['model_hash']==before
                attach(policy,spec.VARIANTS[variant],cost_prior)
                env=env_type(cfg,seed=trace,phase_s=phase,measure_start=spec.WARMUP,
                    measure_end=spec.WARMUP+spec.MEASURE)
                admission=round((spec.WARMUP+spec.MEASURE)/cfg['dt_s'])
                with torch.no_grad():
                    for slot in range(admission+203):
                        env.step(policy,arrivals=slot<admission)
                        if env.slot>=admission and not env.tasks:
                            break
                    else:
                        raise AssertionError('Development arrivals did not settle')
                assert module_hash(policy)==before
                result['rows'].append(dict(users=n,trace_seed=trace,phase_s=phase,full_input_hash=env.full_input_hash,
                    **check_metrics(env)))
    assert len(result['rows'])==len(spec.EXPECTED)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('evaluate','select'))
    parser.add_argument('--cell',choices=spec.CELLS)
    args=parser.parse_args()
    t=transport()
    if args.action=='select':
        cells={c:t.read_small(c+'.enc',c) for c in spec.CELLS if c+'.enc' in t.assets}
        result=spec.select(cells)
        t.small('selection.enc',result,'selection')
        return result['complete']
    assert args.cell in spec.CELLS
    result=dict(completed=False,cell=args.cell,rows=[],model_hash=None)
    try:
        assert inherited.within_dispatch_window(t)
        execute(t,args.cell,result)
        result['completed']=True
    except Exception as error:
        result['error']=dict(type=type(error).__name__,message=str(error))
    t.small(args.cell+'.enc',result,args.cell)
    return result['completed']


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

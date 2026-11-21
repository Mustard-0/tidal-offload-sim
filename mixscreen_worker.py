"""Evaluate fixed policies with paired data; store only encrypted compact outcomes."""

import argparse
import math
import time

import baseline_individual_worker as previous
from baseline_individual_rules import with_adaptive_local
from energy_qos_telemetry import with_task_telemetry
from energy_qos_worker import parent_store, weights_for
from evaluation_adapters import evaluation_environment_types
from focus_fallback import heavy_fallback
from mixscreen_admission import attach_admission, with_admission_quote
import mixscreen_protocol as spec
import optimization2_worker as store
from review_checks import bootstrap, check_metrics
from scalability_protocol import settings


def evaluate(t, method, arm, result):
    bootstrap()
    from experiments.lr_screen.runtime import seed_all, torch
    from learners import cost_prior
    from pilot_worker import module_hash
    from pressure9.policies import evaluation_policy

    torch.set_num_threads(1)
    plain, tt, greedy, local = evaluation_environment_types()
    kind = with_adaptive_local(plain) if method == 'Local-Only' else heavy_fallback(
        tt if method == 'TMAPO' else plain)
    if arm == 'admission':
        kind = with_admission_quote(kind)
    kind = with_task_telemetry(kind)
    seed = None if method in spec.RULES else spec.SEED
    key = method if seed is None else f'{method}_seed{seed}'
    parent = parent_store(t) if seed is not None and method != 'TMAPO' else None
    weights, expected = weights_for(t,parent,key,method,seed)
    deadline = time.monotonic() + spec.WALL
    for mix,n,trace,phase in spec.CASES:
        if time.monotonic() >= deadline:
            raise TimeoutError('Wall budget before next scene')
        seed_all(trace)
        cfg,train = settings(method,seed,n)
        cfg = spec.settings_for_mix(cfg,mix)
        options = dict(seed=trace,phase_s=phase,measure_start=spec.WARM,measure_end=spec.WARM+spec.MEASURE)
        env = kind(cfg,**options) if method=='TMAPO' else kind(cfg,family=method,**options)
        model = None
        if seed is None:
            policy = local() if method=='Local-Only' else greedy(cfg)
        else:
            policy = evaluation_policy(method,train,cfg,weights)
            model = module_hash(policy)
            expected = model if expected is None else expected
            assert model==expected
            if method=='TMAPO':
                original=policy.policy_logits
                def logits(stage,obs):
                    value=original(stage,obs)
                    return value+cost_prior('route',obs,cfg,1.) if stage=='route' else value
                policy.policy_logits=logits
                if arm=='admission':
                    attach_admission(policy,env)
        stop = round((spec.WARM+spec.MEASURE)/cfg['dt_s'])
        with torch.no_grad():
            for i in range(stop+math.ceil(cfg['lifetime_s']/cfg['dt_s'])+3):
                env.step(policy,arrivals=i<stop)
                if env.slot>=stop and not env.tasks:
                    break
                if i%100==99 and time.monotonic()>=deadline:
                    raise TimeoutError(f'Wall budget in {mix} N={n}')
            else:
                raise AssertionError('Natural drainage did not finish')
        assert seed is None or module_hash(policy)==model
        m=check_metrics(env)
        result['rows'].append(dict(mix=mix,method=method,arm=arm,training_seed=seed,users=n,
            trace_seed=trace,phase_s=phase,measure_s=spec.MEASURE,model_hash=model,
            full_input_hash=env.full_input_hash,arrival_hash=env.arrival_hash,
            admission_edits=getattr(env,'measured_admission_edits',0),
            energy_by_task_group=env.accounting(m),**m))
        print(f'Completed {method} {arm} {mix} N={n}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('method')
    p.add_argument('arm')
    a=p.parse_args()
    assert (a.method,a.arm) in spec.ARMS
    previous.spec=spec
    store.CAMPAIGN=spec.CAMPAIGN
    t=previous.Transport()
    result=dict(method=a.method,arm=a.arm,completed=False,rows=[])
    try:
        assert store.within_dispatch_window(t)
        evaluate(t,a.method,a.arm,result)
        assert len(result['rows'])==len(spec.CASES)
        result['completed']=True
    except Exception as err:
        result['error']=dict(type=type(err).__name__,message=str(err))
    name=a.method+'.'+a.arm
    t.small(name+'.enc',result,name)
    return result['completed']


if __name__=='__main__':
    raise SystemExit(0 if main() else 2)

"""Finite real-N300 path and speed preflight; no final-test or LR selection."""
import hashlib
import json
from pathlib import Path
import time
import unittest
from .protocol import ROOT,configuration,population,ABLATIONS,source_hashes,digest,plan
from .policies import learner,update
from .environment import Environment
from experiments.lr_screen.runtime import torch,np,seed_all,available_ram_gib,runtime_info,rng_state,checkpoint_write


def run():
    out=ROOT/'evidence/preflight'/str(time.time_ns())
    out.mkdir(parents=True,exist_ok=True)
    import io
    from .test_protocol import Contract
    stream=io.StringIO()
    test_result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    (out/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    if not test_result.wasSuccessful():raise RuntimeError('Preflight contract tests failed')
    rows=[]
    for method in ('TMAPO','IPPO','PPO','SD-SAC-Adapted','MARATD3-Adapted',*ABLATIONS):
        free=available_ram_gib()
        if free<1.25:raise MemoryError('Need1.25GiB for real-population preflight')
        seed_all(87021)
        c,t=configuration(method,3 if method.startswith('TMAPO') else 1)
        c=population(c,300)
        p=learner(method,t,c)
        module=p.actors if hasattr(p,'actors') else p.actor
        before=[v.detach().clone() for v in module.parameters()]
        e=Environment(c,seed=814001,phase_s=30.,ablation=method)
        e.arrival_stop_slot=29800
        started=time.monotonic();rewards=0.
        for slot in range(220):
            reward,records,_,_=e.step(p,True)
            p.remember(e.agent_rewards(0.),records,False)
            rewards+=float(reward)
            if (slot+1)%100==0:
                context,glob=e.prepare(True);stats=update(p,context,glob)
                if not all(np.isfinite(float(v)) for v in stats.values()):raise FloatingPointError(method)
        if not any(not torch.equal(a,b) for a,b in zip(before,module.parameters())):
            raise AssertionError('No actor update: '+method)
        if e.max_observed_wait>1e-9 or e.windows_seen!=2 or len(e.pred_error)!=1:
            raise AssertionError('WAIT or causal-history contract')
        row=dict(method=method,actual_users=e.u,steps=220,forecast_windows=e.windows_seen,
            error_windows=len(e.pred_error),active_wait_max=e.max_observed_wait,
            arrivals=e.arrivals,trace_hash=e.arrival_hash,actor_updated=True,
            wall_seconds=time.monotonic()-started,available_before_gib=free,
            available_after_gib=available_ram_gib(),stats=stats)
        rows.append(row)
        (out/'partial.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
        del p,e,before
    if len({r['trace_hash'] for r in rows})!=1:raise AssertionError('Unpaired external tasks')
    result=dict(status='passed',component_only=True,tests=test_result.testsRun,
        formal_training_started=False,source_hashes=source_hashes(),plan_hash=digest(plan()),
        runtime=runtime_info('cpu'),profiles=rows)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'evidence/latest_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':run()

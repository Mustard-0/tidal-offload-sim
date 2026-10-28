"""Real updates and full environment/replay/RNG resume before any formal cell."""
from collections import deque
import gc
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from .protocol import METHODS,SEEDS,configuration,plan,digest,source_hashes
from .environment import Environment
from .policies import learner
from .runner import train
from experiments.lr_screen.runtime import torch,np,seed_all,json_write,runtime_info,available_ram_gib,read_csv


def fingerprint(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,np.generic):
            visit(v.item())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):
                if k=='training_wall_s':
                    continue
                visit(k);visit(v[k])
        elif isinstance(v,(list,tuple,deque)):
            h.update(type(v).__name__.encode())
            for item in v:
                visit(item)
        elif hasattr(v,'__dict__') and not isinstance(v,type):
            h.update(type(v).__qualname__.encode());visit(vars(v))
        elif isinstance(v,(str,int,float,bool,bytes,type(None))):
            h.update(repr(v).encode())
        else:
            raise TypeError('Unexpected checkpoint value '+str(type(v)))
    visit(value)
    return h.hexdigest()


class Contract(unittest.TestCase):
    def test_shared_environment_and_finite_matrix(self):
        cfg=configuration('TMAPO')[0]
        for method in METHODS:
            c,t=configuration(method)
            self.assertEqual(c,cfg)
            self.assertEqual(t['updates']*t['rollout_steps'],4*t['cycle_steps'])
            self.assertEqual(t['admission_steps']*c['dt_s'],300.)
            self.assertEqual(t['actor_cost_prior_scale'],0.)
        self.assertEqual(len(plan()['cells']),10)

    def test_no_weighted_hints_or_tm_forecast_in_baselines(self):
        for method in METHODS:
            cfg,_=configuration(method)
            e=Environment(cfg,seed=8398101,family=method)
            e.generate()
            e.previous_forecast[:]=123.
            e.pred_error.append(np.ones(3))
            context,glob,*_=e.observations()
            self.assertTrue((context[:e.u,17:19]==0).all())
            self.assertTrue((context[:e.u,26]==0).all())
            if method!='TMAPO':
                self.assertTrue((context[:e.u,4:6]==0).all())
                self.assertTrue((glob[28:34]==0).all())

    def test_truth_is_not_policy_information(self):
        cfg,_=configuration('TMAPO')
        for method in METHODS:
            e=Environment(cfg,seed=8398101,family=method)
            e.generate();before=e.observations()
            e.xy+=np.array([99999.,-99999.])
            after=e.observations()
            np.testing.assert_array_equal(before[0],after[0])
            np.testing.assert_array_equal(before[1],after[1])

    def test_report_enters_routing_and_no_blind_wait(self):
        cfg,_=configuration('TMAPO')
        for method in METHODS:
            e=Environment(cfg,seed=8398101,family=method)
            e.generate();j=next(iter(e.tasks.values()))
            j.report=np.array([1200.,2000.]);a=e.route_features(j)[0]
            j.report=np.array([2800.,2000.]);b=e.route_features(j)[0]
            self.assertFalse(np.array_equal(a,b))
            if method!='TMAPO':
                self.assertFalse(e.wait_information([j])['allow'])

    def test_prepare_does_not_duplicate_or_shift_current_arrivals(self):
        cfg,_=configuration('TMAPO')
        e=Environment(cfg,seed=8398101,family='TMAPO')
        e.prepare(True);first=(e.serial,e.arrival_hash,e.full_input_hash)
        e.time-=5e-12;e.prepare(True)
        self.assertEqual(first,(e.serial,e.arrival_hash,e.full_input_hash))
        self.assertEqual(e.time,0.)

    def test_credit_is_mean_conserving(self):
        from study.credit import mean_preserving_credit
        costs=[(i%7)*.01 for i in range(100)]
        values,team=mean_preserving_credit(costs,[34,33,33],.06,.7)
        self.assertAlmostEqual(float(np.mean(values[:100])),team)
        self.assertAlmostEqual(float(np.mean(values[100:])),team)

    def test_sac_slot_discount_and_terminal(self):
        cfg,t=configuration('SD-SAC-Adapted')
        seed_all(87021);p=learner('SD-SAC-Adapted',t,cfg)
        obs=np.zeros((2,64),np.float32);mask=np.array([[False,True,True]]*2)
        for slot in range(2):
            p.begin(obs,np.zeros(64,np.float32),2)
            p.act(obs,mask,'user',np.array([0,1]));p.remember(-.1*(slot+1),[],slot==1)
        self.assertFalse(p.pending)
        self.assertEqual([r[-1] for r in p.replay],[t['gamma'],t['gamma'],0.,0.])
        np.testing.assert_allclose([r[4] for r in p.replay],[-.1,-.1,-.2,-.2])

    def test_td3_critic_action_matches_discrete_execution(self):
        from .recent import masked_stage_action,INTENT
        logits=torch.randn(2,3,INTENT,requires_grad=True)
        legal=torch.zeros_like(logits,dtype=torch.bool)
        legal[0,0,1:3]=True;legal[0,1,6:9]=True
        action=masked_stage_action(logits,legal)
        self.assertTrue((action.detach()[~legal]==0).all())
        self.assertEqual(float(action.detach()[0,0].sum()),1.)
        self.assertEqual(float(action.detach()[0,1].sum()),1.)
        self.assertTrue((action.detach()[1]==0).all())
        (action*torch.arange(INTENT)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_td3_records_executed_actions_not_latent_intents(self):
        cfg,t=configuration('MARATD3-Adapted')
        p=learner('MARATD3-Adapted',t,cfg)
        context=np.zeros((3,64),np.float32)
        p.begin(context,np.zeros(64,np.float32),2)
        mask=np.array([[False,True,True]])
        acts,_=p.act(context[:1],mask,'user',np.array([0]))
        p.remember(-.2,[],True)
        stored=p.replay[0][1]
        np.testing.assert_array_equal(stored[0,0,:3],np.eye(3)[acts[0]])
        np.testing.assert_array_equal(stored[1,0,:3],mask[0])
        self.assertIsNone(p.pending)

    def test_wait_is_optional_causal_and_only_tm(self):
        from unittest.mock import patch
        import study.environment as module
        for method in METHODS:
            cfg,_=configuration(method)
            e=Environment(cfg,seed=8398101,family=method)
            e.generate();j=next(iter(e.tasks.values()))
            e.time=.15;e.slot=15;e.group_target=2
            e.admission_history[(j.region,j.profile)].extend(((.10,j.user),(.16,j.user)))
            # Stub the fill/gain estimators ONLY to exercise an otherwise rare
            # gate branch; never part of actual inference or evidence of gain.
            with patch.object(module,'fill_probability',return_value=.99) as fill:
                with patch.object(e,'_public_cost_samples',side_effect=lambda tasks,n,hold: [1.]*16 if n==1 else [.5]*16):
                    info=e.wait_information([j])
            self.assertEqual(info['allow'],method=='TMAPO')
            if method=='TMAPO':
                self.assertLessEqual(info['horizon_s'],.02)
                self.assertEqual(fill.call_args[0][3],1)  # future .16 excluded
            e._gate_cache={};e.group_target=1
            self.assertFalse(e.wait_information([j])['allow'])

    def test_deterministic_deployment_is_common_not_selected(self):
        from .policies import evaluation_policy,weights_only
        for method in METHODS:
            cfg,t=configuration(method)
            p=learner(method,t,cfg)
            clone=evaluation_policy(method,t,cfg,weights_only(p))
            self.assertTrue(clone.deterministic)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    (out/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    if not result.wasSuccessful():
        raise RuntimeError('Moderate contracts failed; formal training forbidden')
    profiles=[];signatures=[]
    for method in METHODS:
        if available_ram_gib()<2.:
            raise MemoryError('Insufficient safe cloud RAM for all-method gate')
        full=out/'components'/method/'full'
        split=out/'components'/method/'split'
        assert train(full,method,87021,component=True,maximum_seconds=900)
        assert not train(split,method,87021,component=True,stop_update=2,maximum_seconds=900)
        assert train(split,method,87021,component=True,resume=True,maximum_seconds=900)
        a=torch.load(full/'latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(split/'latest.pt',map_location='cpu',weights_only=False)
        self_state={k:a[k] for k in ('learner','rng','environment','update','env_steps','cycle','rows','cycles')}
        next_state={k:b[k] for k in self_state}
        assert fingerprint(self_state)==fingerprint(next_state),method+' exact resume mismatch'
        done=json.loads((full/'complete.json').read_text())
        seed_all(87021);c,t=configuration(method)
        t.update(updates=6,cycle_steps=600,admission_steps=400,episodes=1,validation_updates=())
        initial=learner(method,t,c)
        from .runner import model_hash
        assert model_hash(initial)!=done['actor_hash'] and model_hash(initial,True)!=done['critic_hash'],method+' did not really learn'
        rows=read_csv(full/'training.csv')
        # First four blocks admit actual N150/rate9 tasks; final two drain.
        seconds=[float(rows[i]['training_wall_s'])-float(rows[i-1]['training_wall_s']) for i in (2,3)]
        per_update=sum(seconds)/len(seconds)
        profiles.append(dict(method=method,actual_users=150,actor_updated=True,critic_updated=True,
            exact_midcycle_resume=True,seconds_per_100_slot_update=per_update,
            estimated_training_seconds_1208_updates=per_update*1208,
            available_ram_after_gib=available_ram_gib()))
        signatures.append([r['full_input_hash'] for r in rows])
        del initial,a,b,self_state,next_state
        gc.collect()
        print('All-method learning/resume component verified',flush=True)
    assert all(s==signatures[0] for s in signatures),'Full input prefix differs between algorithms'
    if max(r['estimated_training_seconds_1208_updates'] for r in profiles)*1.5>18000:
        raise RuntimeError('Short profile projects beyond declared training wall budget; no formal dispatch')
    value=dict(passed=True,tests=result.testsRun,profiles=profiles,all_5_real_actor_critic_updates=True,
        all_5_exact_midcycle_resume=True,full_exogenous_prefix_pairing=True,
        plan_hash=digest(plan()),source_hashes_hash=digest(source_hashes()),runtime=runtime_info('cpu'))
    json_write(out/'verification.json',value)
    return value

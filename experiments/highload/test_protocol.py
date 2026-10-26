import copy
import json
import tempfile
import time
from pathlib import Path
import unittest
from .protocol import ROOT,configuration,plan,population,ABLATIONS
from .environment import Environment
from .policies import learner,weights_only,evaluation_policy,update
from .runner import train,evaluate_one
from .baselines import CompactReplay,INTENT,OBS_DIM
from experiments.lr_screen.runtime import torch,np,seed_all
from environment import GreedyPolicy


class Contract(unittest.TestCase):
    def setUp(self):
        seed_all(87021)
        self.c,self.t=configuration('TMAPO')
        self.c=population(self.c,6)

    def test_finite_budget_and_population(self):
        p=plan()
        self.assertEqual(len(p['cells']),31)
        self.assertEqual(len(p['conditional_ablation_cells']),45)
        self.assertEqual(p['maximum_training_cells'],40)
        self.assertEqual(len(set(x['id'] for x in p['cells'])),31)
        self.assertEqual(self.t['updates']*self.t['rollout_steps'],300000)
        self.assertEqual(self.c['weight_quality'],0)
        for n in (100,200,300,400):
            self.assertEqual(sum(population(self.c,n)['region_user_counts']),n)

    def test_preparation_does_not_duplicate_current_arrivals(self):
        a=Environment(self.c,seed=814001)
        b=Environment(self.c,seed=814001)
        for e in (a,b):e.arrival_stop_slot=400
        p,q=GreedyPolicy(self.c),GreedyPolicy(self.c)
        for slot in range(300):
            if slot%100==0:
                a.prepare(True);first=a.arrivals
                a.prepare(True);self.assertEqual(first,a.arrivals)
            a.step(p,True);b.step(q,True)
        self.assertEqual(a.arrival_hash,b.arrival_hash)
        self.assertEqual(a.arrivals,b.arrivals)
        self.assertAlmostEqual(a.cost_total,b.cost_total,places=10)
        self.assertGreaterEqual(a.windows_seen,3)
        self.assertGreaterEqual(len(a.pred_error),2)

    def test_no_forecast_masks_only_information_not_workload(self):
        a=Environment(self.c,seed=814001)
        b=Environment(self.c,seed=814001,ablation='TMAPO-NoForecast')
        for e in (a,b):
            e.previous_forecast[:]=99.;e.last_observed[:]=77.;e.pred_error.append(np.ones(3))
            e.generate()
        aa=a.observations();bb=b.observations()
        self.assertEqual(a.arrival_hash,b.arrival_hash)
        np.testing.assert_array_equal(bb[0][:6,4:6],0)
        np.testing.assert_array_equal(bb[1][28:34],0)
        np.testing.assert_array_equal(aa[0][:6,3],bb[0][:6,3])
        np.testing.assert_array_equal(aa[1][:22],bb[1][:22])

    def test_report_inputs_not_true_xy(self):
        e=Environment(self.c,seed=814001)
        e.generate();a=e.observations()
        e.xy+=np.array([100.,-100.]);b=e.observations()
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(b[1][46:48],0.)

    def test_compact_replay_matches_full_histories(self):
        replay=CompactReplay(32,4)
        expected=[]
        for episode in range(3):
            h=np.zeros((5,4,OBS_DIM+INTENT),np.float32)
            for step in range(10):
                h=np.roll(h,-1,axis=1);h[:,-1]=episode*10+step+1
                a=np.full((5,INTENT),step,np.float32)
                nh=np.concatenate((h[:,1:],np.full((5,1,OBS_DIM+INTENT),episode*10+step+2,np.float32)),axis=1)
                done=step==9
                if done:nh[:]=0
                row=(h.copy(),a,1.,nh,done);expected.append(row);replay.append(row)
        for i in range(len(replay)):
            got=replay[i];ref=expected[replay.rows[replay.valid()[i]][0]]
            np.testing.assert_array_equal(got[0],ref[0]);np.testing.assert_array_equal(got[3],ref[3])
        replay.rows=type(replay.rows)(list(replay.rows)[5:],maxlen=32)
        for i in range(len(replay)):
            got=replay[i];ref=expected[replay.rows[replay.valid()[i]][0]]
            np.testing.assert_array_equal(got[0],ref[0])

    def test_all_learners_update_and_do_not_wait(self):
        for method in ('TMAPO','IPPO','PPO','SD-SAC-Adapted','MARATD3-Adapted',*ABLATIONS):
            c,t=configuration(method)
            c=population(c,6);t.update(hidden=8,rollout_steps=40,ablation=method)
            p=learner(method,t,c)
            if not hasattr(p,'actors'):p.cfg['batch']=2
            module=p.actors if hasattr(p,'actors') else p.actor
            before=[v.detach().clone() for v in module.parameters()]
            e=Environment(c,seed=814001,ablation=method)
            e.arrival_stop_slot=40
            for slot in range(40):
                _,records,_,_=e.step(p,True)
                p.remember(e.agent_rewards(0.),records,False)
            context,glob=e.prepare(True)
            stats=update(p,context,glob)
            if method=='MARATD3-Adapted':stats=update(p,context,glob)
            self.assertTrue(all(np.isfinite(float(v)) for v in stats.values()),method)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,module.parameters())),method)
            self.assertLessEqual(e.max_observed_wait,.0000001)
            clone=evaluation_policy(method,t,c,weights_only(p))
            m,_=evaluate_one(clone,c,method,814019,30.,.1,.1)
            self.assertTrue(np.isfinite(m['mean_task_cost']))
            if hasattr(clone,'replay'):self.assertEqual(len(clone.replay),0)

    def test_variable_population_replay_is_bucketed(self):
        replay=CompactReplay(64,4)
        for n in (5,8):
            for step in range(10):
                h=np.full((n,4,OBS_DIM+INTENT),step,np.float32)
                replay.append((h,np.zeros((n,INTENT),np.float32),0.,h,step==9))
        for n in (5,8):
            indexes=replay.indices_for_entities(n)
            self.assertEqual(len(indexes),10)
            self.assertEqual(np.stack([replay[i][0] for i in indexes]).shape[1],n)

    def test_exact_midcycle_resume(self):
        root=ROOT/'evidence/component_tests'/str(time.time_ns())
        root.mkdir(parents=True)
        # Preserve these bounded component receipts; no recursive cleanup.
        with self.subTest(workspace=str(root)):
            component=dict(updates=6,cycle_steps=600,drain_slots=200,
                populations=[6],hidden=8,validation_updates=[0,6],checkpoint_updates=1)
            for method in ('TMAPO','SD-SAC-Adapted','MARATD3-Adapted'):
                full=root/(method+'_full');split=root/(method+'_split')
                train(full,method,1,87021,component=component)
                train(split,method,1,87021,component=component,stop_update=2)
                paused=torch.load(split/'latest.pt',weights_only=False)
                self.assertIsNotNone(paused['environment'])
                self.assertEqual(paused['environment'].prepared_slot,200)
                train(split,method,1,87021,resume=True,component=component)
                a=torch.load(full/'latest.pt',weights_only=False)
                b=torch.load(split/'latest.pt',weights_only=False)
                key='actors' if method=='TMAPO' else 'actor'
                for name in a['learner'][key]:
                    torch.testing.assert_close(a['learner'][key][name],b['learner'][key][name],rtol=0,atol=0)
                self.assertEqual([r['trace_hash'] for r in a['rows']],[r['trace_hash'] for r in b['rows']])
                self.assertEqual([r['rollout_return'] for r in a['rows']],[r['rollout_return'] for r in b['rows']])


if __name__=='__main__':unittest.main(verbosity=2)

"""Real learning, privacy, observation separation and checkpoint continuation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from experiments.lr_screen.runtime import np,torch,seed_all
from experiments.lr_screen.algorithms import learner_for
from .protocol import configuration
from .environment import Environment
from .runner import episode,new_environment,train,actor_hash


class CoreTests(unittest.TestCase):
    def test_all_three_update_their_actors(self):
        for method in ('TMAPO','IPPO','PPO'):
            seed_all(4701);c,t=configuration(method)
            p=learner_for(method,t,c,'cpu');before=actor_hash(p)
            e=new_environment(c,method,87021,0)
            reward,stats=episode(e,p,learning=True)
            self.assertNotEqual(before,actor_hash(p))
            self.assertTrue(np.isfinite(reward));self.assertFalse(e.tasks)
            self.assertEqual(e.arrivals,e.ended)
            self.assertLessEqual(e.max_observed_wait,.02+1e-9)
            self.assertGreaterEqual(e.windows_seen,9)
            self.assertFalse(p.buffer)

    def test_basic_methods_do_not_get_forecasts_or_cost_hints(self):
        for method in ('IPPO','PPO'):
            c,t=configuration(method);e=Environment(c,seed=44,family=method)
            e.last_observed[:]=2.;e.previous_forecast[:]=10.;e.pred_error.append(np.ones(3))
            e.generate();obs,glob,*_=e.observations()
            self.assertFalse(np.any(obs[:e.u,4:6]))
            self.assertFalse(np.any(obs[:e.u,17:19]))
            self.assertFalse(np.any(obs[:e.u,26]))
            self.assertFalse(np.any(glob[28:34]));self.assertFalse(np.any(glob[46:48]))
            self.assertGreater(np.abs(e.previous_forecast).sum(),0.)

    def test_TMAPO_really_has_forecast_channels(self):
        c,t=configuration('TMAPO');e=Environment(c,seed=44)
        e.previous_forecast[:]=10.;e.generate();obs,glob,*_=e.observations()
        self.assertTrue(np.any(obs[:e.u,4]));self.assertTrue(np.any(glob[28:31]))
        self.assertFalse(np.any(obs[:e.u,17:19]))

    def test_no_true_xy_enters_pre_report_actor(self):
        c,_=configuration('TMAPO');e=Environment(c,seed=45)
        e.generate();a=e.observations()
        e.xy+=np.array([10.,-12.]);b=e.observations()
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_report_coordinates_really_change_route_inputs(self):
        c,_=configuration('TMAPO');e=Environment(c,seed=46)
        e.generate();j=next(iter(e.tasks.values()))
        j.report=np.asarray(c['region_centers_m'][j.region],float);j.batch_n=1;j.epsilon=.001
        a=e.route_features(j)[0];j.report=j.report+np.array([200.,0.]);b=e.route_features(j)[0]
        self.assertFalse(np.array_equal(a,b))

    def test_tiny_batches_cannot_invent_amplification(self):
        c,_=configuration('TMAPO');e=Environment(c,seed=1)
        for n in (1,2,5,10,20):self.assertEqual(e.eps0(n),1.)
        self.assertGreater(e.group_target,20)

    def test_wait_gates_fail_closed_without_history_and_at_drain(self):
        c,_=configuration('TMAPO');e=Environment(c,seed=1)
        j=SimpleNamespace(id=0,user=0,region=0,profile='balanced',deadline=.35,arrival=0.,active_wait=0.)
        self.assertFalse(e.wait_information([j])['allow'])
        e.draining=True;self.assertEqual(e.wait_information([j])['reason'],'no_new_arrivals')
        for method in ('IPPO','PPO'):
            other=Environment(c,seed=1,family=method)
            self.assertEqual(other.wait_information([j])['reason'],'basic_immediate')

    def test_public_physical_wait_model_has_no_truth_dependency(self):
        c,_=configuration('TMAPO');e=Environment(c,seed=47)
        e.generate();j=next(iter(e.tasks.values()))
        a=e._public_cost_samples([j],1,0.)
        e.xy+=np.array([100.,100.])
        b=e._public_cost_samples([j],1,0.)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.isfinite(a).all())

    def test_same_episode_has_identical_exogenous_tasks(self):
        signatures=[]
        for method in ('TMAPO','IPPO','PPO'):
            c,t=configuration(method);seed_all(22)
            e=new_environment(c,method,87021,0,collect=True)
            p=learner_for(method,t,c,'cpu');episode(e,p)
            signatures.append((e.arrival_hash,[(r['task_id'],r['user'],r['arrival_s'],r['task_bits'],
                r['task_cycles'],r['deadline_s']) for r in sorted(e.rows,key=lambda r:r['task_id'])]))
        self.assertEqual(signatures[0],signatures[1]);self.assertEqual(signatures[0],signatures[2])

    def test_checkpoint_resume_exact_for_all_algorithms(self):
        for method in ('TMAPO','IPPO','PPO'):
            with tempfile.TemporaryDirectory() as d:
                a,b=Path(d)/'a',Path(d)/'b'
                train(a,method,87021,episodes=4,component=True)
                self.assertFalse(train(b,method,87021,episodes=4,stop_at=2,component=True))
                train(b,method,87021,episodes=4,resume=True,component=True)
                x=torch.load(a/'latest.pt',weights_only=False,map_location='cpu')
                y=torch.load(b/'latest.pt',weights_only=False,map_location='cpu')
                self.assertEqual(x['env_steps'],y['env_steps'])
                for name in ('actors','critic'):
                    for k in x['learner'][name]:
                        self.assertTrue(torch.equal(x['learner'][name][k],y['learner'][name][k]))
                self.assertEqual([r['trace_hash'] for r in x['rows']],[r['trace_hash'] for r in y['rows']])


if __name__=='__main__':unittest.main()

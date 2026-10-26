"""Small transfer compatibility and RL integration checks, never a LR screen."""
import copy
import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from .runtime import ROOT, np, seed_all
from .protocol import configuration, LRS
from .mechanism import actual_batch, report_for
from .restricted_environment import CandidateRestrictedEnvironment
from .vendor.accountant import calibrate, delta_bound

TRANSFER = ROOT / 'evidence/transfer_source_20261025/mechanism_transfer'


class MechanismChecks(unittest.TestCase):
    def setUp(self):
        seed_all(87021)
        self.cfg, _ = configuration('TMAPO', 1)

    def test_exact_original_transfer_reports_and_accountant(self):
        # Separate process prevents greedy reference imports shadowing RL physics.
        code = """
import json
from transfer import report_for, calibrate
out=[]
for n in range(1,101):
 for s,m in [(0.5,'Shrink05'),(0.75,'Shrink075')]:
  e=calibrate(n,1.,1e-6)
  raw,y=report_for([2100.,2250.],[2000.,2000.],500.,e,17,n,m)
  out.append([n,s,e,raw.tolist(),y.tolist()])
print(json.dumps(out))
"""
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT/'.runtime')
        rows = json.loads(subprocess.check_output([sys.executable, '-c', code],
                          cwd=TRANSFER, env=env, text=True, timeout=120))
        self.assertEqual(len(rows), 200)
        for n, s, e, raw, y in rows:
            self.assertEqual(e, calibrate(n, 1., 1e-6))
            self.assertLessEqual(delta_bound(e, n, 1.), 1e-6+1e-12)
            a, b = report_for([2100.,2250.], [2000.,2000.], 500., e, 17, n, s)
            np.testing.assert_array_equal(a, raw)
            np.testing.assert_array_equal(b, y)
        self.assertEqual(calibrate(1), 1.)

    def test_batch_membership_duplicates_and_network_count_rejected(self):
        tasks = [SimpleNamespace(id=i, user=i, region=0, profile='a') for i in range(100)]
        self.assertEqual(actual_batch(tasks, self.cfg)[0], 100)
        for bad in ([], tasks+[tasks[0]], tasks+[SimpleNamespace(id=100,user=100,region=0,profile='a')],
                    [tasks[0], SimpleNamespace(id=1,user=100,region=1,profile='a')],
                    [tasks[0], SimpleNamespace(id=1,user=1,region=0,profile='b')]):
            with self.assertRaises(ValueError):
                actual_batch(bad, self.cfg)
        env = CandidateRestrictedEnvironment(self.cfg, seed=3)
        with self.assertRaises(ValueError):
            env.eps0(300)

    def test_sampling_bounds_and_invalid_truth(self):
        for truth in ([0.,0.], [500.,0.], [-300.,400.]):
            for seed in range(20):
                raw, y = report_for(truth, [0.,0.], 500., 1., seed, 3)
                self.assertLessEqual(np.linalg.norm(raw), 500.+1e-9)
                self.assertLessEqual(np.linalg.norm(y), 250.+1e-9)
                self.assertLessEqual(np.linalg.norm(y-truth), 750.+1e-9)
        with self.assertRaises(ValueError):
            report_for([501.,0.], [0.,0.], 500., 1., 1, 1)

    def test_ql_diagnostic_not_reward(self):
        env = CandidateRestrictedEnvironment(self.cfg, seed=3)
        env.generate()
        j = next(iter(env.tasks.values()))
        a, b = copy.deepcopy(env), copy.deepcopy(env)
        center = np.asarray(self.cfg['region_centers_m'][j.region])
        with patch('experiments.lr_screen.restricted_environment.report_for', return_value=(center, center)):
            a.release([a.tasks[j.id]])
        alternative = center+np.array([200.,0.])
        with patch('experiments.lr_screen.restricted_environment.report_for', return_value=(alternative, alternative)):
            b.release([b.tasks[j.id]])
        self.assertNotEqual(a.tasks[j.id].ql, b.tasks[j.id].ql)
        self.assertEqual(a.step_cost, b.step_cost)
        self.assertEqual(a.cost_total, b.cost_total)
        self.assertEqual(a.cfg['weight_quality'], 0.)

    def test_no_future_read_or_ql_wait_and_clock_guard(self):
        env = CandidateRestrictedEnvironment(self.cfg, seed=3)
        env.generate()
        j = next(iter(env.tasks.values()))
        with patch.object(env, 'q_estimate', side_effect=AssertionError('QL used as wait benefit')):
            self.assertFalse(env.wait_information([j])['allow'])
        env.time = .001
        with self.assertRaisesRegex(AssertionError, 'clock'):
            env.step(None)

    def test_all_eleven_share_physics_reward_and_budget(self):
        reference, common = configuration('TMAPO', 1)
        for method, rates in LRS.items():
            for i in range(1, len(rates)+1):
                cfg, training = configuration(method, i)
                self.assertEqual(cfg, reference)
                self.assertEqual(cfg['weight_quality'], 0.)
                for field in ('actor_lr', 'critic_lr'):
                    training.pop(field)
                expected = {k:v for k,v in common.items() if k not in ('actor_lr','critic_lr')}
                self.assertEqual(training, expected)

    def test_bounded_update_path_at_each_registered_learning_rate(self):
        from .algorithms import learner_for
        for method, rates in LRS.items():
            for i in range(1, len(rates)+1):
                seed_all(87021)
                cfg, training = configuration(method, i)
                learner = learner_for(method, training, cfg)
                env = CandidateRestrictedEnvironment(cfg, seed=20261026, phase_s=30.)
                for update in range(4):
                    for _ in range(64):
                        _, records, _, _ = env.step(learner)
                        learner.remember(env.agent_rewards(0.), records, False)
                    ctx, glob, *_ = env.observations()
                    stats = learner.update(ctx, glob)
                    self.assertTrue(all(np.isfinite(v) for v in stats.values()), (method,i,update))
                print('COMPONENT_ONLY', method, i, 'env_slots', env.slot,
                      'kl', stats['kl'], 'value_loss', stats['value_loss'], flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)

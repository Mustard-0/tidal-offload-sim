"""Necessary component checks; NOT formal training or transfer verification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .runtime import ROOT, np, torch, seed_all, rng_state, rng_load
from .protocol import configuration, plan
from .algorithms import learner_for, CentralPPO, DecentralizedPPO
from .restricted_environment import CandidateRestrictedEnvironment, expected_quality
from .vendor.accountant import calibrate, delta_bound
from .readiness import require_transfer, TransferUnavailable
from .runner import train_cell, evaluate, pid_running
from environment import Environment, Task


class Immediate:
    """Explicit TEST rule; never labeled as TMAPO."""
    deterministic = True
    is_rule_based = True
    def begin(self, *args):
        pass
    def act(self, obs, mask, stage, agents):
        if stage == "route":
            cost = obs[:, :11]+obs[:, 11:22]
            return np.where(mask, cost, np.inf).argmin(axis=1), []
        return np.asarray([2 if m[2] else 1 for m in mask]), []


class Components(unittest.TestCase):
    def setUp(self):
        seed_all(87021)
        self.cfg, self.training = configuration("TMAPO", 2)

    def test_exact_grid_and_gate(self):
        p = plan()
        self.assertEqual([r["algorithm"] for r in p["cells"]].count("TMAPO"), 5)
        self.assertEqual(len(p["cells"]), 11)
        self.assertEqual({r["env_steps"] for r in p["cells"]}, {256000})
        self.assertEqual(p["total_training_env_steps"], 2816000)
        self.assertTrue(all(abs(r["critic_lr"]/r["actor_lr"]-3)<1e-12 for r in p["cells"]))
        with patch("experiments.lr_screen.readiness.APPROVAL", ROOT/"nonexistent_preflight.json"):
            with self.assertRaises(TransferUnavailable):
                require_transfer()

    def test_no_data_report_keeps_blocked_state_and_does_not_invent_curves(self):
        from .runtime import json_write
        from .report import report
        base = ROOT/"evidence/component_checks"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            out = Path(temporary)
            json_write(out/"plan.json", plan())
            report(out)
            self.assertFalse(list(out.glob("*.png")))
            self.assertFalse(json.loads((out/"report_status.json").read_text())["curves_generated"])
            self.assertEqual((out/"status.csv").read_text().count("not_started_pending_preflight"), 11)

    def test_accountant_matches_existing_code_and_true_n(self):
        from privacy import calibrate as original_calibrate
        for n in (1, 2, 5, 16, 32, 100):
            e = calibrate(n, 1., 1e-6)
            self.assertEqual(e, original_calibrate(n, 1., 1e-6))
            self.assertLessEqual(delta_bound(e, n, 1.), 1e-6+1e-12)
        self.assertEqual(calibrate(1), 1.)
        env = CandidateRestrictedEnvironment(self.cfg, seed=17)
        with self.assertRaises(ValueError):
            env.eps0(300)

    def test_no_planar_fallback_reward_queue_and_trace_pairing(self):
        trace = set()
        for method in ("TMAPO", "IPPO", "PPO"):
            learner = learner_for(method, self.training, self.cfg)
            learner.deterministic = True
            with patch.object(Environment, "release", side_effect=AssertionError("old sampler called")):
                m, env = evaluate(learner, self.cfg, 814001, 30., 0, 24)
            trace.add(m["trace_hash"])
            self.assertEqual(env.u, 300)
            self.assertFalse(env.tasks)
            self.assertLessEqual(m["max_active_wait_s"], .02+1e-9)
            self.assertTrue(all(r["n_distinct"] <= 100 for r in env.batch_rows))
            self.assertGreater(m["published"], 0)
            self.assertFalse(m["quality_cost_is_in_reward"])
            self.assertEqual(m["objective_quality"], 0.)
            components = sum(m[k] for k in ("objective_time", "objective_energy", "objective_quality", "objective_qos"))
            self.assertAlmostEqual(components, m["mean_task_cost"], places=8)
        self.assertEqual(len(trace), 1)

    def test_report_used_and_true_coordinates_not_in_observations(self):
        env = CandidateRestrictedEnvironment(self.cfg, seed=24)
        env.generate()
        u = next(i for i, q in enumerate(env.pending) if q)
        j = env.tasks[env.pending[u].popleft()]
        j.status = "selected"
        env.release([j])
        center = np.asarray(self.cfg["region_centers_m"][j.region])
        self.assertLessEqual(np.linalg.norm(j.report-center), 250.+1e-7)
        obs1, mask1, _ = env.route_features(j)
        all1 = env.observations()
        env.xy[:] = center  # physics truth changes, public state stays fixed
        obs2, mask2, _ = env.route_features(j)
        all2 = env.observations()
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(mask1, mask2)
        for a, b in zip(all1[:3], all2[:3]):
            np.testing.assert_array_equal(a, b)
        j.report = center+np.asarray([200., 0.])
        changed, _, _ = env.route_features(j)
        self.assertFalse(np.array_equal(obs1[:22], changed[:22]))
        np.testing.assert_allclose(changed[46:48]*10000, j.report, rtol=1e-6)

    def test_wait_gates_empty_history_sufficient_batch_deadline_and_limit(self):
        env = CandidateRestrictedEnvironment(self.cfg, seed=18)
        env.generate()
        j = next(iter(env.tasks.values()))
        self.assertFalse(env.wait_information([j])["allow"])
        env.time = 2.
        j.arrival = 1.99
        self.assertFalse(env.wait_information([j])["allow"])
        j.active_wait = .02
        self.assertFalse(env.wait_information([j])["allow"])
        j.active_wait = 0.
        j.deadline = .01
        self.assertFalse(env.wait_information([j])["allow"])
        j.deadline = .5
        sufficient = [copy.copy(j) for _ in range(env.group_target)]
        for i, task in enumerate(sufficient):
            task.user = i
        self.assertFalse(env.wait_information(sufficient)["allow"])
        self.assertGreater(expected_quality(1.), expected_quality(2.))

    def test_all_methods_update_actor_and_critic(self):
        for method in ("TMAPO", "IPPO", "PPO"):
            seed_all(87021)
            learner = learner_for(method, self.training, self.cfg)
            env = CandidateRestrictedEnvironment(self.cfg, seed=20261025)
            before_a = copy.deepcopy(learner.actors.state_dict())
            before_v = copy.deepcopy(learner.critic.state_dict())
            for _ in range(32):
                reward, records, _, _ = env.step(learner)
                learner.remember(env.agent_rewards(0.), records, False)
            ctx, glob, *_ = env.observations()
            stats = learner.update(ctx, glob)
            self.assertTrue(any(not torch.equal(before_a[k], v) for k, v in learner.actors.state_dict().items()), method)
            self.assertTrue(any(not torch.equal(before_v[k], v) for k, v in learner.critic.state_dict().items()), method)
            self.assertTrue(all(np.isfinite(x) for x in stats.values()))
            self.assertEqual(learner.critic[0].in_features, 128 if method == "TMAPO" else 64)
            self.assertEqual(learner.actors["route"][0].in_features, 128 if method == "PPO" else 64)

    def test_terminal_does_not_bootstrap_future_value(self):
        learner = CentralPPO(self.training, self.cfg)
        learner.buffer = [(np.zeros(64, dtype=np.float32), 500., 2., [], True)]
        # With done=True, target = reward, regardless of current/next V.
        learner.raw_value = lambda g: 1e9
        learner.update(np.zeros((303, 64)), np.zeros(64))
        self.assertAlmostEqual(learner.value_norm.mean, 2.)

    def test_joint_likelihood_is_sum_of_decision_log_ratios(self):
        # A one-slot fixture with two executed factors: KL must be that of
        # the joint ratio (2*2), not an average of two independent ratios.
        learner = CentralPPO(self.training, self.cfg)
        learner.begin(np.zeros((303, 64)), np.zeros(64), 300)
        obs = np.zeros((2, 64), dtype=np.float32)
        _, records = learner.act(obs, np.ones((2, 3), bool), "user", np.array([0, 1]))
        for row in records:
            row["logp"] -= math_log_two()
        learner.remember(np.zeros(303), records, True)
        stats = learner.update(np.zeros((303, 64)), np.zeros(64))
        self.assertAlmostEqual(stats["kl"], 4-1-2*math_log_two(), places=5)

    def test_checkpoint_resume_matches_uninterrupted_for_all_methods(self):
        settings = dict(updates=2, rollout_steps=8, episode_steps=240, validation_interval=2,
                        checkpoint_interval=1, validation_seeds=[814019], validation_phases=[30.],
                        validation_warmup_s=0., validation_measure_s=.08)
        base = ROOT/"evidence/component_checks"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            for method in ("TMAPO", "IPPO", "PPO"):
                a, b = Path(temporary)/(method+"_full"), Path(temporary)/(method+"_resume")
                self.assertTrue(train_cell(a, method, 1, component_settings=settings))
                self.assertFalse(train_cell(b, method, 1, stop_after=1, component_settings=settings))
                self.assertTrue(train_cell(b, method, 1, resume=True, component_settings=settings))
                x = torch.load(a/"latest.pt", weights_only=False)
                y = torch.load(b/"latest.pt", weights_only=False)
                for network in ("actors", "critic"):
                    for key in x["learner"][network]:
                        torch.testing.assert_close(x["learner"][network][key], y["learner"][network][key], rtol=0, atol=0)
                self.assertEqual(x["environment"].arrival_hash, y["environment"].arrival_hash)
                self.assertEqual(x["rows"][-1]["cost_total"], y["rows"][-1]["cost_total"])
                self.assertEqual(x["selected"], y["selected"])
                self.assertTrue((b/"recovery").exists())

    def test_current_pid_is_alive(self):
        import os
        self.assertTrue(pid_running(os.getpid()))

    def test_cpu_checkpoint_does_not_initialize_cuda(self):
        with patch.object(torch.cuda, 'is_initialized', return_value=False), \
             patch.object(torch.cuda, 'get_rng_state_all', side_effect=AssertionError('Unneeded CUDA initialization')):
            self.assertEqual(rng_state()['cuda'], [])

    def test_crash_recovery_and_tampered_csv_rejection(self):
        settings = dict(updates=2, rollout_steps=8, episode_steps=240, validation_interval=2,
                        checkpoint_interval=1, validation_seeds=[814019], validation_phases=[30.],
                        validation_warmup_s=0., validation_measure_s=.08)
        base = ROOT/"evidence/component_checks"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            out = Path(temporary)/"crash"
            original = CentralPPO.update
            calls = []
            def fail_second(obj, *args):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("injected interrupted optimizer")
                return original(obj, *args)
            with patch.object(CentralPPO, "update", fail_second):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    train_cell(out, "PPO", 1, component_settings=settings)
            self.assertTrue((out/"failure.json").exists())
            self.assertEqual(torch.load(out/"latest.pt", weights_only=False)["update"], 1)
            self.assertTrue(train_cell(out, "PPO", 1, resume=True, component_settings=settings))
            self.assertTrue(any((out/"recovery").glob("*/failure.json")))
            bad = Path(temporary)/"corrupt_csv"
            train_cell(bad, "PPO", 1, stop_after=1, component_settings=settings)
            raw = (bad/"train.csv").read_text(encoding="utf-8")
            # Alter the canonical environment step field of the saved prefix.
            raw = raw.replace("1,8,", "1,999,", 1)
            (bad/"train.csv").write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Raw CSV was modified"):
                train_cell(bad, "PPO", 1, resume=True, component_settings=settings)


def math_log_two():
    import math
    return math.log(2.)


if __name__ == "__main__":
    unittest.main(verbosity=2)

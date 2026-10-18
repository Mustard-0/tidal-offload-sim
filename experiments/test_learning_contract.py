"""Analytic contracts are unit fixtures, not a second simulation environment."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "model")]
from paper_convergence import configuration, learner_for
from cross_slot_environment import CrossSlotWLOEnvironment
from cross_slot_wlo_train import CrossSlotRule
from environment import OBS_DIM, Task
from test_paper_convergence import settings


class OneHold(CrossSlotRule):
    def __init__(self, cfg, hold):
        super().__init__("immediate_remote", cfg)
        self.hold = hold

    def act(self, obs, mask, stage, agents):
        if stage == "group" and self.hold and mask[0, 0]:
            self.hold -= 1
            return np.asarray([0]), []
        if stage == "route":
            ground = np.flatnonzero(mask[0, :8])
            if not len(ground):
                raise AssertionError("Contract fixture requires a feasible ground node")
            return np.asarray([ground[0]]), []
        return super().act(obs, mask, stage, agents)


def one_task(hold):
    cfg, _ = configuration(dict(settings(), training_profile="credit_v3"))
    env = CrossSlotWLOEnvironment(cfg, seed=17, phase_s=30., collect=True)
    task = Task(0, 0, 0, 0., 50000., 80e6, .2, profile="interactive", arrival_slot=0)
    env.tasks[0] = task
    env.pending[0].append(0)
    env.arrivals = env.measure_arrivals = env.serial = 1
    policy = OneHold(cfg, hold)
    policy.env = env
    rewards = []
    for _ in range(220):
        reward, *_ = env.step(policy, arrivals=False)
        rewards.append(reward)
        if not env.tasks:
            break
    if env.tasks:
        raise AssertionError("Single task did not drain")
    return env, rewards


class LearningContractTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(20261018)
        torch.manual_seed(20261018)
        torch.set_num_threads(1)

    def test_one_slot_wait_adds_exact_physical_cost_without_amplification(self):
        a, ra = one_task(0)
        b, rb = one_task(1)
        self.assertEqual(len(a.rows), 1)
        self.assertEqual(len(b.rows), 1)
        self.assertEqual(a.rows[0]["qos"], 0)
        self.assertEqual(b.rows[0]["qos"], 0)
        self.assertAlmostEqual(a.rows[0]["energy_j"], b.rows[0]["energy_j"])
        self.assertEqual(a.rows[0]["ql_m"], b.rows[0]["ql_m"])
        self.assertEqual(a.rows[0]["batch_n"], 1)
        self.assertEqual(b.rows[0]["batch_n"], 1)
        expected = a.cfg["weight_time"] * a.cfg["dt_s"] / a.cfg["time_ref_s"]
        self.assertAlmostEqual(b.cost_total - a.cost_total, expected, places=10)
        self.assertGreater(sum(ra), sum(rb))

    def fixture(self, method, diagnostics=False):
        cfg, train = configuration(dict(settings(), training_profile="credit_v3"))
        train.update(critic_lr=0., value_normalization=False, actor_cost_prior_scale=0.,
                     entropy=0., actor_warmup_steps=0, epochs=1, minibatch=256,
                     record_advantage_diagnostics=diagnostics)
        learner = learner_for(method, train, cfg)
        for parameter in learner.critic.parameters():
            parameter.data.zero_()
        context = np.zeros((cfg["users_per_region"] * 3 + 3, OBS_DIM), np.float32)
        glob = np.zeros(OBS_DIM, np.float32)
        return learner, context, glob

    def fill_terminal_bandit(self, learner, context, glob):
        obs = np.zeros(OBS_DIM, np.float32)
        obs[63] = -1/3
        mask = np.asarray([True, False, True])
        for i in range(64):
            action = 0 if i % 2 == 0 else 2
            learner.begin(context, glob, len(context)-3)
            record = dict(stage="user", obs=obs.copy(), mask=mask.copy(), agent=len(context)-3,
                          action=action, logp=-np.log(2.))
            cost = 1. if action == 0 else .1
            learner.remember(np.full(len(context), -cost, np.float32), [record], True)
        return obs, mask

    def test_ppo_increases_probability_of_lower_cost_action(self):
        for method in ("TMPPO", "IPPO"):
            learner, context, glob = self.fixture(method)
            obs, mask = self.fill_terminal_bandit(learner, context, glob)
            before = learner.policy_logits("user", learner.tensor(obs[None])).detach()
            learner.update(context, glob)
            after = learner.policy_logits("user", learner.tensor(obs[None])).detach()
            self.assertGreater(float(after[0, 2] - after[0, 0]),
                               float(before[0, 2] - before[0, 0]))

    def test_gae_matches_terminal_hand_calculation_and_ignores_bootstrap(self):
        learner, context, glob = self.fixture("TMPPO")
        learner.cfg["gae_lambda"] = 1.
        learner.cfg["actor_lr"] = 0.
        for group in learner.aopt.param_groups:
            group["lr"] = 0.
        for reward, done in ((-1., False), (-2., False), (-3., True)):
            learner.begin(context, glob, len(context)-3)
            learner.remember(np.full(len(context), reward, np.float32), [], done)
        captured = []
        original = learner.normalize_targets
        with patch.object(learner, "normalize_targets", side_effect=lambda x: (captured.append(x.copy()), original(x))[1]):
            with patch.object(learner, "raw_values", return_value=np.full(len(context), 123.)):
                learner.update(context, glob)
        targets = captured[0].reshape(3, len(context))
        np.testing.assert_allclose(targets[:, 0], [-3., -5., -6.])

    def test_diagnostics_do_not_change_actor_or_critic_update(self):
        a, context, glob = self.fixture("TMPPO")
        self.fill_terminal_bandit(a, context, glob)
        b = copy.deepcopy(a)
        b.cfg["record_advantage_diagnostics"] = True
        state = torch.get_rng_state()
        x = a.update(context, glob)
        torch.set_rng_state(state)
        y = b.update(context, glob)
        for k, v in a.actors.state_dict().items():
            torch.testing.assert_close(v, b.actors.state_dict()[k], atol=0, rtol=0)
        for k, v in a.critic.state_dict().items():
            torch.testing.assert_close(v, b.critic.state_dict()[k], atol=0, rtol=0)
        self.assertNotIn("group_raw_adv_std", x)
        self.assertGreater(y["group_raw_adv_std"], 0.)

    def test_ablation_changes_no_physics_reward_discount_or_learning_rates(self):
        base, train = configuration(dict(settings(), training_profile="credit_v3"))
        for variant in ("control", "trace1", "noentropy", "trace1_noentropy"):
            actual, t = configuration(dict(settings(), training_profile="contract_v4",
                                           diagnostic_variant=variant))
            self.assertEqual(actual, base)
            for key in train:
                if key not in ("entropy", "gae_lambda"):
                    self.assertEqual(t[key], train[key])
            self.assertEqual(t["gamma"], 1.)
        with self.assertRaises(ValueError):
            configuration(dict(settings(), training_profile="contract_v4", diagnostic_variant="picked"))

    def test_campaign_registry_has_bounded_budget_and_no_heldout_seeds(self):
        from learning_contract_campaign import registered_settings, VARIANTS
        self.assertEqual(len(VARIANTS), 4)
        cases = [registered_settings(v) for v in VARIANTS]
        self.assertEqual(sum(c["updates"] * c["rollout"] for c in cases), 65536)
        self.assertTrue(all(c["seed"] == 87021 for c in cases))
        self.assertTrue(all(c["baseline_diagnostics"] is False for c in cases))
        self.assertEqual(registered_settings("control", 87035)["seed"], 87035)
        with self.assertRaises(ValueError):
            registered_settings("control", 9848031)


if __name__ == "__main__":
    unittest.main()

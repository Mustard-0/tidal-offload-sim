"""Finite-horizon value inputs, common actor burn-in, and unchanged physics."""
import copy
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"experiments"), str(ROOT/"model")]
from paper_convergence import configuration, learner_for, train_cell
from cross_slot_environment import CrossSlotWLOEnvironment
from cross_slot_learners import training_clock
from environment import OBS_DIM
from hybrid_campaign import extract_evidence
from test_paper_convergence import settings


class CreditV3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        np.random.seed(91)
        self.settings = dict(settings(), training_profile="credit_v3")
        self.cfg, self.train = configuration(self.settings)
        self.cfg["users_per_region"] = 2

    def test_environment_reward_privacy_and_learning_rates_are_unchanged(self):
        old, _ = configuration(dict(self.settings, training_profile="team_v2"))
        new, train = configuration(self.settings)
        self.assertEqual(old, new)
        self.assertEqual(train["local_credit_fraction"], 0.)
        self.assertEqual(train["actor_warmup_steps"], 2048)
        self.assertEqual(train["critic_clock_features"], 2)
        self.assertEqual(train["actor_lr"], 1e-4)
        self.assertEqual(train["critic_lr"], 3e-4)

    def test_horizon_is_normalized_and_rejects_invalid_values(self):
        np.testing.assert_allclose(training_clock(200, 100, 1000), [.2, .1])
        for args in ((1, 2, 3), (-1, 0, 10), (2, 1, 0)):
            with self.assertRaises(ValueError):
                training_clock(*args)

    def test_critic_can_distinguish_horizons_without_leaking_into_actor(self):
        context = np.zeros((9, OBS_DIM), np.float32)
        glob = np.zeros(OBS_DIM, np.float32)
        obs = np.ones((1, OBS_DIM), np.float32)
        for method, width in (("TMPPO", 2*OBS_DIM+2), ("IPPO", OBS_DIM+2)):
            learner = learner_for(method, self.train, self.cfg)
            learner.set_training_clock(1000, 800, 1000)
            a = learner.critic_input(context, glob)
            logits = learner.policy_logits("user", learner.tensor(obs)).detach().clone()
            learner.set_training_clock(10, 0, 1000)
            b = learner.critic_input(context, glob)
            self.assertEqual(a.shape[-1], width)
            np.testing.assert_equal(a[:, :-2], b[:, :-2])
            self.assertFalse(np.array_equal(a[:, -2:], b[:, -2:]))
            torch.testing.assert_close(logits, learner.policy_logits("user", learner.tensor(obs)))

    def test_actor_burn_in_trains_critics_only_for_all_methods(self):
        for method in ("TMPPO", "IPPO", "MARATD3-Adapted"):
            train = dict(self.train, actor_warmup_steps=128, td3_start_steps=4,
                         td3_batch=2, rollout_steps=8)
            learner = learner_for(method, train, self.cfg)
            actor = learner.actor if method == "MARATD3-Adapted" else learner.actors
            critic = learner.q if method == "MARATD3-Adapted" else learner.critic
            before_actor = copy.deepcopy(actor.state_dict())
            before_critic = copy.deepcopy(critic.state_dict())
            env = CrossSlotWLOEnvironment(self.cfg, seed=17)
            for _ in range(32):
                learner.set_training_clock(256-env.slot, max(0, 55-env.slot), 256)
                _, records, *_ = env.step(learner)
                learner.remember(env.agent_rewards(0.), records, False)
            learner.set_training_clock(224, 23, 256)
            ctx, glob, *_ = env.observations()
            diagnostic = learner.update(ctx, glob)
            self.assertEqual(diagnostic["actor_frozen"], 1.)
            self.assertTrue(all(torch.equal(before_actor[k], v) for k, v in actor.state_dict().items()))
            self.assertTrue(any(not torch.equal(before_critic[k], v) for k, v in critic.state_dict().items()))

    def test_td3_replay_keeps_event_specific_critic_clocks(self):
        learner = learner_for("MARATD3-Adapted", self.train, self.cfg)
        ctx, glob = np.zeros((9, OBS_DIM)), np.zeros(OBS_DIM)
        obs, mask = np.zeros((1, OBS_DIM)), np.ones((1, 3), bool)
        learner.set_training_clock(256, 55, 256)
        learner.begin(ctx, glob, 6)
        learner.act(obs, mask, "user", [0])
        learner.remember(np.ones(9), [], False)
        learner.set_training_clock(255, 54, 256)
        learner.begin(ctx, glob, 6)
        learner.act(obs, mask, "user", [0])
        transition = learner.replay[0]
        self.assertEqual(transition["history"].shape[-1], OBS_DIM+16)
        self.assertEqual(len(transition["global_obs"]), OBS_DIM+2)
        self.assertEqual(float(transition["global_obs"][-2]), 1.)
        self.assertLess(float(transition["next_global"][-2]), 1.)

    def test_clock_and_actor_warmup_position_survive_checkpoint(self):
        for method in ("TMPPO", "IPPO", "MARATD3-Adapted"):
            learner = learner_for(method, self.train, self.cfg)
            learner.steps = 113
            learner.set_training_clock(200, 10, 256)
            clone = learner_for(method, self.train, self.cfg)
            clone.load(copy.deepcopy(learner.state()))
            self.assertEqual(clone.steps, 113)
            np.testing.assert_equal(clone.training_clock, learner.training_clock)

    def test_ppo_profile_pause_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root/"full", root/"resumed"
            self.assertTrue(train_cell(a, self.settings, wall_seconds=120))
            self.assertFalse(train_cell(b, self.settings, wall_seconds=120, stop_after=1))
            self.assertTrue(train_cell(b, self.settings, resume=True, wall_seconds=120))
            x, y = (torch.load(p/"latest.pt", weights_only=False) for p in (a, b))
            for key in ("actors", "critic"):
                for name in x["learner"][key]:
                    torch.testing.assert_close(x["learner"][key][name], y["learner"][key][name], atol=0, rtol=0)
            self.assertEqual(x["validation"], y["validation"])

    def test_deep_windows_archive_extraction_retains_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/"input.zip"
            member = "/".join(["nested0123456789"]*15+["evidence.txt"])
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr(member, "unchanged")
            destination = root/"out"
            extract_evidence(archive, destination)
            absolute = str((destination/member).resolve())
            if sys.platform == "win32":
                absolute = "\\\\?\\"+absolute
            self.assertEqual(Path(absolute).read_text(), "unchanged")
            # tempfile's ordinary-path cleanup cannot descend MAX_PATH trees.
            if sys.platform == "win32":
                self.assertTrue(destination.resolve().is_relative_to(root.resolve()))
                shutil.rmtree("\\\\?\\"+str(destination.resolve()))


if __name__ == "__main__":
    unittest.main()

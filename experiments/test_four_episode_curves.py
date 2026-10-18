import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"experiments"), str(ROOT/"model")]
from four_episode_curves import settings, trailing_mean, build_plan
import paper_convergence as pc
from environment import OBS_DIM


class FourEpisodeCurveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_four_cell_budget_and_no_test_seed_search(self):
        p = build_plan("fourcurves-campaign-20261018T150000Z", "abc", {})
        cells = p["stages"]["probe"]
        self.assertEqual(len(cells), 4)
        self.assertEqual(p["total_primary_interactions"], 3072000)
        for c in cells:
            s = c["settings"]
            self.assertEqual(s["seed"], 87021)
            self.assertEqual(s["updates"]*s["rollout"]//s["episode_steps"], 128)
            self.assertEqual(s["updates"]*s["rollout"] % s["episode_steps"], 0)
        with self.assertRaises(ValueError):
            settings("MARATD3-Adapted", 1)

    def test_smoothing_is_trailing_and_does_not_change_data(self):
        values = list(range(1, 102))
        before = values.copy()
        smooth = trailing_mean(values)
        self.assertEqual(values, before)
        self.assertEqual(smooth[0], 1.)
        self.assertEqual(smooth[49], 25.5)
        self.assertEqual(smooth[50], 26.5)
        self.assertEqual(trailing_mean(values[:60]), smooth[:60])
        with self.assertRaises(ValueError):
            trailing_mean([float("nan")])

    def test_same_actual_actor_initialization_no_prior_and_no_ties(self):
        actors = None
        for i, method in enumerate(("TMPPO", "IPPO")):
            for lr in (1, 2):
                torch.manual_seed(300+i+lr)
                cfg, train = pc.configuration(settings(method, lr))
                policy = pc.learner_for(method, train, cfg)
                state = copy.deepcopy(policy.actors.state_dict())
                if actors is None:
                    actors = state
                else:
                    for key, value in state.items():
                        torch.testing.assert_close(value, actors[key], atol=0, rtol=0)
                obs = np.random.default_rng(91).normal(size=(10, OBS_DIM)).astype(np.float32)
                for stage in ("user", "route"):
                    x = policy.tensor(obs)
                    torch.testing.assert_close(policy.policy_logits(stage, x), policy.actors[stage](x), atol=0, rtol=0)
                    self.assertGreater(float(policy.actors[stage](x).std()), 0.)
        base, _ = pc.configuration(dict(settings("TMPPO", 1), training_profile="credit_v3"))
        self.assertEqual(base, cfg)

    def test_checkpoint_rejects_different_initialization_protocol(self):
        cfg, train = pc.configuration(settings("TMPPO", 1))
        new = pc.learner_for("TMPPO", train, cfg)
        _, old = pc.configuration(dict(settings("TMPPO", 1), training_profile="credit_v3"))
        legacy = pc.learner_for("TMPPO", old, cfg)
        with self.assertRaises(ValueError):
            new.load(legacy.state())
        clone = pc.learner_for("TMPPO", train, cfg)
        clone.load(copy.deepcopy(new.state()))

    def test_complete_episode_reward_logs_and_pause_resume(self):
        s = dict(settings("TMPPO", 1), updates=65, rollout=8, episode_steps=260,
                 schedule_updates=65, validate_every=32, validation_steps=8, warmup_steps=4,
                 validation_seeds=[814001], validation_phases=[30.], checkpoint_every=32,
                 policy_diagnostics=False)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"cell"
            self.assertFalse(pc.train_cell(out, s, stop_after=8, wall_seconds=90))
            saved = torch.load(out/"latest.pt", map_location="cpu", weights_only=False)
            self.assertNotEqual(saved["episode_reward_sum"], 0.)
            self.assertEqual(saved["episodes"], [])
            self.assertTrue(pc.train_cell(out, s, resume=True, wall_seconds=90))
            rows = list(csv.DictReader((out/"episodes.csv").open()))
            self.assertEqual([int(r["episode"]) for r in rows], [1, 2])
            self.assertEqual([int(r["env_steps"]) for r in rows], [260, 520])
            for r in rows:
                self.assertAlmostEqual(float(r["mean_training_reward"])*260, float(r["episode_reward_sum"]))
            self.assertEqual(json.loads((out/"complete.json").read_text())["completed_episodes"], 2)
            initial = torch.load(out/"initial_checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(initial["training_steps"], 0)


if __name__ == "__main__":
    unittest.main()

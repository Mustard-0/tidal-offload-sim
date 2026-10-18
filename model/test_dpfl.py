import unittest

import numpy as np
import torch

from common import read_json
from learners import make_policy
from run import plan


class DPFLProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = read_json("config.json")
        self.config["environment"]["users_per_region"] = 20
        self.config["training"]["hidden"] = 32
        torch.manual_seed(2026)
        self.policy = make_policy("DPFL-Adapted", self.config)

    def test_plan_covers_each_registered_method_and_scale(self):
        jobs = plan(self.config, "main", 2026)
        self.assertEqual(len(jobs), 8 * 5)
        self.assertFalse(any(j["users_per_region"] == 0 for j in jobs))
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))

    def test_same_mask_and_checkpoint_round_trip(self):
        context = np.zeros((63, 64), np.float32)
        glob = np.zeros(64, np.float32)
        self.policy.begin(context, glob, 60)
        obs = np.zeros((2, 64), np.float32)
        mask = np.array([[False, True, False], [False, False, True]])
        self.policy.deterministic = True
        actions, records = self.policy.act(obs, mask, "user", np.array([0, 21]))
        self.assertEqual(actions.tolist(), [1, 2])
        self.assertEqual(records, [])
        self.policy.deterministic = False
        self.policy.begin(context, glob, 60)
        _, records = self.policy.act(obs, np.array([[False,True,True]]*2), "user", np.array([0,21]))
        self.assertEqual([r["region"] for r in records], [0, 1])
        self.policy.remember(-0.5, records, False)
        stats = self.policy.update(context, glob)
        self.assertTrue(np.isfinite(stats["value_loss"]))
        state = self.policy.state(for_evaluation=True)
        restored = make_policy("DPFL-Adapted", self.config)
        restored.load(state)
        self.assertIsNone(restored.history)


if __name__ == "__main__":
    unittest.main()

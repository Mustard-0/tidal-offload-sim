import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

import paper_convergence as pc
from four_episode_curves import settings
from cross_slot_wlo_train import CrossSlotRule
from physics import candidate_nodes, candidate_matrix


class LongTrainingIOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_append_preserves_prefix_and_rejects_schema_change(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"raw.csv"
            pc.csv_append(path, [dict(update=1, reward=.3)])
            prefix = path.read_bytes()
            pc.csv_append(path, [dict(update=2, reward=.4)])
            self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertEqual(len(pc.read_csv(path)), 2)
            with self.assertRaises(ValueError):
                pc.csv_append(path, [dict(wrong=3)])

    def test_exact_nondivisible_budget_and_same_resume_parameters(self):
        s = dict(settings("TMPPO", 1), training_profile="paper2500_v8",
                 updates=3, total_env_steps=520, rollout=256, episode_steps=260,
                 schedule_updates=3, validate_every=3, validation_steps=4, warmup_steps=2,
                 validation_seeds=[814001], validation_phases=[30.], checkpoint_every=3,
                 policy_diagnostics=False, append_training_csv=True, status_every=25)
        with tempfile.TemporaryDirectory() as name, patch.object(pc, "free_ram_gib", return_value=8.):
            root = Path(name)
            direct, resumed = root/"direct", root/"resumed"
            self.assertTrue(pc.train_cell(direct, s, wall_seconds=90))
            self.assertFalse(pc.train_cell(resumed, s, stop_after=1, wall_seconds=90))
            prefix = (resumed/"train.csv").read_bytes()
            self.assertTrue(pc.train_cell(resumed, s, resume=True, wall_seconds=90))
            self.assertTrue((resumed/"train.csv").read_bytes().startswith(prefix))
            complete = json.loads((resumed/"complete.json").read_text())
            self.assertEqual((complete["env_steps"], complete["completed_episodes"]), (520, 2))
            rows = pc.read_csv(resumed/"train.csv")
            self.assertEqual([int(r["env_steps"]) for r in rows], [256, 512, 520])
            left = torch.load(direct/"latest.pt", weights_only=False)
            right = torch.load(resumed/"latest.pt", weights_only=False)
            for group in ("actors", "critic"):
                for key in left["learner"][group]:
                    torch.testing.assert_close(left["learner"][group][key], right["learner"][group][key], rtol=0, atol=0)
            self.assertEqual(left["episodes"], right["episodes"])

    def test_overrides_cannot_change_tasks_or_projection(self):
        s = dict(settings("TMPPO", 1), environment_overrides={"project_public_region": True})
        with self.assertRaises(ValueError):
            pc.configuration(s)
        s["environment_overrides"] = {"arrival_per_user_s": .001}
        with self.assertRaises(ValueError):
            pc.configuration(s)

    def test_notidal_is_same_centralized_learner_with_only_masking(self):
        s = dict(settings("TMPPO", 1), training_profile="paper2500_v8")
        cfg, train = pc.configuration(s)
        masked, mt = pc.configuration(dict(s, method="NoTidal"))
        self.assertEqual(train, mt)
        difference = {k for k in cfg if cfg[k] != masked[k]}
        self.assertEqual(difference, {"hide_tidal_observations"})
        self.assertEqual(pc.learner_for("NoTidal", mt, masked).method, "TMAPO")

    def test_physics_cache_preserves_full_raw_trajectory(self):
        cfg, _ = pc.configuration(settings("TMPPO", 1))
        for phase in (30., 230.):
            left, right = [], []
            for enabled, target in ((False, left), (True, right)):
                c = dict(cfg, cache_route_physics=enabled)
                policy = CrossSlotRule("immediate_remote", c)
                metrics, env = pc.evaluate(policy, c, 814001, phase, 220, 30, collect=True)
                target.extend((metrics, env.rows, env.batch_rows))
            self.assertEqual(left, right)

    def test_vectorized_coverage_is_bit_identical(self):
        cfg, _ = pc.configuration(settings("TMPPO", 1))
        for t in np.random.default_rng(88).uniform(-2000, 2000, 100):
            expected = np.stack([candidate_nodes(cfg, g, t) for g in range(3)])
            np.testing.assert_array_equal(candidate_matrix(cfg, t), expected)


if __name__ == "__main__":
    unittest.main()

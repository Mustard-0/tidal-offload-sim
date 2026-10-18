import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import hybrid_campaign as h
import paper2500_campaign as c
import paper2500_coordinator as coordinator
import paper2500_evaluation as evaluation
import paper_convergence as pc
import privacy_quality_simple as quality
from cross_slot_dqn import CrossSlotDQN


class QueueContractTests(unittest.TestCase):
    def plan(self):
        return dict(plan_hash="test", stamp="20261019T010000Z", commit="a"*40,
                    runtime=dict(maximum_segments_per_task=48))

    def test_exact_budget_and_fixed_single_seed(self):
        tasks = c.convergence_tasks()+c.auxiliary_tasks({"TMPPO": 1})+c.sensitivity_tasks(1)
        self.assertEqual(len(tasks), 24)
        self.assertEqual(sum(t["settings"]["total_env_steps"] for t in tasks), c.MAX_INTERACTIONS)
        self.assertEqual({t["settings"]["seed"] for t in tasks}, {87021})
        for task in tasks:
            s = task["settings"]
            self.assertEqual((s["updates"], s["updates"]*s["rollout"]-s["total_env_steps"]), (117188, 64))
            request = coordinator.request(self.plan(), [dict(task=task, parent=None, segment=1)], 1, False)
            c.validate_request(self.plan(), request)

    def test_mutated_budget_seed_or_fourth_extra_cycle_rejected(self):
        for change in (dict(total_env_steps=15000001), dict(seed=42), dict(lr=3)):
            task = copy.deepcopy(c.convergence_tasks()[0])
            task["settings"].update(change)
            request = coordinator.request(self.plan(), [dict(task=task, parent=None, segment=1)], 1, False)
            with self.assertRaises(ValueError):
                c.validate_request(self.plan(), request)
        request = coordinator.request(self.plan(), [dict(task=c.convergence_tasks()[0], parent=None, segment=49)], 1, False)
        with self.assertRaises(ValueError):
            c.validate_request(self.plan(), request)

    def test_scaling_and_sensitivity_only_no_extra_privacy_grid(self):
        tasks = evaluation.evaluation_tasks("scaling", [])
        self.assertEqual(len(tasks), 15)
        for t in tasks:
            evaluation.validate_evaluation_task(t)
        with self.assertRaises(ValueError):
            evaluation.evaluation_tasks("privacy", [])
        for task in c.sensitivity_tasks(1):
            weights = task["settings"]["environment_overrides"]
            self.assertAlmostEqual(sum(weights[k] for k in ("weight_time", "weight_energy", "weight_quality")), .8)
            self.assertEqual(weights["weight_violation"], 1.)

    def test_cloud_publish_is_idempotent_without_duplicate_tag_creation(self):
        p = self.plan()
        req = coordinator.request(p, [dict(task=c.convergence_tasks()[0], parent=None, segment=1)], 1, True)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c.immutable(root/"requests"/req["filename"], req)
            def api(*args):
                if str(args[-1]).endswith("per_page=100"):
                    return [dict(tag_name=req["release"])]
                return dict(assets=[dict(name="wave.json", digest="sha256:"+h.file_hash(root/"requests"/req["filename"]))])
            calls = []
            def command(args, **kwargs):
                calls.append(args)
                return p["commit"] if args[1] == "rev-parse" else ("existing" if args[1] == "tag" else "")
            with patch.object(c, "public_only"), patch.object(c, "gh_json", side_effect=api), \
                    patch.object(c, "gh") as gh, patch.object(h, "command", side_effect=command):
                first = coordinator.publish_request(root, p, req)
                self.assertEqual(first, coordinator.publish_request(root, p, req))
                gh.assert_not_called()
            self.assertFalse(any(args[1] == "tag" and "--list" not in args for args in calls))

    def test_wrong_checkpoint_lineage_rejected_before_copy(self):
        p = self.plan()
        task = c.convergence_tasks()[0]
        with tempfile.TemporaryDirectory() as d, patch.object(c, "receive_asset", return_value=(Path(d),
                dict(plan_hash="wrong", task_id=task["id"], commit=p["commit"], outcome="paused"), {})):
            with self.assertRaises(ValueError):
                c.restore(Path(d), p, task, dict(name="x"))


class DQNContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg, self.training = pc.configuration(c.settings("MATORA-DQN-Adapted", 1, 87021))
        self.training.update(actor_warmup_steps=0, hidden=16)

    def make(self):
        return CrossSlotDQN("MATORA-DQN-Adapted", self.training, self.cfg)

    def test_real_q_update_target_and_replay_resume(self):
        np.random.seed(7)
        learner = self.make()
        obs = np.ones((1, 64), dtype=np.float32)*.05
        mask = np.asarray([[True, True, True]])
        for _ in range(140):
            learner.act(obs, mask, "user", [0])
            learner.remember(np.full(60, -.02), [], False)
        learner.remember(np.full(60, -.03), [], True)
        self.assertFalse(learner.pending)
        initial = copy.deepcopy(learner.q.state_dict())
        saved = copy.deepcopy(learner.state())
        rng = np.random.get_state()
        result = learner.update()
        clone = self.make()
        clone.load(saved)
        np.random.set_state(rng)
        self.assertEqual(result, clone.update())
        self.assertGreater(result["dqn_updates"], 0)
        self.assertTrue(any(not torch.equal(v, learner.q.state_dict()[k]) for k, v in initial.items()))
        for k, v in learner.q.state_dict().items():
            torch.testing.assert_close(v, clone.q.state_dict()[k], atol=0, rtol=0)
        for k, v in learner.target.state_dict().items():
            torch.testing.assert_close(v, initial[k], atol=0, rtol=0)

    def test_empty_masks_and_wrong_checkpoint_rejected(self):
        learner = self.make()
        with self.assertRaises(ValueError):
            learner.act(np.zeros((1, 64)), np.zeros((1, 3), dtype=bool), "user", [0])
        with self.assertRaises(ValueError):
            learner.load(dict(schema="renamed-ppo"))

    def test_deterministic_evaluation_does_not_store_training_events(self):
        learner = self.make()
        learner.deterministic = True
        mask = np.asarray([[False, False, True]])
        action, records = learner.act(np.zeros((1, 64)), mask, "user", [0])
        self.assertEqual(action.tolist(), [2])
        self.assertFalse(learner.pending)
        self.assertFalse(learner.replay)


class SimpleC1Tests(unittest.TestCase):
    def test_actual_distinct_batch_paired_noise_and_raw_no_projection(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"C1"
            result = quality.run(out, repetitions=2)
            self.assertFalse(result["dp_proven"])
            self.assertEqual(result["samples"], 3000)
            groups = {}
            for row in pc.read_csv(out/"reports_raw.csv"):
                groups.setdefault(row["batch_id"], []).append(row)
                ratio = float(row["epsilon"])/float(row["epsilon0_shuffle"])
                self.assertAlmostEqual(float(row["ql_shuffle_m"])/float(row["ql_geoi_m"]), ratio, places=10)
            for rows in groups.values():
                self.assertEqual(len({r["user"] for r in rows}), 100)
                self.assertEqual({int(r["shuffled_output_index"]) for r in rows}, set(range(100)))
            self.assertEqual(result, quality.run(out, repetitions=2))
            with self.assertRaises(FileExistsError):
                quality.run(out, repetitions=3)


if __name__ == "__main__":
    unittest.main()

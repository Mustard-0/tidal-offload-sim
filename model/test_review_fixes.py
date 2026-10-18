import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common import ROOT, read_json, save_json, source_hashes, seed_all
from environment import Environment, Task, OBS_DIM
from evaluation import evaluation_phase
from learners import CONTROL_DIM, STAGE_SLICES, client_rounds, make_policy
from run import apply_selection, plan, selected_learning_rates
from workload import analytical_load_rows, expected_batch_fill_s, tide_factors


class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT/"config.json")
        self.config["training"].update(hidden=16, epochs=1, minibatch=32)
        self.c = self.config["environment"]
        seed_all(14, 1)

    def stage(self, env, count, age=0):
        for user in range(count):
            j = Task(user, user, 0, env.time-age, self.c["task_bits"], self.c["task_cycles"], self.c["deadline_s"], status="staged")
            env.tasks[j.id] = j
            env.staging[0].append(j.id)
            env.staged_users.add(user)

    def test_release_is_always_available_below_threshold(self):
        env = Environment(self.c, 12)
        self.stage(env, 20)
        env.windows_seen = 1
        env.last_observed[:] = 1000
        _, mask = env.batch_features(0, np.zeros(OBS_DIM))
        self.assertEqual(mask.tolist(), [True, True])

    def test_insufficient_arrival_rate_forces_release(self):
        env = Environment(self.c, 12)
        self.stage(env, 1)
        env.windows_seen = 1
        env.last_observed[:] = 1
        _, mask = env.batch_features(0, np.zeros(OBS_DIM))
        self.assertEqual(mask.tolist(), [False, True])

    def test_service_time_is_reserved_before_waiting(self):
        env = Environment(self.c, 12)
        self.stage(env, 20, age=self.c["deadline_s"]-.015)
        env.windows_seen = 1
        env.last_observed[:] = 10000
        _, mask = env.batch_features(0, np.zeros(OBS_DIM))
        self.assertEqual(mask.tolist(), [False, True])

    def test_no_prediction_leak_through_batch_gate(self):
        env = Environment(self.c, 12)
        self.stage(env, 20)
        env.windows_seen = 1
        env.last_observed[:] = 1000
        env.previous_forecast[:] = 100000
        _, a = env.batch_features(0, np.zeros(OBS_DIM))
        env.method = "TMAPO-NoPred"
        env.previous_forecast[:] = 0
        _, b = env.batch_features(0, np.zeros(OBS_DIM))
        np.testing.assert_array_equal(a, b)

    def test_wait_fill_uses_distinct_users(self):
        self.assertAlmostEqual(expected_batch_fill_s(30, 29, 0, 30, 3), 1/3)
        self.assertEqual(expected_batch_fill_s(30, 29, 1, 30, 3), 0)
        self.assertTrue(np.isinf(expected_batch_fill_s(20, 1, 0, 30, 3)))

    def test_tide_has_global_and_regional_variation(self):
        samples = np.stack([tide_factors(self.c, t) for t in np.linspace(0, 300, 1000, endpoint=False)])
        self.assertGreater(samples.min(), 0)
        self.assertGreater(samples.sum(axis=1).max()-samples.sum(axis=1).min(), 2)
        np.testing.assert_allclose(samples.mean(axis=0), np.ones(3), atol=1e-12)
        rows = analytical_load_rows(self.c, [20, 100])
        self.assertLess(rows[0]["all_remote_offered_peak"], .25)
        self.assertGreater(rows[1]["all_remote_offered_peak"], 1)
        self.assertLess(rows[1]["ue_offered_mean"], 1)

    def test_existing_users_keep_positions_and_arrivals_across_scales(self):
        a = Environment(dict(self.c, users_per_region=2), 101)
        b = Environment(dict(self.c, users_per_region=3), 101)
        indexes = [0, 1, 3, 4, 6, 7]
        np.testing.assert_array_equal(a.xy, b.xy[indexes])
        for slot in range(20):
            for env in (a, b):
                env.slot, env.time = slot, slot*self.c["dt_s"]
                env.generate()
            self.assertEqual([len(x) for x in a.pending], [len(b.pending[i]) for i in indexes])

    def test_report_processing_exact_slot_does_not_add_an_extra_slot(self):
        c = dict(self.c, privacy_processing_s=.01)
        e = Environment(c, 1)
        j = Task(0, 0, 0, 0, c["task_bits"], c["task_cycles"], c["deadline_s"])
        e.release([j])
        self.assertAlmostEqual(j.report_ready, .01)

    def test_evaluation_phases_span_a_cycle_and_do_not_depend_on_seed_subset(self):
        t = self.config["training"]
        centers = [(evaluation_phase(self.config, s, 10, 10)+15)%300 for s in t["test_seeds"]]
        np.testing.assert_allclose(centers, [30, 90, 150, 210, 270])

    def test_td3_stage_outputs_do_not_alias(self):
        p = make_policy("MARATD3-Adapted", self.config)
        p.action = np.zeros((1, CONTROL_DIM), np.float32)
        p.action[0, STAGE_SLICES["route"]][4] = .5
        obs = np.zeros((1, OBS_DIM), np.float32)
        mask = np.ones((1, 11), bool)
        a = p.act(obs, mask, "route", np.array([0]))[0]
        p.action[0, STAGE_SLICES["user"]] = [1, -1, 1]
        p.action[0, STAGE_SLICES["batch"]] = [-1, 1]
        b = p.act(obs, mask, "route", np.array([0]))[0]
        self.assertEqual(int(a[0]), 4)
        np.testing.assert_array_equal(a, b)

    def test_td3_uses_same_zero_residual_cost_guidance(self):
        p = make_policy("MARATD3-Adapted", self.config)
        p.action = np.zeros((1, CONTROL_DIM), np.float32)
        obs = np.zeros((1, OBS_DIM), np.float32)
        obs[0, 17:19] = [.1, .8]
        a = p.act(obs, np.array([[False, True, True]]), "user", np.array([0]))[0]
        self.assertEqual(int(a[0]), 1)

    def test_dpfl_history_contains_previous_context_once(self):
        p = make_policy("DPFL-Adapted", self.config)
        p.deterministic = True
        one = np.ones((303, OBS_DIM), np.float32)
        p.begin(one, np.zeros(OBS_DIM), 300)
        self.assertTrue((p.history == 0).all())
        p.begin(one*2, np.zeros(OBS_DIM), 300)
        self.assertTrue((p.history[:, -1] == 1).all())
        self.assertTrue((p.history[:, :-1] == 0).all())

    def test_federated_minibatches_cover_each_sample_once(self):
        seen = {0: [], 1: [], 2: []}
        data = {0: list(range(11)), 1: list(range(7)), 2: []}
        for batch in client_rounds(data, 4):
            for region, rows in batch.items():
                self.assertLessEqual(len(rows), 4)
                seen[region].extend(rows)
        for g in data:
            self.assertEqual(sorted(seen[g]), data[g])

    def test_noamp_inherits_selected_tmapo_lr(self):
        jobs = plan(self.config, "privacy", 2026)
        apply_selection(jobs, {"TMAPO": {"lr": 1}, "MARATD3-Adapted": {"lr": 0}})
        self.assertTrue(all(j["lr"] == 1 and "LR2" in j["id"] for j in jobs if j["method"].startswith("TMAPO")))
        self.assertEqual(len(plan(self.config, "baseline-tuning", 2026)), 8)

    def test_lr_selection_allows_scale_transfer_but_rejects_changed_workload(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test_selection_") as directory:
            path = Path(directory)
            save_json(path/"config.json", self.config)
            save_json(path/"source_hashes.json", source_hashes())
            save_json(path/"selection.json", {"TMAPO": {"lr": 0}})
            other = copy.deepcopy(self.config)
            other["environment"]["users_per_region"] = 20
            self.assertIn("TMAPO", selected_learning_rates(other, [path/"selection.json"]))
            other["environment"]["arrival_per_user_s"] = 9
            with self.assertRaises(ValueError):
                selected_learning_rates(other, [path/"selection.json"])


if __name__ == "__main__":
    unittest.main()

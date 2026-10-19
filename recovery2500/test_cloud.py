import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
TEST_TMP = (HERE.parent / ".test_tmp").resolve()
assert HERE.parent.resolve() in TEST_TMP.parents
TEST_TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TEST_TMP)
spec = importlib.util.spec_from_file_location("recovery_cloud", HERE / "cloud.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
r.bind(Path(os.environ.get("FROZEN_SOURCE", str(HERE.parent / "frozen"))))


def context():
    plan = dict(commit=r.MODEL_COMMIT, plan_hash=r.PLAN_HASH,
                runtime=dict(maximum_segments_per_task=48, local_workers=4, github_workers=16))
    policy = dict(original_plan_hash=r.PLAN_HASH, maximum_total_segments_per_task=96,
                  prior_segments=dict(zip([t["id"] for t in r.c.convergence_tasks()], [48, 1, 1, 1])),
                  free_runner="ubuntu-22.04", maximum_wall_days=7, maximum_training_cells=24,
                  maximum_primary_training_interactions=360000000, runtime_commit="b" * 40,
                  original_started_epoch=1792349805.672066)
    policy["policy_hash"] = r.h.digest(policy)
    task = r.c.convergence_tasks()[0]
    parent = dict(release="test", name="parent.zip", digest="sha256:" + "a" * 64)
    wave = dict(plan_hash=r.PLAN_HASH, cloud=True, entries=[dict(
        task=task, segment=49, previous_segment=48, parent=parent)],
        recovery_policy_hash=policy["policy_hash"], runtime_commit=policy["runtime_commit"],
        stage_tasks=r.c.convergence_tasks())
    return plan, policy, wave


class RecoveryContractTests(unittest.TestCase):
    def test_approved_overlay_leaves_scientific_plan_unchanged(self):
        p, policy, _ = context()
        original = copy.deepcopy(p)
        actual = r.execution_plan(p, policy)
        self.assertEqual(p, original)
        self.assertEqual(actual["runtime"]["maximum_segments_per_task"], 96)
        self.assertEqual(actual["runtime"]["local_workers"], 0)
        self.assertEqual(actual["commit"], p["commit"])

    def test_historical_48_is_not_reset(self):
        p, policy, wave = context()
        r.validate_wave(p, policy, wave)
        wave["entries"][0].update(segment=1, previous_segment=0, parent=None)
        with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_skipped_segment_is_rejected(self):
        p, policy, wave = context()
        wave["entries"][0]["segment"] = 50
        with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_96_is_finite_and_97_rejected(self):
        p, policy, wave = context()
        wave["entries"][0].update(segment=96, previous_segment=95)
        r.validate_wave(p, policy, wave)
        wave["entries"][0].update(segment=97, previous_segment=96)
        with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_resumed_task_requires_parent(self):
        p, policy, wave = context()
        wave["entries"][0]["parent"] = None
        with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_no_seed_learning_rate_or_budget_change(self):
        for change in (dict(seed=42), dict(lr=3), dict(total_env_steps=15000001)):
            p, policy, wave = context()
            wave["entries"][0]["task"]["settings"].update(change)
            with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_wrong_plan_or_runtime_commit_rejected(self):
        for key in ("recovery_policy_hash", "runtime_commit", "plan_hash"):
            p, policy, wave = context()
            wave[key] = "wrong"
            with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_duplicate_stage_registry_rejected(self):
        p, policy, wave = context()
        wave["stage_tasks"].append(wave["stage_tasks"][0])
        with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_task_outside_registry_rejected(self):
        p, policy, wave = context()
        wave["stage_tasks"] = wave["stage_tasks"][1:]
        with self.assertRaises(ValueError): r.validate_wave(p, policy, wave)

    def test_policy_tampering_and_unlimited_recovery_rejected(self):
        for key, value in (("maximum_total_segments_per_task", 100000),
                           ("maximum_training_cells", 25),
                           ("free_runner", "large-gpu"),
                           ("maximum_wall_days", 8)):
            p, policy, _ = context()
            policy[key] = value
            policy["policy_hash"] = r.h.digest({k: v for k, v in policy.items() if k != "policy_hash"})
            with self.assertRaises(ValueError): r.execution_plan(p, policy)

    def test_new_task_starts_at_one_only(self):
        p, policy, wave = context()
        task = r.c.auxiliary_tasks({"TMPPO": 1})[0]
        wave["stage_tasks"] = [task]
        wave["entries"] = [dict(task=task, parent=None, segment=1, previous_segment=0)]
        r.validate_wave(p, policy, wave)

    def test_unknown_plan_commit_rejected(self):
        p, policy, _ = context()
        p["commit"] = "x" * 40
        with self.assertRaises(ValueError): r.execution_plan(p, policy)

    def provenance(self):
        _, policy, wave = context()
        e = wave["entries"][0]
        origin = dict(plan_hash=r.PLAN_HASH, commit=r.MODEL_COMMIT, task_id=e["task"]["id"],
                      parent=e["parent"], run_id="123", attempt="1", outcome="paused")
        execution = dict(policy_hash=policy["policy_hash"], runtime_commit=policy["runtime_commit"],
                         segment=49, task_id=e["task"]["id"], run_id="123", attempt="1",
                         parent=e["parent"], outcome="paused")
        return policy, e, origin, execution

    def test_origin_matches_frozen_model_and_actual_workflow(self):
        policy, e, origin, execution = self.provenance()
        r.verify_cloud_origin(origin, execution, e["task"], e["parent"], 49, 123, policy)

    def test_wrong_attempt_workflow_or_parent_cannot_resume(self):
        for key, value in (("attempt", "2"), ("runtime_commit", "bad"), ("parent", None), ("segment", 48)):
            policy, e, origin, execution = self.provenance()
            execution[key] = value
            with self.assertRaises(ValueError):
                r.verify_cloud_origin(origin, execution, e["task"], e["parent"], 49, 123, policy)

    def test_model_commit_cannot_be_relabelled(self):
        policy, e, origin, execution = self.provenance()
        origin["commit"] = policy["runtime_commit"]
        with self.assertRaises(ValueError):
            r.verify_cloud_origin(origin, execution, e["task"], e["parent"], 49, 123, policy)

    def test_immutable_asset_reused_without_upload(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "result.zip"
            path.write_bytes(b"evidence")
            asset = dict(name=path.name, digest="sha256:" + r.h.file_hash(path), size=8)
            with patch.object(r.c, "gh_json", return_value=dict(assets=[asset])), patch.object(r.c, "gh") as upload:
                self.assertEqual(r.publish_file("release", path)["digest"], asset["digest"])
                upload.assert_not_called()

    def test_conflicting_asset_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "result.zip"
            path.write_bytes(b"evidence")
            with patch.object(r.c, "gh_json", return_value=dict(assets=[dict(name=path.name, digest="bad", size=8)])), patch.object(r.c, "gh") as upload:
                with self.assertRaises(ValueError): r.publish_file("release", path)
                upload.assert_not_called()

    def test_no_stage_transition_without_verified_gate(self):
        p, _, _ = context()
        with self.assertRaises(ValueError):
            r.next_stage(Path("unused"), p, dict(stage="convergence", tasks=[]))

    def test_learning_failure_stops_without_tuning(self):
        p, _, _ = context()
        state = dict(stage="convergence", tasks=[], last_verified_gate=dict(stage="convergence"))
        with patch.object(r.h, "read_json", return_value=dict(passed=False)):
            r.next_stage(Path("unused"), p, state)
        self.assertEqual(state["status"], "needs_review")
        self.assertNotIn("choices", state)

    def test_missing_cloud_zip_stops_without_duplicate_training(self):
        p, policy, wave = context()
        task = wave["entries"][0]["task"]
        state = dict(status="running", stage="convergence", tasks=[task],
                     task_status={task["id"]: dict(status="running_cloud", segments=48)})
        wave.update(release="test")
        launched = dict(run_id=123, request=wave)
        responses = [dict(headSha=policy["runtime_commit"], attempt=1, status="completed", conclusion="failure"),
                     dict(assets=[])]
        with patch.object(r, "POLICY", policy), patch.object(r.c, "gh_json", side_effect=responses), \
                patch.object(r.h, "write_json"):
            self.assertTrue(r.receive_wave(Path("unused"), p, launched, state))
        self.assertEqual(state["status"], "needs_review")
        self.assertEqual(state["task_status"][task["id"]]["segments"], 48)

    def test_partial_wave_does_not_run_expensive_gate(self):
        _, policy, wave = context()
        wave.update(completed={}, release="test")
        with tempfile.TemporaryDirectory() as d, patch.object(r.c, "gh_json", return_value=dict(assets=[])):
            self.assertFalse(r.ready_to_finalize(Path(d), policy, wave))

    def test_total_registered_training_budget_still_exact(self):
        tasks = r.c.convergence_tasks() + r.c.auxiliary_tasks({"TMPPO": 1}) + r.c.sensitivity_tasks(1)
        self.assertEqual(len(tasks), 24)
        self.assertEqual(sum(t["settings"]["total_env_steps"] for t in tasks), 360000000)
        self.assertEqual({t["settings"]["seed"] for t in tasks}, {87021})

    def test_legacy_entrypoint_is_unchanged_not_reimplemented(self):
        text = (HERE / "cloud.py").read_text()
        self.assertIn("return c.run_cell(root, p, task, entry[\"parent\"])", text)
        self.assertNotIn("optimizer.step(", text)
        self.assertNotIn("train_cell(", text)

    def test_workflow_only_standard_public_cpu_and_exact_model(self):
        text = (HERE.parent / ".github/workflows/paper2500_recovery.yml").read_text()
        self.assertEqual(text.count("runs-on: ubuntu-22.04"), 3)
        self.assertEqual(text.count("ref: " + r.MODEL_COMMIT), 3)
        self.assertNotIn("upload-artifact", text)
        self.assertNotIn("actions/cache", text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn("tags: ['recovery2500-wave-*']", text)
        self.assertIn("!github.event.repository.private", text)


if __name__ == "__main__":
    unittest.main()

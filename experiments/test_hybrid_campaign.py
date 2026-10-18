"""Tests prepared for the next model. Not executed in the authoring turn."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_campaign as hybrid


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def fixture(root):
    content = b"# immutable fixture source\n"
    sources = {"model/fixture.py": hashlib.sha256(content).hexdigest()}
    plan = hybrid.build_plan("paper-campaign-20261018T000000Z", "fixture-commit", sources)
    cfg = dict(weight_time=.25, weight_energy=.25, weight_quality=.25, weight_violation=1.,
               time_ref_s=1., energy_ref_j=1., quality_ref_m=10., dt_s=.01)
    for cell in plan["stages"]["screen"]:
        s = cell["settings"]
        s.update(updates=2, rollout=8, validate_every=1, validation_seeds=[814001],
                 validation_phases=[30.])
        directory = root/"cells/screen"/cell["id"]
        directory.mkdir(parents=True)
        source = directory/"source/model/fixture.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(content)
        runtime = {k: plan["runtime"][k] for k in ("torch", "numpy", "scipy", "matplotlib")}
        runtime.update(python_version=plan["runtime"]["python"], device="cpu", torch_threads=1)
        actor, critic = ((1e-4, 3e-4), (3e-4, 1e-3))[s["lr"]-1]
        hybrid.write_json(directory/"manifest.json", dict(settings=s, environment=cfg,
            training=dict(actor_lr=actor, critic_lr=critic, local_credit_fraction=0.,
                          td3_discrete_objective="categorical_expectation"),
            runtime=runtime, source_commit=plan["commit"], source_hashes=sources))
        hybrid.write_json(directory/"complete.json", dict(updates=2, env_steps=16))
        hybrid.write_json(directory/"selection.json", dict(update=0, validation_cost=.5, test_used=False))
        for checkpoint in ("latest.pt", "best_checkpoint.pt"):
            with zipfile.ZipFile(directory/checkpoint, "w") as archive:
                archive.writestr("archive/data.pkl", b"fixture-only; never unpickle")
        write_csv(directory/"train.csv", [dict(update=i, env_steps=i*8) for i in (1, 2)])
        validation = []
        for update in (0, 1, 2):
            tag = dict(update=update, env_steps=update*8, method=s["method"], seed=814001, phase_s=30.)
            validation.append(dict(tag, arrivals=1, completed=1, failed=0,
                mean_task_cost=.5, qos_rate=0., energy_j_per_arrival=1., publication_ratio=0.,
                delay_s=1., ql_m=""))
            task = dict(tag, task_id=1, user=0, region=0, arrival_s=0., task_profile="interactive",
                task_bits=50000., task_cycles=80e6, deadline_s=2., arrival_tide_factor=1.,
                end_s=1., completed=1, qos=0, energy_j=1., transmit_energy_j=0.,
                compute_energy_j=1., report_energy_j=0., published=0, ql_m="")
            write_csv(directory/"validation_traces"/("tasks_%06d.csv" % update), [task])
        write_csv(directory/"validation.csv", validation)
    return plan


class HybridCampaignTests(unittest.TestCase):
    def test_owners_and_budgets_are_disjoint_and_complete(self):
        plan = hybrid.build_plan("paper-campaign-20261018T000000Z", "sha", {})
        self.assertEqual(len(plan["stages"]["screen"]), 6)
        self.assertEqual(len(plan["stages"]["formal"]), 18)
        for stage in ("screen", "formal"):
            cells = plan["stages"][stage]
            local = {c["id"] for c in cells if c["owner"] == "local"}
            cloud = {c["id"] for c in cells if c["owner"] == "github"}
            self.assertFalse(local & cloud)
            self.assertEqual(len(local | cloud), len(cells))
            for cell in cells:
                self.assertEqual(cell["settings"]["schedule_updates"], 2000)
        self.assertEqual(sum(c["settings"]["updates"]*c["settings"]["rollout"]
                             for c in plan["stages"]["formal"]), 4608000)

    def test_worker_command_passes_exact_settings_and_explicit_long_authorization(self):
        plan = hybrid.build_plan("paper-campaign-20261018T000000Z", "sha", {})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"absent"
            cmd = hybrid.worker_command(plan["stages"]["formal"][0], out, plan, "formal")
            self.assertIn("--allow-long-cell", cmd)
            self.assertIn("--phase-count", cmd)
            self.assertEqual(cmd[cmd.index("--training-profile")+1], "team_v2")
            self.assertNotIn("--validation-seeds", cmd)
            out.mkdir()
            with self.assertRaises(ValueError):
                hybrid.worker_command(plan["stages"]["formal"][0], out, plan, "formal")

    def test_nonimproving_but_stable_cells_can_pass_without_a_manufactured_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verdict = hybrid.gate(fixture(root), root)
            self.assertTrue(verdict["passed"], verdict["errors"])
            self.assertTrue(all(not c["improves_initialization"] for c in verdict["cells"]))

    def test_corrupt_task_energy_blocks_formal_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root)
            path = next((root/"cells/screen").glob("*/validation_traces/tasks_000001.csv"))
            raw = list(hybrid.rows(path))
            raw[0]["energy_j"] = "2"
            write_csv(path, raw)
            verdict = hybrid.gate(plan, root)
            self.assertFalse(verdict["passed"])
            self.assertTrue(any("Raw metric mismatch" in e for e in verdict["errors"]))

    def test_changed_workload_and_wrong_snapshot_block_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root)
            cell = root/"cells/screen"/plan["stages"]["screen"][0]["id"]
            path = cell/"validation_traces/tasks_000001.csv"
            raw = list(hybrid.rows(path))
            raw[0]["task_bits"] = "99999"
            write_csv(path, raw)
            self.assertFalse(hybrid.gate(plan, root)["passed"])
            (cell/"source/model/fixture.py").write_text("# changed\n")
            self.assertTrue(any("snapshot" in e for e in hybrid.gate(plan, root)["errors"]))

    def test_missing_checkpoint_and_unregistered_extra_cell_block_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root)
            cell = root/"cells/screen"/plan["stages"]["screen"][0]["id"]
            (cell/"best_checkpoint.pt").unlink()
            (root/"cells/screen/unregistered").mkdir()
            verdict = hybrid.gate(plan, root)
            self.assertFalse(verdict["passed"])
            self.assertTrue(any("Unregistered" in e for e in verdict["errors"]))

    def test_bad_checkpoint_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root)
            cell = root/"cells/screen"/plan["stages"]["screen"][0]["id"]
            hybrid.write_json(cell/"selection.json", dict(update=2, validation_cost=.5, test_used=False))
            self.assertFalse(hybrid.gate(plan, root)["passed"])

    def test_safe_bundle_roundtrip_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root)
            hybrid.write_json(root/"campaign.json", plan)
            archive = hybrid.bundle_screen(plan, root)
            hybrid.extract_evidence(archive, root/"extracted")
            self.assertEqual(hybrid.gate(plan, root/"extracted"), hybrid.read_json(root/"screen_gate.json"))
            bad = root/"bad.zip"
            with zipfile.ZipFile(bad, "w") as zipped:
                zipped.writestr("../escaped.txt", "outside")
            with self.assertRaises(ValueError):
                hybrid.extract_evidence(bad, root/"bad_extract")
            self.assertFalse((root/"escaped.txt").exists())

    def test_lf_source_hashes_are_portable_but_binary_hashes_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/"lf", Path(tmp)/"crlf"
            first.write_bytes(b"a\nb\n")
            second.write_bytes(b"a\r\nb\r\n")
            self.assertEqual(hybrid.file_hash(first, True), hybrid.file_hash(second, True))
            self.assertNotEqual(hybrid.file_hash(first), hybrid.file_hash(second))

    def test_private_repo_is_rejected_before_release_or_tag_push(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(hybrid, "gh_json", return_value=dict(isPrivate=True)), \
                 patch.object(hybrid, "command") as command:
                with self.assertRaises(ValueError):
                    hybrid.ensure_release({}, Path(tmp), "owner/private")
                command.assert_not_called()

    def test_cloud_outputs_use_release_archives_with_exact_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root)
            cell = next(c for c in plan["stages"]["screen"] if c["owner"] == "github")
            with patch.object(hybrid, "command") as command:
                hybrid.publish_cloud_output(plan, root, "owner/public", "screen", cell["id"], "123", "1")
                call = command.call_args[0][0]
                self.assertEqual(call[:3], ["gh", "release", "upload"])
                self.assertNotIn("--clobber", call)
            archive = next((root/"uploads").glob("*.zip"))
            with zipfile.ZipFile(archive) as zipped:
                origin = json.loads(zipped.read("execution_origin.json"))
                self.assertEqual(origin["run_id"], "123")
                self.assertIn("cells/screen/"+cell["id"]+"/train.csv", zipped.namelist())


if __name__ == "__main__":
    unittest.main()

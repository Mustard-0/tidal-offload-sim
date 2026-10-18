"""No long training: strict budget, opt-in launch and transient-write handling."""
import copy
import os
import re
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import atomic_io
import long_probe
import hybrid_campaign as hybrid


class LongProbeTests(unittest.TestCase):
    def test_fixed_one_seed_six_cells_and_2000_updates(self):
        p=long_probe.build_plan("longprobe-campaign-20261018T000000Z","sha",{})
        cells=p["stages"]["probe"]
        self.assertEqual(len(cells),6)
        self.assertEqual({c["settings"]["seed"] for c in cells},{87021})
        self.assertEqual({c["settings"]["updates"] for c in cells},{2000})
        self.assertEqual(sum(c["settings"]["updates"]*c["settings"]["rollout"] for c in cells),1536000)
        self.assertFalse(p["screen_passed"])
        self.assertFalse(p["formal_matrix"])
        self.assertFalse(p["publication_grade"])
        self.assertEqual(p["runtime"]["github_workers"],6)
        for c in cells:
            self.assertTrue(c["settings"]["diagnostics_endpoints_only"])
            self.assertEqual(c["settings"]["validate_every"],100)
            self.assertEqual(c["settings"]["episode_steps"],6000)

    def test_long_worker_args_have_explicit_long_cell_budget(self):
        p=long_probe.build_plan("longprobe-campaign-20261018T000000Z","sha",{})
        with tempfile.TemporaryDirectory() as tmp:
            cmd=hybrid.worker_command(p["stages"]["probe"][0],Path(tmp)/"absent",p,"formal")
        self.assertIn("--allow-long-cell",cmd)
        self.assertIn("--diagnostics-endpoints-only",cmd)
        self.assertEqual(cmd[cmd.index("--updates")+1],"2000")
        self.assertEqual(cmd[cmd.index("--schedule-updates")+1],"2000")

    def test_rehashed_budget_tamper_is_rejected(self):
        p=long_probe.build_plan("longprobe-campaign-20261018T000000Z","sha",{})
        p["verification"]=dict(passed=True,source_hashes={})
        p["stages"]["probe"][0]["settings"]["updates"]=2001
        p["plan_hash"]=hybrid.digest({k:v for k,v in p.items() if k!="plan_hash"})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"plan.json"
            hybrid.write_json(path,p)
            with self.assertRaises(ValueError):
                long_probe.load_plan(path)

    def test_worker_requires_explicit_execute(self):
        argv=["long_probe.py","worker","--out","unused","--cell","TMPPO_LR1_seed87021"]
        with patch.object(sys,"argv",argv),patch.object(long_probe,"load_plan",return_value={}), \
                patch.object(long_probe,"worker") as work:
            with self.assertRaises(SystemExit) as exc:
                long_probe.main()
            self.assertEqual(exc.exception.code,2)
            work.assert_not_called()

    def test_atomic_transient_windows_failure_retries_without_rewriting_data(self):
        error=PermissionError("sharing violation")
        error.winerror=5
        with patch.object(Path,"replace",side_effect=[error,Path("target")]) as replace, \
                patch.object(atomic_io.time,"sleep") as sleep:
            atomic_io.replace_file("source","target")
        self.assertEqual(replace.call_count,2)
        sleep.assert_called_once_with(.05)

    def test_atomic_retry_exhaustion_and_nontransient_errors_are_not_hidden(self):
        error=PermissionError("sharing violation")
        error.winerror=32
        with patch.object(Path,"replace",side_effect=error) as replace,patch.object(atomic_io.time,"sleep"):
            with self.assertRaises(PermissionError):
                atomic_io.replace_file("source","target",attempts=2)
            self.assertEqual(replace.call_count,2)
        with patch.object(Path,"replace",side_effect=PermissionError("not transient")), \
                patch.object(atomic_io.time,"sleep") as sleep:
            with self.assertRaises(PermissionError):
                atomic_io.replace_file("source","target")
            sleep.assert_not_called()

    def test_live_progress_is_labelled_and_uploaded_without_clobber(self):
        p=long_probe.build_plan("longprobe-campaign-20261018T000000Z","sha",{})
        match=re.match(r"VALIDATE .+ update (\d+) cost ([\d.eE+-]+) best (\d+)",
                       "VALIDATE TMPPO LR 1 update 100 cost 0.4 best 0")
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,dict(
                GITHUB_RUN_ID="123",GITHUB_RUN_ATTEMPT="1",GITHUB_REPOSITORY="owner/repo")), \
                patch.object(hybrid,"command") as command:
            root=Path(tmp)
            long_probe.publish_progress(p,root,"TMPPO_LR1_seed87021",match)
            path=next(root.rglob("probe_progress*.json"))
            value=hybrid.read_json(path)
            self.assertEqual(value["validation_completed_update"],100)
            self.assertEqual(value["best_update"],0)
            self.assertNotIn("--clobber",command.call_args[0][0])

    def test_endpoint_sampling_flag_does_not_add_midpoint_diagnostics(self):
        from paper_convergence import configuration,learner_for,validation_snapshot
        from test_paper_convergence import settings
        s=dict(settings(),training_profile="credit_v3",policy_diagnostics=True,
               diagnostics_endpoints_only=True)
        cfg,train=configuration(s)
        cfg["users_per_region"]=2
        learner=learner_for("TMPPO",train,cfg)
        with tempfile.TemporaryDirectory() as tmp:
            middle=validation_snapshot("TMPPO",learner,train,cfg,s,1,Path(tmp))
            first=validation_snapshot("TMPPO",learner,train,cfg,s,0,Path(tmp))
        self.assertEqual({r["method"] for r in middle},{"TMPPO"})
        self.assertEqual({r["method"] for r in first},{"TMPPO","TMPPO:sampled-diagnostic"})

    def test_single_algorithm_partial_report_does_not_select_an_lr(self):
        from paper_convergence import train_cell,report
        from test_paper_convergence import settings
        import json
        s=dict(settings("IPPO"),updates=1,validate_every=1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertTrue(train_cell(root/"IPPO_LR1_seed87021",s,wall_seconds=120))
            report(root)
            selection=json.loads((root/"learning_rate_selection.json").read_text())
            self.assertFalse(selection["final_selection_allowed"])
            self.assertIsNone(selection["selected_lr"])
            self.assertTrue((root/"convergence_pilot.png").exists())


if __name__=="__main__":
    unittest.main()

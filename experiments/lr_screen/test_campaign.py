"""Mocked queue lifecycle; never starts a training subprocess."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import campaign
from .runtime import ROOT, json_write
from .protocol import plan


class CampaignChecks(unittest.TestCase):
    def run_fixture(self, kind):
        base = ROOT/'evidence/component_checks'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root = Path(tmp)
            p = plan()
            p['cells'] = p['cells'][:2]
            calls = []
            def fake_child(cmd, **kwargs):
                i = len(calls)
                calls.append(cmd)
                cell = p['cells'][i]
                if kind != 'failure':
                    json_write(root/cell['id']/'complete.json', dict(env_steps=cell['env_steps'], component_only=False))
                return type('FakeChild', (), dict(pid=123456, wait=lambda self: 1 if kind=='failure' else 0))()
            if kind == 'stop':
                (root/'STOP_AFTER_UPDATE').touch()
            with patch.object(campaign, 'plan', return_value=p), \
                 patch.object(campaign, 'require_transfer', return_value={'passed':True}), \
                 patch.object(campaign, 'available_ram_gib', return_value=3.), \
                 patch.object(campaign.time, 'sleep'), \
                 patch.object(campaign.subprocess, 'Popen', side_effect=fake_child), \
                 patch('experiments.lr_screen.report.report'):
                if kind == 'failure':
                    with self.assertRaisesRegex(RuntimeError, 'Cell failed'):
                        campaign.run(root)
                else:
                    campaign.run(root)
            import json
            state = json.loads((root/'campaign_status.json').read_text())
            self.assertFalse((root/'campaign.lock').exists())
            self.assertEqual(len(calls), {'success':2,'failure':1,'stop':0}[kind])
            self.assertEqual(state['state'], {'success':'completed','failure':'failed','stop':'paused_explicit_stop'}[kind])

    def test_finite_sequential_completion(self):
        self.run_fixture('success')

    def test_failure_is_not_retried_or_expanded(self):
        self.run_fixture('failure')

    def test_explicit_stop_prevents_dispatch(self):
        self.run_fixture('stop')


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Execution-only tests: no Torch, no network, no formal interactions."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ops.cloud_owner import ROOT,collect
from ops.cloud_pack import groups
from ops.packed_owner import ReadRetryGitHub,build_request,worker_groups,adoption
from experiments.highload.protocol import source_hashes,digest,plan


class PackingContracts(unittest.TestCase):
    def test_unchanged_science(self):
        c=json.loads((ROOT/'cloud/contract.json').read_text())
        self.assertEqual(c['source_hashes'],source_hashes());self.assertEqual(c['plan_hash'],digest(plan()))

    def test_read_retries_disconnect_but_not_write(self):
        a=ReadRetryGitHub('unit-test-only')
        with patch('ops.github_api.GitHub.request',side_effect=[http.client.RemoteDisconnected(),{'ok':True}]),patch('time.sleep'):
            self.assertEqual(a.request('GET','/read'),{'ok':True})
        with patch('ops.github_api.GitHub.request',side_effect=http.client.RemoteDisconnected()) as f:
            with self.assertRaises(http.client.RemoteDisconnected):a.request('POST','/write',{})
            self.assertEqual(f.call_count,1)

    def test_pack_only_ready_distinct_cells(self):
        s=copy.deepcopy(json.loads((ROOT/'runs/highload_cloud_01/coordinator.json').read_text()))
        ready=[k for k,c in s['cells'].items() if c['state']=='ready']
        req=build_request(s,ready[:11])
        self.assertTrue(all(len(g)<=4 for g in groups(req['entries'])))
        self.assertEqual(len(groups(req['entries'])),3)
        self.assertEqual(len(req['entries']),11)
        with self.assertRaises(ValueError):build_request(s,[ready[0],ready[0]])
        old=next(k for k,c in s['cells'].items() if c['state']=='active')
        with self.assertRaises(ValueError):build_request(s,[old])

    def test_concurrent_groups_not_cells(self):
        s=dict(waves=[dict(finalized=False,run_id=None,received=[],entries=[dict(cell=str(i),job_name='pack-00') for i in range(4)])])
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as d:self.assertEqual(worker_groups(s,Path(d)),1)

    def test_mixed_commit_collect_preserves_old_ownership(self):
        entries=lambda name,job=None:[dict(cell=name,kind='train',method='TMAPO',lr=1,seed=87021,segment=1,**(dict(job_name=job) if job else {}))]
        s=dict(public_commit='old',campaign='c',plan_hash='p',release_id=1,cells={},waves=[
            dict(finalized=False,tag='old',run_id='1',received=[],entries=entries('a')),
            dict(finalized=False,tag='new',run_id='2',public_commit='new',received=[],entries=entries('b','pack-00'))])
        class Fake:
            def request(self,method,path):
                if '/assets?' in path:return []
                if '/jobs?' in path:return dict(jobs=[dict(name='a' if '/1/' in path else 'pack-00',status='in_progress',conclusion=None)])
                return dict(head_sha='old' if path.endswith('/1') else 'new',run_attempt=1,status='in_progress',conclusion=None)
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as d:collect(Fake(),s,Path(d),b'x'*32)
        self.assertEqual([w['status'] for w in s['waves']],['in_progress','in_progress'])

    def test_adoption_requires_real_matching_probe(self):
        export=dict(commit='probe-sha')
        with self.assertRaises(ValueError):adoption(ROOT/'runs/highload_cloud_01',export,dict(passed=False,equivalence_passed=False,source_commit='probe-sha'))


if __name__=='__main__':unittest.main()

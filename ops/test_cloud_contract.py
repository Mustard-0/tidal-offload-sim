"""No-network orchestration/transport tests, no model import or training."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from ops.cloud_owner import ROOT,HANDOFF,prefix_audit,ready_order,specifications,verify_asset
from ops.transport import pack,unpack,seal,unseal,sha
from experiments.highload.protocol import source_hashes,plan,digest
from experiments.highload.cloud_worker import request_from_tag


class Contracts(unittest.TestCase):
    def test_scientific_freeze(self):
        old=json.loads((ROOT/'evidence/latest_preflight.json').read_text())
        self.assertEqual(old['source_hashes'],source_hashes())
        self.assertEqual(old['plan_hash'],digest(plan()))
        self.assertEqual(len(specifications()),31)

    def test_saved_prefix(self):
        cp=prefix_audit(ROOT/'runs/highload_local_01/cells'/HANDOFF,specifications()[HANDOFF])
        self.assertEqual(cp['env_steps'],19700)

    def test_ciphertext_authenticated(self):
        key=b'x'*32;header=dict(cell=HANDOFF,segment=2)
        packet=seal(b'private results',key,header)
        self.assertEqual(unseal(packet,key,header)[0],b'private results')
        with self.assertRaises(ValueError):unseal(packet,key,dict(segment=3))
        with self.assertRaises(Exception):unseal(packet[:-1]+bytes([packet[-1]^1]),key,header)

    def test_transfer_clean_and_nonoverwriting(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as td:
            folder=Path(td)/'input';folder.mkdir()
            for n in ('STOP_AFTER_UPDATE','STOP','.worker.lock','owner.lock','latest.pt','failure.json'):(folder/n).write_text(n)
            raw=pack(folder)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:self.assertEqual(set(z.namelist()),{'latest.pt','failure.json'})
            dest=Path(td)/'output';unpack(raw,dest)
            with self.assertRaises(FileExistsError):unpack(raw,dest)

    def test_zip_traversal(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('../escape','no')
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as td:
            with self.assertRaises(ValueError):unpack(b.getvalue(),Path(td)/'out')

    def test_fair_first_wave(self):
        specs=specifications();state=dict(specs=specs,cells={k:dict(state='ready') for k in specs})
        ordered=ready_order(state)
        self.assertEqual(ordered[0],HANDOFF)
        self.assertEqual(len(ordered),len(set(ordered)))
        self.assertEqual({specs[k]['method'] for k in ordered[:16]},set(plan_method for plan_method in ('TMAPO','IPPO','PPO','SD-SAC-Adapted','MARATD3-Adapted')))
        state['cells'][HANDOFF]['state']='active'
        self.assertNotIn(HANDOFF,ready_order(state))

    def test_dispatch_guards(self):
        entry=dict(cell=HANDOFF,method='TMAPO',lr=3,seed=87021,kind='train',segment=2,input=dict(name='parent'))
        req=dict(plan_hash=digest(plan()),entries=[entry])
        env=dict(GITHUB_REF_NAME='highload-run-test',GITHUB_RUN_ATTEMPT='1')
        with patch.dict(os.environ,env),patch('subprocess.check_output',side_effect=lambda *a,**k:json.dumps(req)):
            self.assertEqual(request_from_tag()['entries'][0],entry)
            req['entries'].append(entry)
            with self.assertRaises(ValueError):request_from_tag()
        with patch.dict(os.environ,dict(env,GITHUB_RUN_ATTEMPT='2')):
            with self.assertRaises(ValueError):request_from_tag()

    def test_published_digest(self):
        verify_asset(dict(size=3,digest='sha256:'+sha(b'abc')),b'abc')
        with self.assertRaises(ValueError):verify_asset(dict(size=3,digest=None),b'abc')


if __name__=='__main__':unittest.main()

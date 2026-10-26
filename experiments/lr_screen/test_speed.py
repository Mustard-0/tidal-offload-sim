"""Local exact-equivalence and saved-prefix migration checks, not new results."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from .runtime import ROOT, torch
from .portable import adopt

PARENT_ROOT = Path('F:/stin_project/tmapo_lr_screen_20261025')


class SpeedChecks(unittest.TestCase):
    def test_hoisted_observations_and_updates_are_bitwise_equal_to_parent(self):
        code = """
import hashlib,json
from experiments.lr_screen.runtime import np,torch,seed_all
from experiments.lr_screen.protocol import configuration
from experiments.lr_screen.algorithms import learner_for
from experiments.lr_screen.restricted_environment import CandidateRestrictedEnvironment
c,t=configuration('TMAPO',2);seed_all(87021)
p=learner_for('TMAPO',t,c);e=CandidateRestrictedEnvironment(c,seed=20261026,phase_s=30.)
h=hashlib.sha256()
for update in range(2):
 for _ in range(64):
  reward,records,*_=e.step(p);p.remember(e.agent_rewards(0.),records,False)
  context,glob,mask,*_=e.observations()
  for a in [context,glob,mask,np.array([reward,e.cost_total])]:h.update(a.tobytes())
  for r in records:h.update(str(r['action']).encode())
 p.update(context,glob)
for m in [p.actors,p.critic]:
 for a in m.state_dict().values():h.update(a.cpu().numpy().tobytes())
print(json.dumps({'sha256':h.hexdigest(),'trace':e.arrival_hash,'cost':e.cost_total}))
"""
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(PARENT_ROOT/'.runtime'))
        a = subprocess.check_output([sys.executable,'-c',code],cwd=PARENT_ROOT,env=env,text=True,timeout=120)
        b = subprocess.check_output([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,timeout=120)
        self.assertEqual(json.loads(a),json.loads(b))

    def test_real_paused_checkpoint_adoption_preserves_model_optimizer_rng(self):
        source = PARENT_ROOT/'runs/lr_screen_20261026_01/TMAPO_LR1_seed87021'
        base = ROOT/'evidence/component_checks'
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as d:
            folder = Path(d)/source.name
            shutil.copytree(source,folder)
            before = torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
            adopt(folder,'TMAPO',1)
            after = torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
            self.assertEqual(before['update'],28)
            self.assertEqual(before['rows'],after['rows'])
            self.assertEqual(before['valrows'],after['valrows'])
            self.assertEqual(before['environment'].arrival_hash,after['environment'].arrival_hash)
            self.assertEqual(before['environment'].cost_total,after['environment'].cost_total)
            for module in ('actors','critic'):
                for k,v in before['learner'][module].items():
                    torch.testing.assert_close(v,after['learner'][module][k],rtol=0,atol=0)
            for optimizer in ('aopt','copt'):
                for k,v in before['learner'][optimizer]['state'].items():
                    for field,value in v.items():
                        actual = after['learner'][optimizer]['state'][k][field]
                        if torch.is_tensor(value):
                            torch.testing.assert_close(value,actual,rtol=0,atol=0)
                        else:
                            self.assertEqual(value,actual)
            self.assertEqual(before['rng']['python'],after['rng']['python'])
            torch.testing.assert_close(before['rng']['torch'],after['rng']['torch'],rtol=0,atol=0)
            self.assertTrue((folder/'origins').exists())
            self.assertNotEqual(before['identity_hash'],after['identity_hash'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

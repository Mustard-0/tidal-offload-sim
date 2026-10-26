"""Only the new counting/checkpoint path; original mechanisms remain unchanged."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from .protocol import configuration, plan, ROOT
from .runner import train
from experiments.lr_screen.runtime import torch

COMPONENT = dict(episodes=2, episode_steps=256, checkpoint_episodes=1)


class EpisodeTests(unittest.TestCase):
    def test_budget(self):
        p = plan()
        self.assertEqual(len(p['cells']), 3)
        for c in p['cells']:
            _, t = configuration(c['algorithm'])
            self.assertEqual(t['episodes']*t['episode_steps'],12000000)
            self.assertEqual(t['updates']*t['rollout_steps'],12000000)
            self.assertNotIn('validation_interval',t)

    def test_updates_episode_count_and_exact_resume(self):
        scratch = ROOT/'.test_tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root = Path(tmp)
            reference = None
            for method in ('TMAPO','IPPO','PPO'):
                out = root/method
                self.assertTrue(train(out, method, component=COMPONENT))
                saved = torch.load(out/'latest.pt',weights_only=False)
                self.assertEqual(saved['completed_episodes'],2)
                self.assertEqual(saved['env_steps'],512)
                self.assertEqual([r['env_steps'] for r in saved['rows']],[256,512])
                self.assertEqual([r['episode'] for r in saved['rows']],[1,2])
                if method == 'TMAPO':
                    reference = saved
            split = root/'split'
            self.assertFalse(train(split,'TMAPO',component=COMPONENT,stop_update=3))
            prefix = torch.load(split/'latest.pt',weights_only=False)
            self.assertEqual(prefix['env_steps'],384)
            self.assertEqual(prefix['completed_episodes'],1)
            self.assertTrue(train(split,'TMAPO',resume=True,component=COMPONENT))
            actual = torch.load(split/'latest.pt',weights_only=False)
            self.assertEqual(reference['rows'],actual['rows'])
            self.assertEqual(pickle.dumps(reference['environment']),pickle.dumps(actual['environment']))
            self.assertTrue(torch.equal(reference['rng']['torch'],actual['rng']['torch']))
            for module in ('actors','critic'):
                for name,value in reference['learner'][module].items():
                    self.assertTrue(torch.equal(value,actual['learner'][module][name]))
            initial_cfg, initial_t = configuration('TMAPO')
            from experiments.lr_screen.runtime import seed_all
            from experiments.lr_screen.algorithms import learner_for
            seed_all(initial_t['seed'])
            initial = learner_for('TMAPO',initial_t,initial_cfg,'cpu').state()
            self.assertTrue(any(not torch.equal(v,reference['learner']['actors'][k])
                                for k,v in initial['actors'].items()))


if __name__ == '__main__':
    unittest.main()

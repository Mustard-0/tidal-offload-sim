"""Only the new counting/checkpoint path; original mechanisms remain unchanged."""
import json
from pathlib import Path
from collections import deque
import tempfile
import unittest
from .protocol import configuration, plan, ROOT
from .runner import train
from experiments.lr_screen.runtime import np, torch

COMPONENT = dict(episodes=2, episode_steps=256, checkpoint_episodes=1)


class EpisodeTests(unittest.TestCase):
    def assert_state_equal(self, left, right, path='state'):
        self.assertEqual(type(left),type(right),path)
        if isinstance(left,torch.Tensor):
            self.assertTrue(torch.equal(left,right),path)
        elif isinstance(left,np.ndarray):
            self.assertEqual(left.dtype,right.dtype,path)
            np.testing.assert_array_equal(left,right,err_msg=path)
        elif isinstance(left,np.random.Generator):
            self.assert_state_equal(left.bit_generator.state,right.bit_generator.state,path+'.generator')
        elif isinstance(left,np.random.RandomState):
            self.assert_state_equal(left.get_state(),right.get_state(),path+'.rng')
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right),path)
            for key in left:
                self.assert_state_equal(left[key],right[key],path+'.'+str(key))
        elif isinstance(left,(tuple,list,deque)):
            self.assertEqual(len(left),len(right),path)
            for i,(a,b) in enumerate(zip(left,right)):
                self.assert_state_equal(a,b,path+'.'+str(i))
        elif hasattr(left,'__dict__'):
            self.assert_state_equal(vars(left),vars(right),path+'.vars')
        else:
            self.assertEqual(left,right,path)

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
            # Pickle's reference memo/alias encoding is not state semantics.
            # Compare every environment field and full learner/optimizer/RNG
            # value instead; this also diagnoses the exact differing field.
            self.assert_state_equal(reference['learner'],actual['learner'],'learner')
            self.assert_state_equal(reference['environment'],actual['environment'],'environment')
            self.assert_state_equal(reference['rng'],actual['rng'],'rng')
            initial_cfg, initial_t = configuration('TMAPO')
            from experiments.lr_screen.runtime import seed_all
            from experiments.lr_screen.algorithms import learner_for
            seed_all(initial_t['seed'])
            initial = learner_for('TMAPO',initial_t,initial_cfg,'cpu').state()
            self.assertTrue(any(not torch.equal(v,reference['learner']['actors'][k])
                                for k,v in initial['actors'].items()))


if __name__ == '__main__':
    unittest.main()

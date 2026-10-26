"""Only the new counting/checkpoint path; original mechanisms remain unchanged."""
import json
from pathlib import Path
from collections import deque
import tempfile
import unittest
from .protocol import configuration, plan, ROOT, SEEDS
from .runner import train
from experiments.lr_screen.runtime import np, torch

COMPONENT = dict(episodes=2, checkpoint_episodes=1)


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
        self.assertEqual(len(p['cells']), 23)
        self.assertEqual(len({c['id'] for c in p['cells']}),23)
        self.assertEqual({m:sum(c['algorithm']==m for c in p['cells']) for m in SEEDS},
                         {'TMAPO':15,'IPPO':4,'PPO':4})
        for c in p['cells']:
            _, t = configuration(c['algorithm'],c['lr_index'],c['seed'])
            self.assertEqual(t['episodes'],2000)
            self.assertEqual(t['arrival_steps'],100)
            self.assertEqual(t['seed'],c['seed'])
            self.assertEqual(t['actor_lr'],c['actor_lr'])
            self.assertEqual(t['critic_lr'],c['critic_lr'])
            self.assertEqual(p['population'],[34,33,33])
            self.assertNotIn('validation_interval',t)

    def test_baseline_formal_prefix_is_unchanged(self):
        expected={'TMAPO':(-24.231917200949628,-30.218177066922983),
                  'IPPO':(-25.451871794091865,-30.556954513329558),
                  'PPO':(-26.770828368360984,-28.796063202142946)}
        scratch=ROOT/'.test_tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            for method,values in expected.items():
                out=Path(tmp)/method
                self.assertFalse(train(out,method,component={'checkpoint_episodes':5},stop_episode=2))
                saved=torch.load(out/'latest.pt',weights_only=False)
                for row,value in zip(saved['rows'],values):
                    self.assertAlmostEqual(row['episode_return'],value,places=10)

    def test_new_lr_seed_paths_and_resume(self):
        scratch=ROOT/'.test_tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            traces=[]
            for method,lr,seed in [('TMAPO',5,87023),('IPPO',2,87022),('PPO',2,87022)]:
                out=Path(tmp)/method
                self.assertTrue(train(out,method,component=COMPONENT,lr=lr,seed=seed))
                manifest=json.loads((out/'manifest.json').read_text())
                _,t=configuration(method,lr,seed)
                self.assertEqual(manifest['lr'],lr)
                self.assertEqual(manifest['training']['seed'],seed)
                self.assertEqual(manifest['training']['actor_lr'],t['actor_lr'])
                saved=torch.load(out/'latest.pt',weights_only=False)
                traces.append([r['train_trace_hash'] for r in saved['rows']])
                if method=='TMAPO':
                    reference=saved
            self.assertNotEqual(traces[0],traces[1])
            self.assertEqual(traces[1],traces[2])
            split=Path(tmp)/'split'
            self.assertFalse(train(split,'TMAPO',component=COMPONENT,stop_episode=1,lr=5,seed=87023))
            self.assertTrue(train(split,'TMAPO',resume=True,component=COMPONENT,lr=5,seed=87023))
            actual=torch.load(split/'latest.pt',weights_only=False)
            self.assert_state_equal(reference['learner'],actual['learner'],'new_lr_seed_learner')
            self.assert_state_equal(reference['rng'],actual['rng'],'new_lr_seed_rng')
            self.assert_state_equal(reference['environment'],actual['environment'],'new_lr_seed_environment')

    def test_real_nonuniform_membership_and_batch(self):
        from types import SimpleNamespace
        from population import region_counts,user_region
        from experiments.lr_screen.mechanism import actual_batch
        from .runner import new_environment
        cfg,t=configuration('TMAPO')
        self.assertEqual(region_counts(cfg),(34,33,33))
        self.assertEqual([user_region(cfg,u) for u in (0,33,34,66,67,99)],[0,0,1,1,2,2])
        env=new_environment(cfg,t['seed'],0,100)
        self.assertEqual(env.u,100)
        np.testing.assert_array_equal(np.bincount(env.region),[34,33,33])
        self.assertEqual(env.xy.shape,(100,2))
        tasks=[SimpleNamespace(id=i,user=u,region=1,profile='balanced') for i,u in enumerate((34,35))]
        n,eps,bound=actual_batch(tasks,cfg)
        self.assertEqual(n,2)
        self.assertLessEqual(bound,1e-6+1e-12)
        tasks[0].user=0
        with self.assertRaises(ValueError): actual_batch(tasks,cfg)
        with self.assertRaises(ValueError): user_region(cfg,100)

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
                self.assertEqual([r['episode'] for r in saved['rows']],[1,2])
                self.assertEqual(saved['env_steps'],sum(r['actual_steps'] for r in saved['rows']))
                self.assertTrue(all(r['arrival_steps']==100 and r['actual_steps']<=303
                    and r['arrivals']==r['ended'] for r in saved['rows']))
                self.assertTrue(any(r['drain_steps']>0 for r in saved['rows']))
                self.assertFalse(saved['environment'].tasks)
                if method == 'TMAPO':
                    reference = saved
                else:
                    self.assertEqual([r['train_trace_hash'] for r in reference['rows']],
                                     [r['train_trace_hash'] for r in saved['rows']])
                from experiments.lr_screen.runtime import seed_all
                from experiments.lr_screen.algorithms import learner_for
                cfg,t=configuration(method)
                seed_all(t['seed'])
                initial=learner_for(method,t,cfg,'cpu').state()
                self.assertTrue(any(not torch.equal(v,saved['learner']['actors'][k])
                                    for k,v in initial['actors'].items()))
            split = root/'split'
            self.assertFalse(train(split,'TMAPO',component=COMPONENT,stop_episode=1))
            prefix = torch.load(split/'latest.pt',weights_only=False)
            self.assertEqual(prefix['completed_episodes'],1)
            self.assertTrue(train(split,'TMAPO',resume=True,component=COMPONENT))
            actual = torch.load(split/'latest.pt',weights_only=False)
            self.assertEqual([{k:v for k,v in r.items() if k!='training_wall_s'} for r in reference['rows']],
                             [{k:v for k,v in r.items() if k!='training_wall_s'} for r in actual['rows']])
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

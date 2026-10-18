import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common import ROOT, read_json, seed_all
from environment import Environment, GreedyPolicy, OBS_DIM, Task
from learners import ReturnNormalizer, make_policy
from run import training_phase, train_job


class LongTrainingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read_json(ROOT/"config.json")
        self.cfg["training"].update(hidden=16, epochs=1, minibatch=32, torch_threads=1)
        seed_all(42, 1)

    def test_task_mix_preserves_means(self):
        c=self.cfg["environment"]
        for key, common in (("bits","task_bits"),("cycles","task_cycles"),("deadline_s","deadline_s")):
            self.assertAlmostEqual(sum(p["probability"]*p[key] for p in c["task_profiles"]),c[common])

    def test_task_profiles_are_paired_across_scales(self):
        c=self.cfg["environment"]
        a,b=Environment(dict(c,users_per_region=2),17),Environment(dict(c,users_per_region=4),17)
        for slot in range(200):
            for env in (a,b):
                env.slot,env.time=slot,slot*c["dt_s"]
                env.generate()
        for user in range(a.u):
            mapped=(user//2)*4+user%2
            pa=[(a.tasks[x].arrival_slot,a.tasks[x].profile) for x in a.pending[user]]
            pb=[(b.tasks[x].arrival_slot,b.tasks[x].profile) for x in b.pending[mapped]]
            self.assertEqual(pa,pb)

    def test_route_actor_receives_forecast_and_ablation_masks_it(self):
        c=self.cfg["environment"]; env=Environment(c,17)
        j=Task(0,0,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
        env.previous_forecast[:]=[100,200,300]
        a=env.route_features(j,c["region_centers_m"][0])[0]
        self.assertGreater(a[57],0)
        self.assertTrue((a[59:62]>0).all())
        env.method="TMAPO-NoPred"
        b=env.route_features(j,c["region_centers_m"][0])[0]
        self.assertTrue((b[57:62]==0).all())
        np.testing.assert_array_equal(a[:57],b[:57])

    def test_public_remote_estimate_uses_actual_task_size(self):
        c=self.cfg["environment"]; env=Environment(c,17)
        for user, bits in ((0,200000),(1,600000)):
            j=Task(user,user,0,0,bits,80000000,.2)
            env.tasks[user]=j; env.pending[user].append(user)
        obs=env.observations()[0]
        self.assertLess(obs[0,18],obs[1,18])

    def test_credit_conserves_cost_for_users_and_regions(self):
        c=self.cfg["environment"]; env=Environment(dict(c,users_per_region=2),17)
        greedy=GreedyPolicy(env.cfg)
        for _ in range(150):
            team=env.step(greedy)[0]
            rewards=env.agent_rewards(.7)
            self.assertAlmostEqual(float(env.user_step_cost.sum()),env.step_cost,places=7)
            self.assertAlmostEqual(float(rewards[:env.u].mean()),team,places=5)
            self.assertAlmostEqual(float(rewards[env.u:].mean()),team,places=5)

    def test_initial_deterministic_policy_matches_greedy(self):
        c=dict(self.cfg["environment"],users_per_region=3)
        cfg=copy.deepcopy(self.cfg); cfg["environment"]=c
        for method in ("TMAPO","IPPO","DPFL-Adapted","MARATD3-Adapted"):
            a,b=Environment(c,27,method),Environment(c,27,"Greedy")
            policy=make_policy(method,cfg); policy.deterministic=True
            greedy=GreedyPolicy(c)
            for _ in range(150):
                self.assertAlmostEqual(a.step(policy)[0],b.step(greedy)[0],places=7)
            self.assertEqual(a.arrival_hash,b.arrival_hash)

    def test_normalizer_matches_pooled_moments_and_checkpoint(self):
        norm=ReturnNormalizer()
        x=np.arange(100,dtype=float)-50
        norm.update(x[:37]); norm.update(x[37:])
        self.assertAlmostEqual(norm.mean,float(x.mean()))
        self.assertAlmostEqual(norm.std,float(x.std()))
        np.testing.assert_allclose(norm.denormalize(norm.normalize(x)),x)
        norm2=ReturnNormalizer(); norm2.load(norm.state())
        self.assertEqual(norm.state(),norm2.state())
        p=make_policy("TMAPO",self.cfg); p.value_norm.load(norm.state())
        q=make_policy("TMAPO",self.cfg); q.load(copy.deepcopy(p.state()))
        inputs=np.zeros((5,OBS_DIM*2),np.float32)
        np.testing.assert_array_equal(p.raw_values(inputs),q.raw_values(inputs))

    def test_strata_cover_a_cycle_in_each_block(self):
        for block in (0,5,10):
            values=[training_phase(self.cfg,42,ep) for ep in range(block,block+5)]
            self.assertEqual(sorted(values),[30.,90.,150.,210.,270.])

    def test_schedule_has_fixed_terminal_budget(self):
        p=make_policy("TMAPO",self.cfg)
        p.set_progress(1,2000); self.assertEqual(p.aopt.param_groups[0]["lr"],self.cfg["training"]["actor_lr"])
        p.set_progress(2000,2000)
        self.assertAlmostEqual(p.aopt.param_groups[0]["lr"],self.cfg["training"]["actor_lr"]*.1)

    def test_pause_resume_matches_uninterrupted_training(self):
        cfg=copy.deepcopy(self.cfg)
        cfg["environment"]["users_per_region"]=2
        cfg["training"].update(updates=2,rollout_steps=8,episode_steps=16,validation_interval=2,
                               pilot_warmup_s=0.,pilot_measure_s=1.,validation_seeds=[501],test_seeds=[601])
        job=dict(id="test",method="TMAPO",users_per_region=2,epsilon_req=1.,lr=0,seed=42,kind="convergence")
        with tempfile.TemporaryDirectory(dir=ROOT,prefix="test_resume_") as tmp:
            root=Path(tmp); a,b=root/"full",root/"resumed"
            # Plotting is a suite-level operation; this test targets exact checkpoints.
            import unittest.mock
            with unittest.mock.patch("plots.plot_run"):
                self.assertTrue(train_job(cfg,job,a,"cpu"))
                self.assertFalse(train_job(cfg,job,b,"cpu",wall_seconds=1e-8))
                self.assertTrue(train_job(cfg,job,b,"cpu"))
            x=torch.load(a/"latest.pt",weights_only=False)
            y=torch.load(b/"latest.pt",weights_only=False)
            for key in ("actors","critic"):
                for name in x["policy"][key]:
                    torch.testing.assert_close(x["policy"][key][name],y["policy"][key][name],rtol=0,atol=0)
            self.assertEqual(x["policy"]["value_norm"],y["policy"]["value_norm"])
            self.assertEqual(x["env_steps"],16)
            self.assertEqual(x["validation_rows"],y["validation_rows"])


if __name__=="__main__": unittest.main()

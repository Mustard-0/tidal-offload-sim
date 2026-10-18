import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"experiments"), str(ROOT/"model")]
from paper_four_studies import full_config, paper_plan, TEST_SEEDS, VALIDATION_SEEDS, TRAIN_SEEDS
from paper_convergence import (configuration, learner_for, train_cell, validation_snapshot,
                               rng_state, evaluate, new_environment, validation_mean)
from cross_slot_learners import cross_slot_prior
from learners import CONTROL_DIM
from cross_slot_environment import CrossSlotWLOEnvironment
from cross_slot_wlo_train import CrossSlotRule
from environment import Task, OBS_DIM


def settings(method="TMPPO", lr=1):
    return dict(method=method, lr=lr, seed=87021, updates=2, rollout=8,
                episode_steps=256, phase_count=3, validate_every=2,
                validation_steps=12, warmup_steps=0,
                validation_seeds=[814001], validation_phases=[30.])


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        np.random.seed(91)
        self.s = settings()
        self.c, self.t = configuration(self.s)
        self.c["users_per_region"] = 2

    def test_plan_is_exact_and_denominators_explicit(self):
        plan = paper_plan()
        self.assertEqual(len(plan["convergence"]), 18)
        self.assertEqual(len(plan["scaling"]), 35)
        self.assertEqual(len(plan["sensitivity"]), 15)
        self.assertTrue(set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS))
        self.assertTrue(set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS))
        self.assertNotIn("Static-K-Anon", plan["baseline_methods"])

    def test_physics_objective_and_raw_privacy_are_unchanged(self):
        original, _ = full_config()
        actual, _ = configuration(self.s)
        self.assertEqual(original, actual)
        self.assertFalse(actual["project_public_region"])
        self.assertEqual(len(actual["bs_xy_m"]), 8)
        self.assertEqual(len(actual["sat_overhead_s"]), 3)

    def test_team_v2_preserves_physics_and_aligns_all_agent_rewards(self):
        c, t = configuration(dict(self.s, training_profile="team_v2"))
        self.assertEqual(c, full_config()[0])
        self.assertEqual(t["local_credit_fraction"], 0.)
        self.assertTrue(t["linear_lr_decay"])
        self.assertTrue(t["td3_huber"])
        env = CrossSlotWLOEnvironment(self.c, seed=17)
        env.step_cost = 1.
        env.user_step_cost[:] = np.arange(1, env.u+1)
        env.user_step_cost[:] /= env.user_step_cost.sum()
        reward = env.agent_rewards(0.)
        self.assertTrue(np.all(reward == reward[0]))
        self.assertGreater(float(np.std(env.agent_rewards(.7))), 0.)

    def test_pooled_validation_respects_each_metric_denominator(self):
        cohorts = [dict(arrivals=10, completed=5, published=2, mean_task_cost=1.,
                        qos_rate=.5, delay_s=2., ql_m=20.),
                   dict(arrivals=90, completed=90, published=18, mean_task_cost=3.,
                        qos_rate=0., delay_s=4., ql_m=40.)]
        self.assertEqual(validation_mean(cohorts, "mean_task_cost"), 2.)
        self.assertAlmostEqual(validation_mean(cohorts, "mean_task_cost", "pooled_task_cost"), 2.8)
        self.assertAlmostEqual(validation_mean(cohorts, "qos_rate", "pooled_task_cost"), .05)
        self.assertAlmostEqual(validation_mean(cohorts, "delay_s", "pooled_task_cost"), 370/95)
        self.assertAlmostEqual(validation_mean(cohorts, "ql_m", "pooled_task_cost"), 38.)

    def test_td3_discrete_expectation_uses_hard_vertices_and_actor_gradients(self):
        _, t = configuration(dict(self.s, training_profile="team_v2"))
        learner = learner_for("MARATD3-Adapted", t, self.c)
        history = torch.zeros((2, t["td3_history"], OBS_DIM+CONTROL_DIM))
        glob = torch.zeros((2, OBS_DIM))
        mask = torch.zeros(11)
        mask[2:4] = 1.
        seen = []

        def known_q(q, h, g, actions, masks, stages):
            seen.append(actions.detach().clone())
            return actions[:, 7] - actions[:, 8]

        with patch.object(learner, "_critic", side_effect=known_q):
            loss = learner.categorical_actor_loss(history, glob, [mask, mask], ["route", "route"])
            loss.backward()
        self.assertTrue(all(torch.all((a == 0) | (a == 1)) for a in seen))
        self.assertTrue(all(torch.all(a.sum(-1) == 1) for a in seen))
        gradient = learner.actor.head[-2].bias.grad
        self.assertLess(float(gradient[7]), 0.)
        self.assertGreater(float(gradient[8]), 0.)
        self.assertTrue(all(p.grad is None for p in learner.q.parameters()))
        legacy = learner_for("MARATD3-Adapted", self.t, self.c)
        with self.assertRaises(ValueError):
            legacy.load(learner.state())

    def test_sampled_validation_is_labelled_and_restores_training_rng(self):
        learner = learner_for("TMPPO", self.t, self.c)
        state = rng_state()
        s = dict(self.s, policy_diagnostics=True)
        with tempfile.TemporaryDirectory() as tmp:
            rows = validation_snapshot("TMPPO", learner, self.t, self.c, s, 0, Path(tmp))
        self.assertEqual({r["method"] for r in rows}, {"TMPPO", "TMPPO:sampled-diagnostic"})
        self.assertTrue(torch.equal(state["torch"], rng_state()["torch"]))
        np.testing.assert_equal(state["numpy"][1], rng_state()["numpy"][1])
        for row in rows:
            self.assertAlmostEqual(sum(row[key] for key in
                ("objective_time", "objective_energy", "objective_quality", "objective_qos")),
                row["mean_task_cost"])

    def test_pause_saves_checkpoint_even_between_io_intervals(self):
        s = dict(self.s, checkpoint_every=25, schedule_updates=2000, training_profile="team_v2")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"sparse_io"
            self.assertFalse(train_cell(out, s, stop_after=1))
            checkpoint = torch.load(out/"latest.pt", weights_only=False)
            self.assertEqual(checkpoint["update"], 1)
            self.assertEqual(json.loads((out/"paused.json").read_text())["update"], 1)
            self.assertTrue(train_cell(out, s, resume=True))

    def test_wait_prior_pays_opportunity_cost_for_every_method(self):
        obs = torch.zeros((2, OBS_DIM))
        obs[0,17:19] = torch.tensor([.4,.6])
        obs[0,26] = .405
        obs[1,17:19] = torch.tensor([.255,.25])
        obs[1,63] = -1/3
        logits = cross_slot_prior("user", obs, self.c, 2.)
        self.assertEqual(logits.argmax(-1).tolist(), [1,2])

    def test_undiscounted_episode_does_not_reward_postponing_cost(self):
        # At 10ms, legacy gamma=.99 can prefer adding 180ms solely to discount
        # an eventual .75 cost. The paper's undiscounted objective cannot.
        old_delayed = .005*sum(.99**i for i in range(18)) + .75*.99**18
        self.assertLess(old_delayed,.75)
        self.assertEqual(self.t["gamma"],1.)
        corrected = .005*sum(self.t["gamma"]**i for i in range(18)) + .75*self.t["gamma"]**18
        self.assertGreater(corrected,.75)

    def test_group_class_keys_are_distinct_even_with_equal_probabilities(self):
        env = CrossSlotWLOEnvironment(self.c, seed=9)
        learner = learner_for("MARATD3-Adapted", self.t, self.c)
        features = []
        for i, profile in enumerate(self.c["task_profiles"]):
            j = Task(i, i, 0, 0., profile["bits"], profile["cycles"], profile["deadline_s"], profile=profile["name"])
            env.tasks[i] = j
            features.append(env._group_obs(np.zeros(OBS_DIM),0,profile["name"],[i])[0])
        self.assertEqual(len({learner._key(6,"user",f) for f in features}),3)

    def test_no_tidal_masks_direct_route_and_not_physics(self):
        a = CrossSlotWLOEnvironment(self.c, seed=17)
        b = CrossSlotWLOEnvironment(dict(self.c,hide_tidal_observations=True),seed=17)
        for env in (a,b):
            env.previous_forecast[:] = [50,100,150]
            env.last_observed[:] = [40,80,120]
            env.high_tide[:] = True
            env.generate()
        j = Task(91,0,0,0,800000,400e6,.55)
        x = a.route_features(j,self.c["region_centers_m"][0])[0]
        y = b.route_features(j,self.c["region_centers_m"][0])[0]
        self.assertEqual(a.arrival_hash,b.arrival_hash)
        np.testing.assert_equal(x[:52],y[:52])
        self.assertTrue((y[56:62]==0).all())
        self.assertEqual(y[52],0)
        self.assertGreater(float(x[57]),0)

    def test_no_tidal_keeps_instantaneous_public_class_counts(self):
        features=[]
        for hide in (False,True):
            env=CrossSlotWLOEnvironment(dict(self.c,hide_tidal_observations=hide),seed=17)
            task=Task(0,0,0,0.,50000.,80e6,.2,profile="interactive")
            env.tasks[0]=task
            env.pending[0].append(0)
            features.append(env.observations()[0][0,20])
        self.assertGreater(features[0],0)
        self.assertEqual(features[0],features[1])

    def test_clock_snaps_roundoff_to_identical_arrival_slots(self):
        env=CrossSlotWLOEnvironment(self.c,seed=17)
        rule=CrossSlotRule("immediate_remote",self.c)
        rule.env=env
        env.slot=200
        env.time=2.-5e-16
        env.step(rule)
        self.assertEqual(env.time,2.0100000000000002)
        self.assertTrue(all(j.arrival==2. for j in env.tasks.values()))

    def test_exclusive_worker_lock_prevents_duplicate_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"worker"
            (Path(tmp)/".worker.worker.lock").write_text("existing owner")
            with self.assertRaises(FileExistsError):
                train_cell(out,self.s)
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError):
                train_cell(out,self.s,resume=True)

    def test_popart_preserves_raw_predictions(self):
        learner = learner_for("TMPPO",self.t,self.c)
        inputs = np.random.normal(size=(9,2*OBS_DIM)).astype(np.float32)
        before = learner.raw_values(inputs)
        learner.normalize_targets(np.linspace(-100,5,1000))
        np.testing.assert_allclose(before,learner.raw_values(inputs),atol=2e-5)

    def test_td3_actual_route_features_and_executed_one_hot_are_replayed(self):
        learner = learner_for("MARATD3-Adapted",self.t,self.c)
        ctx = np.zeros((9,OBS_DIM),np.float32)
        learner.begin(ctx,np.zeros(OBS_DIM),6)
        first, second = np.zeros((1,OBS_DIM),np.float32),np.ones((1,OBS_DIM),np.float32)
        mask = np.zeros((1,11),bool)
        mask[:,3] = True
        action,_ = learner.act(first,mask,"route",[6])
        learner.remember(np.arange(9,dtype=np.float32),[],False)
        learner.act(second,mask,"route",[6])
        replay = learner.replay[0]
        self.assertEqual(int(action[0]),3)
        np.testing.assert_equal(replay["history"][-1,:OBS_DIM],first[0])
        np.testing.assert_equal(replay["next_history"][-1,:OBS_DIM],second[0])
        self.assertEqual(replay["action"].sum(),1.)
        self.assertEqual(replay["action"][5+3],1.)
        self.assertEqual(replay["reward"],6.)
        self.assertAlmostEqual(replay["discount"],self.t["gamma"])

    def test_td3_event_discount_accumulates_elapsed_slots(self):
        learner = learner_for("MARATD3-Adapted",self.t,self.c)
        obs = np.zeros((1,OBS_DIM),np.float32)
        learner.begin(np.zeros((9,OBS_DIM)),np.zeros(OBS_DIM),6)
        learner.act(obs,np.ones((1,3),bool),"user",[0])
        learner.remember(np.ones(9),[],False)
        learner.remember(np.ones(9)*2,[],False)
        learner.act(obs,np.ones((1,3),bool),"user",[0])
        self.assertAlmostEqual(learner.replay[0]["reward"],1+2*self.t["gamma"])
        self.assertEqual(learner.replay[0]["elapsed"],2)
        self.assertAlmostEqual(learner.replay[0]["discount"],self.t["gamma"]**2)

    def test_td3_deterministic_history_retains_executed_previous_action(self):
        learner = learner_for("MARATD3-Adapted",self.t,self.c)
        learner.deterministic = True
        obs = np.zeros((1,OBS_DIM),np.float32)
        mask = np.asarray([[False,False,True]])
        learner.begin(np.zeros((9,OBS_DIM)),np.zeros(OBS_DIM),6)
        learner.act(obs,mask,"user",[0])
        learner.act(obs,mask,"user",[0])
        self.assertEqual(learner.histories[(0,"user",0.)][-1,OBS_DIM+2],1.)
        self.assertFalse(learner.replay)

    def test_three_algorithms_have_same_initial_deterministic_actions(self):
        obs = np.zeros((3,OBS_DIM),np.float32)
        obs[:,17:19] = [[.4,.6],[.8,.3],[.4,.5]]
        obs[:,26] = obs[:,17:19].min(-1)+.005
        mask = np.ones((3,3),bool)
        actions = []
        for method in ("TMPPO","IPPO","MARATD3-Adapted"):
            learner = learner_for(method,self.t,self.c)
            learner.deterministic = True
            learner.begin(np.zeros((9,OBS_DIM)),np.zeros(OBS_DIM),6)
            actions.append(learner.act(obs,mask,"user",[0,1,2])[0])
        np.testing.assert_equal(actions[0],actions[1])
        np.testing.assert_equal(actions[1],actions[2])

    def test_strict_local_never_publishes_and_energy_cost_audits(self):
        m, env = evaluate(CrossSlotRule("strict_local_only",self.c),self.c,814001,30.,40,0,True)
        self.assertEqual(m["published"],0)
        self.assertIsNone(m["ql_m"])
        self.assertTrue(all(r["node"]<env.u for r in env.rows))

    def test_training_reward_matches_drained_task_objective(self):
        env = CrossSlotWLOEnvironment(self.c,seed=17,collect=True)
        rule = CrossSlotRule("forecast_release",self.c)
        rule.env = env
        total = 0.
        for step in range(260):
            total += -env.step(rule,arrivals=step<50)[0]*env.u*self.c["arrival_per_user_s"]*self.c["dt_s"]
        self.assertFalse(env.tasks)
        self.assertAlmostEqual(total,env.metrics()["mean_task_cost"]*env.arrivals,places=8)

    def test_validation_does_not_change_rng_or_td3_open_history(self):
        learner = learner_for("MARATD3-Adapted",self.t,self.c)
        env = CrossSlotWLOEnvironment(self.c,seed=19)
        for _ in range(12):
            _, rec, *_ = env.step(learner)
            learner.remember(env.agent_rewards(.7),rec,False)
        before = copy.deepcopy(learner.state())
        random_before = rng_state()
        with tempfile.TemporaryDirectory() as tmp:
            validation_snapshot("MARATD3-Adapted",learner,self.t,self.c,self.s,0,Path(tmp))
        after = learner.state()
        self.assertEqual(before["steps"],after["steps"])
        self.assertEqual(set(before["open_events"]),set(after["open_events"]))
        for key in before["histories"]:
            np.testing.assert_equal(before["histories"][key],after["histories"][key])
        self.assertTrue(torch.equal(random_before["torch"],rng_state()["torch"]))
        np.testing.assert_equal(random_before["numpy"][1],rng_state()["numpy"][1])

    def test_environment_phases_and_streams_match_between_methods(self):
        for episode,phase in enumerate((50.,150.,250.)):
            a = new_environment(self.c,87021,episode,3)
            b = new_environment(self.c,87021,episode,3)
            a.generate(); b.generate()
            self.assertEqual(a.phase_s,phase)
            self.assertEqual(a.arrival_hash,b.arrival_hash)

    def test_pause_resume_exact_and_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/"full",Path(tmp)/"resumed"
            self.assertTrue(train_cell(a,self.s,wall_seconds=120))
            self.assertFalse(train_cell(b,self.s,wall_seconds=120,stop_after=1))
            self.assertTrue(train_cell(b,self.s,resume=True,wall_seconds=120))
            x = torch.load(a/"latest.pt",weights_only=False)
            y = torch.load(b/"latest.pt",weights_only=False)
            for key in ("actors","critic"):
                for name in x["learner"][key]:
                    torch.testing.assert_close(x["learner"][key][name],y["learner"][key][name],atol=0,rtol=0)
            self.assertEqual(x["validation"],y["validation"])
            with self.assertRaises(FileExistsError):
                train_cell(a,self.s)
            changed = dict(self.s,updates=3)
            with self.assertRaises(ValueError):
                train_cell(b,changed,resume=True)

    def test_td3_finite_update_and_checkpoint_roundtrip(self):
        self.t.update(td3_start_steps=4,td3_batch=2,rollout_steps=8)
        learner = learner_for("MARATD3-Adapted",self.t,self.c)
        env = CrossSlotWLOEnvironment(self.c,seed=17)
        for _ in range(32):
            _,rec,*_ = env.step(learner)
            learner.remember(env.agent_rewards(.7),rec,False)
        diag = learner.update()
        self.assertGreater(diag["td3_updates"],0)
        self.assertTrue(np.isfinite(list(diag.values())).all())
        clone = learner_for("MARATD3-Adapted",self.t,self.c)
        clone.load(copy.deepcopy(learner.state()))
        self.assertEqual(len(clone.replay),len(learner.replay))
        self.assertEqual(set(clone.histories),set(learner.histories))


if __name__ == "__main__":
    unittest.main()

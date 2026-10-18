import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "model")]
from environment import OBS_DIM
from learning_contract_campaign import STUDIES, registered_settings
from learning_contract_cloud import identity
from paper_convergence import configuration, learner_for


class RoleActorCreditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        np.random.seed(83)
        torch.set_num_threads(1)

    def fixture(self, variant):
        cfg, train = configuration(registered_settings(variant))
        train.update(value_normalization=False, actor_cost_prior_scale=0.,
                     actor_warmup_steps=0, epochs=1, minibatch=512)
        learner = learner_for("TMPPO", train, cfg)
        for p in learner.critic.parameters():
            p.data.zero_()
        context = np.zeros((63, OBS_DIM), np.float32)
        glob = np.zeros(OBS_DIM, np.float32)
        for i in range(64):
            learner.begin(context, glob, 60)
            action = i % 2
            recs = [dict(stage="user", obs=context[0].copy(),
                         mask=np.array([True, False, True]), agent=0,
                         action=action * 2, logp=-np.log(2.)),
                    dict(stage="route", obs=context[1].copy(),
                         mask=np.array([True, True] + [False] * 9), agent=1,
                         action=action, logp=-np.log(2.))]
            learner.remember(np.full(63, -.1 if action else -1., np.float32), recs, True)
        return learner, context, glob

    def test_frozen_actor_unchanged_other_actor_and_critic_update(self):
        for variant, frozen, active in (("role_route_only", "user", "route"),
                                        ("role_wlo_only", "route", "user")):
            learner, context, glob = self.fixture(variant)
            before = copy.deepcopy(learner.state())
            learner.update(context, glob)
            for key, value in learner.actors[frozen].state_dict().items():
                torch.testing.assert_close(value, before["actors"][frozen + "." + key], atol=0, rtol=0)
            self.assertTrue(any(not torch.equal(v, before["actors"][active + "." + k])
                                for k, v in learner.actors[active].state_dict().items()))
            self.assertTrue(any(not torch.equal(v, before["critic"][k])
                                for k, v in learner.critic.state_dict().items()))

    def test_empty_freeze_flag_has_exact_legacy_update(self):
        a, context, glob = self.fixture("role_joint")
        b = copy.deepcopy(a)
        b.cfg.pop("frozen_actor_stages")
        rng = torch.get_rng_state()
        a.update(context, glob)
        torch.set_rng_state(rng)
        b.update(context, glob)
        for module in ("actors", "critic"):
            for key, value in getattr(a, module).state_dict().items():
                torch.testing.assert_close(value, getattr(b, module).state_dict()[key], atol=0, rtol=0)

    def test_freeze_contract_cannot_be_silently_resumed_as_joint(self):
        cfg, train = configuration(registered_settings("role_route_only"))
        a = learner_for("TMPPO", train, cfg)
        _, joint = configuration(registered_settings("role_joint"))
        b = learner_for("TMPPO", joint, cfg)
        with self.assertRaises(ValueError):
            b.load(a.state())
        clone = learner_for("TMPPO", train, cfg)
        clone.load(copy.deepcopy(a.state()))
        with self.assertRaises(ValueError):
            learner_for("TMPPO", dict(train, frozen_actor_stages=["not_a_stage"]), cfg)

    def test_three_case_budget_and_unchanged_physical_objective(self):
        cases = [registered_settings(v) for v in STUDIES["role_credit"]]
        self.assertEqual(len(cases), 3)
        self.assertEqual(sum(c["updates"] * c["rollout"] for c in cases), 49152)
        base_cfg, base_train = configuration(registered_settings("team_value_shared"))
        for case in cases:
            cfg, train = configuration(case)
            self.assertEqual(cfg, base_cfg)
            train.pop("frozen_actor_stages")
            self.assertEqual(train, base_train)
        self.assertEqual(identity("Mustard-0/tidal-offload-sim", "role-actor-credit-20261018T140000Z"),
                         "role-credit-campaign-20261018T140000Z")


if __name__ == "__main__":
    unittest.main()

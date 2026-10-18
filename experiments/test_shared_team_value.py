import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"model")]
from learning_contract_campaign import registered_settings, STUDIES
from paper_convergence import configuration, learner_for
from environment import OBS_DIM


class SharedTeamValueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        torch.set_num_threads(1)
        self.cfg,self.train=configuration(registered_settings("team_value_shared"))
        self.context=np.random.default_rng(51).normal(size=(63,OBS_DIM)).astype(np.float32)
        self.glob=np.random.default_rng(52).normal(size=OBS_DIM).astype(np.float32)

    def test_common_team_predictions_and_agent_order_invariance(self):
        policy=learner_for("TMPPO",self.train,self.cfg)
        self.assertEqual(policy.method,"TMAPO")
        policy.set_training_clock(3000,2799,6000)
        x=policy.critic_input(self.context,self.glob)
        np.testing.assert_equal(x,np.broadcast_to(x[0],x.shape))
        values=policy.raw_values(x)
        np.testing.assert_allclose(values,values[0],atol=1e-6)
        y=policy.critic_input(self.context[::-1],self.glob)
        np.testing.assert_allclose(x,y,atol=1e-6)

    def test_actor_and_physical_objective_are_not_modified(self):
        cfg,control=configuration(registered_settings("team_value_control"))
        self.assertEqual(cfg,self.cfg)
        a=learner_for("TMPPO",control,cfg)
        b=learner_for("TMPPO",self.train,self.cfg)
        with torch.no_grad():
            for p in a.actors.parameters():p.normal_()
        b.actors.load_state_dict(a.actors.state_dict())
        torch.testing.assert_close(a.policy_logits("user",a.tensor(self.context)),
                                   b.policy_logits("user",b.tensor(self.context)),atol=0,rtol=0)
        self.assertEqual(control["gamma"],1.)
        self.assertEqual(control["gae_lambda"],1.)
        self.assertEqual(control["entropy"],0.)
        self.assertEqual({k:v for k,v in control.items() if k!="shared_team_value"},
                         {k:v for k,v in self.train.items() if k!="shared_team_value"})

    def test_checkpoint_contract_and_decentralized_critic_are_explicit(self):
        shared=learner_for("TMPPO",self.train,self.cfg)
        _,train=configuration(registered_settings("team_value_control"))
        control=learner_for("TMPPO",train,self.cfg)
        with self.assertRaises(ValueError):shared.load(control.state())
        with self.assertRaises(ValueError):control.load(shared.state())
        clone=learner_for("TMPPO",self.train,self.cfg)
        clone.load(copy.deepcopy(shared.state()))
        with self.assertRaises(ValueError):learner_for("IPPO",self.train,self.cfg)

    def test_two_case_budget_and_release_identity(self):
        from learning_contract_cloud import identity
        self.assertEqual(len(STUDIES["shared_value"]),2)
        total=sum(registered_settings(v)["updates"]*registered_settings(v)["rollout"]
                  for v in STUDIES["shared_value"])
        self.assertEqual(total,32768)
        self.assertEqual(identity("Mustard-0/tidal-offload-sim","shared-team-value-20261018T130000Z"),
                         "shared-value-campaign-20261018T130000Z")


if __name__=="__main__":unittest.main()

"""Standard-library-only checks; safe on a low-memory desktop."""
import unittest
from .wait_gate import fill_probability,positive_gain
from .protocol import configuration,plan


class GateTests(unittest.TestCase):
    def test_no_events_no_future_users(self):
        self.assertEqual(fill_probability(34,1,5,0,.2,.02),0.)

    def test_distinct_users_not_global_population(self):
        self.assertEqual(fill_probability(34,33,2,10000,.2,.02),0.)

    def test_sufficient_releases_now(self):
        self.assertEqual(fill_probability(34,30,0,0,.2,.01),1.)

    def test_fill_probability_respects_rate_and_time(self):
        low=fill_probability(34,2,10,5,.2,.02)
        high=fill_probability(34,2,10,500,.2,.02)
        self.assertLess(low,high)
        self.assertGreater(high,.8)

    def test_cost_gain_not_quality_loss(self):
        self.assertFalse(positive_gain([.1]*16,[.11]*16)[0])
        self.assertTrue(positive_gain([.2]*16,[.1]*16)[0])

    def test_uncertainty_can_reject_favorable_mean(self):
        self.assertFalse(positive_gain([1.,1.],[.1,1.8])[0])

    def test_simple_finite_plan(self):
        self.assertEqual(len(plan()['cells']),6)
        self.assertFalse(plan()['lr_grid'])
        for m in ('TMAPO','IPPO','PPO'):
            c,t=configuration(m)
            self.assertEqual(c['region_user_counts'],[34,33,33])
            self.assertEqual(t['actor_cost_prior_scale'],0.)
            self.assertEqual(t['episodes'],2000)
            self.assertEqual(c['wait_max_s'],.02)
            self.assertEqual(c['weight_quality'],0.)


if __name__=='__main__':unittest.main()

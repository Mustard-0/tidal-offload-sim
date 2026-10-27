"""Standard-library-only checks; safe on a low-memory desktop."""
import unittest
from .wait_gate import fill_probability,positive_gain
from .protocol import configuration,plan
from .credit import mean_preserving_credit


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

    def test_credit_mean_is_conserved_for_unequal_regions(self):
        costs=[.01]*34+[.03]*33+[.07]*33
        for alpha in (0.,.7,1.):
            values,team=mean_preserving_credit(costs,[34,33,33],.03,alpha)
            self.assertAlmostEqual(sum(values[:100])/100,team)
            self.assertAlmostEqual(sum(values[100:])/3,team)
            self.assertAlmostEqual(sum(values)/103,team)

    def test_credit_has_no_favorable_bonus(self):
        values,team=mean_preserving_credit([0.,.2,.1,.3,.1],[2,2,1],.03,.7)
        self.assertTrue(all(v<=0 for v in values))
        self.assertLess(team,0.)
        with self.assertRaises(ValueError):mean_preserving_credit([-.1],[1],.03,.7)

    def test_only_TMAPO_credit_component_changes(self):
        for method in ('TMAPO','IPPO','PPO'):
            c,t=configuration(method)
            self.assertEqual(t['local_credit_fraction'],.7 if method=='TMAPO' else 0.)
            self.assertEqual(c['weight_time'],.25)
            self.assertEqual(c['weight_energy'],.25)


if __name__=='__main__':unittest.main()

import copy
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binom
from common import ROOT,read_json,seed_all
from privacy import amplification_threshold,delta_bound,local_epsilon,planar_noise
from environment import OBS_DIM,Environment,GreedyPolicy,LocalOnlyPolicy,Task
from physics import propagation,rate
from learners import make_policy
from evaluation import evaluate


def enumeration(e0,n,e):
    p=math.exp(e0); alpha=1/(p+1); result=0
    # Enumerate the joint P/Q construction, independent of the survival-tail implementation.
    for c in range(n):
        weight=binom.pmf(c,n-1,2*alpha)
        for a in range(c+2):
            left=binom.pmf(a-1,c,.5); right=binom.pmf(a,c,.5)
            P=alpha*(p*left+right); Q=alpha*(left+p*right)
            result+=weight*max(0,P-math.exp(e)*Q)
    return result


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(ROOT/"config.json")
        self.config["environment"]["users_per_region"]=2
        self.config["training"].update(hidden=16,epochs=1,rollout_steps=4,minibatch=64,
                                        pilot_warmup_s=.01,pilot_measure_s=.1,td3_start_steps=2,td3_batch=2)
        self.c=self.config["environment"]
        seed_all(88,1)

    def test_local_only_never_publishes_or_routes_remotely(self):
        e=Environment(self.c,88,"Local-Only",collect=True)
        policy=LocalOnlyPolicy()
        for _ in range(20):
            e.step(policy)
        for _ in range(205):
            e.step(policy,arrivals=False)
            if not e.tasks:
                break
        self.assertEqual(e.measure_published,0)
        self.assertFalse(e.batch_rows)
        self.assertTrue(all(r["node"] < e.u for r in e.rows))

    def test_accountant_matches_enumeration(self):
        for n in (1,2,5,17):
            for e0 in (.7,1.4,3.):
                self.assertAlmostEqual(delta_bound(e0,n,.5),enumeration(e0,n,.5),places=11)

    def test_feasible_calibration_and_singleton(self):
        for n in (1,2,17,50,100):
            eps=local_epsilon(self.c,n); e0=eps*2*self.c["region_radius_m"]
            self.assertLessEqual(delta_bound(e0,n,1.),1.000001e-6)
            self.assertGreaterEqual(e0,1.)
        self.assertAlmostEqual(local_epsilon(self.c,1,True),.001)

    def test_batch_threshold_from_accountant(self):
        c=copy.deepcopy(self.c); c["users_per_region"]=100
        for eps,want in ((.5,20),(1.,30),(2.,74),(4.,None)):
            c["epsilon_req"]=eps
            self.assertEqual(amplification_threshold(c),want)

    def test_no_wait_when_amplification_is_impossible(self):
        c=copy.deepcopy(self.c); c["users_per_region"]=20
        e=Environment(c,12,"TMAPO")
        self.assertIsNone(e.min_batch)
        task=Task(0,0,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
        e.tasks[task.id]=task
        e.staging[0].append(task.id)
        _,mask=e.batch_features(0,np.zeros(OBS_DIM,np.float32))
        self.assertEqual(mask.tolist(),[False,True])

    def test_actual_noise_mean_unbounded(self):
        r=np.linalg.norm(planar_noise(np.random.default_rng(9),.001,100000),axis=1)
        self.assertLess(abs(r.mean()-2000),20)
        self.assertGreater(r.max(),10000)

    def test_observation_ablation_masks(self):
        a=Environment(self.c,1,"TMAPO-NoWait"); a.generate()
        # Force a head task if this small slot has no arrivals.
        if not a.tasks:
            j=Task(0,0,0,0,self.c["task_bits"],self.c["task_cycles"],.35)
            a.tasks[0]=j; a.pending[0].append(0)
        ob,gl,ma,users,_=a.observations()
        self.assertTrue(all(not ma[u,0] for u in users))
        a.method="TMAPO-NoPred"; a.previous_forecast[:]=123
        ob,gl,*_=a.observations()
        self.assertTrue((ob[:a.u,4:6]==0).all()); self.assertTrue((gl[28:34]==0).all())

    def test_release_actual_distinct_users(self):
        a=Environment(self.c,1)
        tasks=[Task(i,i,0,0,4e5,2.4e8,.35) for i in range(2)]
        a.release(tasks)
        self.assertTrue(all(j.batch_n==2 for j in tasks))
        self.assertTrue(all(j.report_ready>=self.c["privacy_processing_s"] for j in tasks))
        with self.assertRaises(AssertionError): a.release([tasks[0],tasks[0]])

    def test_regional_release_waits_for_actual_distinct_batch(self):
        c=copy.deepcopy(self.c)
        c["users_per_region"]=30
        e=Environment(c,12,"TMAPO",collect=True)
        self.assertEqual(e.min_batch,30)
        e.windows_seen=1
        e.last_observed[:]=6000

        class Policy:
            def begin(self,*args): pass
            def act(self,obs,mask,stage,agents):
                if stage=="user": return np.where(mask[:,2],2,1),[]
                if stage=="batch": return np.where(mask[:,0],0,1),[]
                return np.argmax(mask,axis=1),[]

        for user in range(29):
            task=Task(user,user,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
            e.tasks[task.id]=task; e.pending[user].append(task.id)
            e.arrivals+=1; e.measure_arrivals+=1
        e.serial=29
        e.step(Policy(),arrivals=False)
        self.assertEqual(len(e.staging[0]),29)
        self.assertEqual(e.measure_published,0)
        self.assertAlmostEqual(e.tasks[0].active_wait,c["dt_s"])
        task=Task(29,29,0,e.time,c["task_bits"],c["task_cycles"],c["deadline_s"])
        e.tasks[29]=task; e.pending[29].append(29)
        e.arrivals+=1; e.measure_arrivals+=1
        e.step(Policy(),arrivals=False)
        self.assertFalse(e.staging[0])
        self.assertEqual(e.measure_published,30)
        self.assertEqual(e.batch_rows[0]["n_distinct"],30)
        self.assertEqual(e.batch_rows[0]["release_reason"],"target_met")
        self.assertGreater(e.batch_rows[0]["epsilon0"],c["epsilon_req"])
        self.assertEqual(e.measure_amplified,30)

    def test_target_batch_cannot_wait_after_reaching_threshold(self):
        c=copy.deepcopy(self.c)
        c["users_per_region"]=30
        e=Environment(c,14,"TMAPO")
        for user in range(30):
            task=Task(user,user,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
            task.status="staged"
            e.tasks[user]=task
            e.staging[0].append(user)
            e.staged_users.add(user)
        _,mask=e.batch_features(0,np.zeros(64,np.float32))
        self.assertEqual(mask.tolist(),[False,True])

    def test_route_reservations_mask_projected_queue_overflow(self):
        e=Environment(self.c,18,"TMAPO")
        j=Task(0,0,0,0,self.c["task_bits"],self.c["task_cycles"],self.c["deadline_s"])
        j.report=e.xy[0].copy()
        count=np.zeros(11,dtype=int)
        work=np.zeros(11)
        visible=np.flatnonzero(e.visibility[0])
        self.assertGreater(len(visible),1)
        node=int(visible[0])
        base,mask,_=e.route_features(j,reservations=(count,work))
        self.assertTrue(mask[node])
        count[node]=2
        cpu=e.speed(e.u+node)
        kind=e.node_kind(e.u+node)
        work[node]=self.c["queue_cap_s"][kind]*cpu
        changed,mask,_=e.route_features(j,reservations=(count,work))
        self.assertFalse(mask[node])
        self.assertGreater(changed[node],base[node])
        self.assertTrue(mask.any())
        work[:]=self.c["queue_cap_s"][kind]*cpu*10
        _,fallback,_=e.route_features(j,reservations=(count,work))
        self.assertTrue(fallback.any())

    def test_inflight_upload_reserves_future_compute_capacity(self):
        c=copy.deepcopy(self.c)
        probe=Environment(c,19,"TMAPO")
        node=int(np.flatnonzero(probe.visibility[0])[0])
        cpu=probe.speed(probe.u+node)
        c["queue_cap_s"][probe.node_kind(probe.u+node)]=1.5*c["task_cycles"]/cpu
        e=Environment(c,19,"TMAPO")
        task=Task(1,0,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
        task.report=e.xy[0].copy()
        empty=(np.zeros(11,dtype=int),np.zeros(11))
        _,before,_=e.route_features(task,reservations=empty)
        self.assertTrue(before[node])
        uploading=Task(2,1,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
        uploading.status="upload"
        uploading.node=node
        uploading.remaining=c["task_bits"]
        e.tasks[uploading.id]=uploading
        feature,after,_=e.route_features(task,reservations=empty)
        self.assertFalse(after[node])
        self.assertGreaterEqual(feature[22+node],c["task_cycles"]/cpu/c["time_ref_s"])

    def test_same_slot_assignment_is_not_counted_twice(self):
        c=self.c
        e=Environment(c,21,"TMAPO")
        for user in (0,1):
            task=Task(user,user,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
            task.status="report"
            task.report=e.xy[user].copy()
            task.report_ready=0
            e.tasks[task.id]=task

        class FirstVisible:
            route_observations=[]
            def begin(self,context,glob,n_user):
                pass
            def act(self,obs,mask,stage,agents):
                if stage=="route":
                    self.route_observations.append(obs[0].copy())
                return np.asarray([np.flatnonzero(row)[0] for row in mask]),[]

        policy=FirstVisible()
        order=np.random.default_rng(np.random.SeedSequence([e.seed,4,e.slot])).permutation(2)
        second_xy=e.tasks[int(order[1])].report.copy()
        e.step(policy,arrivals=False)
        self.assertEqual(len(policy.route_observations),2)
        first_node=int(np.flatnonzero(e.visibility[0])[0])
        cpu=e.speed(e.u+first_node)
        tx=c["task_bits"]/rate(c,second_xy,first_node,0,2,1)
        expected=(tx+2*c["task_cycles"]/cpu+
                  2*propagation(c,second_xy,first_node,0))/c["time_ref_s"]
        self.assertAlmostEqual(policy.route_observations[1][first_node],expected,places=5)

    def test_failed_task_counts_for_qos_but_not_completed_delay(self):
        e=Environment(self.c,20,"TMAPO")
        task=Task(1,0,0,0,self.c["task_bits"],self.c["task_cycles"],self.c["deadline_s"])
        e.tasks[task.id]=task
        e.pending[task.user].append(task.id)
        e.measure_arrivals=1
        e.time=.1
        e.end(task,failed=True)
        result=e.metrics()
        self.assertEqual(result["qos_rate"],1)
        self.assertEqual(result["completion_rate"],0)
        self.assertIsNone(result["delay_s"])

    def test_measured_cohort_uses_slots_not_event_roundoff(self):
        e=Environment(self.c,15,measure_start=.5,measure_end=3.5)
        j=Task(0,0,0,.499999999999,self.c["task_bits"],self.c["task_cycles"],.35,
               arrival_slot=50)
        self.assertTrue(e.measured(j))
        j.arrival_slot=350
        j.arrival=3.499999999999
        self.assertFalse(e.measured(j))

    def test_regional_wait_cap_forces_fallback(self):
        c=copy.deepcopy(self.c)
        c.update(users_per_region=30,wait_max_s=.02)
        e=Environment(c,13,"TMAPO",collect=True)
        e.windows_seen=1
        e.last_observed[:]=3000000

        class Policy:
            def begin(self,*args): pass
            def act(self,obs,mask,stage,agents):
                if stage=="user": return np.where(mask[:,2],2,1),[]
                if stage=="batch": return np.where(mask[:,0],0,1),[]
                return np.argmax(mask,axis=1),[]

        for user in range(5):
            task=Task(user,user,0,0,c["task_bits"],c["task_cycles"],c["deadline_s"])
            e.tasks[user]=task; e.pending[user].append(user)
            e.arrivals+=1; e.measure_arrivals+=1
        for _ in range(3): e.step(Policy(),arrivals=False)
        self.assertEqual(e.measure_published,5)
        self.assertEqual(e.batch_rows[0]["release_reason"],"insufficient_wait_budget")
        self.assertEqual(e.batch_rows[0]["epsilon0"],c["epsilon_req"])
        self.assertEqual(e.measure_amplified,0)
        self.assertAlmostEqual(e.tasks[0].active_wait,.02)

    def test_paired_exogenous_trace_and_cost(self):
        configs=["TMAPO","TMAPO-NoAmp"]
        hashes=[]
        for method in configs:
            e=Environment(self.c,23,method)
            p=GreedyPolicy(self.c)
            for _ in range(20): e.step(p)
            for _ in range(210):
                if not e.tasks: break
                e.step(p,False)
            m=e.metrics(); hashes.append(m["trace_hash"])
            self.assertAlmostEqual(e.cost_total,e.sum_cost,places=7)
            self.assertEqual(e.arrivals,e.ended)
            self.assertAlmostEqual(m["energy_j_per_arrival"],e.sum_energy/e.arrivals)
        self.assertEqual(*hashes)

    def test_report_does_not_change_physical_route(self):
        base=Environment(self.c,4)
        j=Task(0,0,0,0,4e5,2.4e8,.35,status="selected")
        base.tasks[0]=j; base.arrivals=base.measure_arrivals=1
        base.release([j]); base.assign(j,0)
        a=copy.deepcopy(base); b=copy.deepcopy(base)
        b.tasks[0].report+=1e6; b.tasks[0].ql+=1e6
        for env in (a,b):
            while env.tasks: env.advance(env.time+.01)
        self.assertAlmostEqual(a.sum_delay,b.sum_delay)
        self.assertAlmostEqual(a.sum_energy,b.sum_energy)
        self.assertAlmostEqual(a.sum_cost,b.sum_cost)

    def test_learners_update_and_restore(self):
        for method in ("TMAPO","IPPO","MARATD3-Adapted"):
            p=make_policy(method,self.config)
            e=Environment(self.c,5,method)
            for _ in range(8):
                r,rec,_,_=e.step(p); p.remember(r,rec,False)
            context,glob,*_=e.observations()
            stats=p.update(context,glob)
            self.assertTrue(all(np.isfinite(v) for v in stats.values()))
            q=make_policy(method,self.config); q.load(copy.deepcopy(p.state()))
            p.deterministic=q.deterministic=True
            p.reset_history(); q.reset_history(); p.begin(context,glob,e.u); q.begin(context,glob,e.u)
            mask=np.ones((len(context),3),bool); ids=np.arange(len(context))
            self.assertTrue(np.array_equal(p.act(context,mask,"user",ids)[0],q.act(context,mask,"user",ids)[0]))

    def test_cost_guidance_initial_decisions_and_mask(self):
        p=make_policy("TMAPO",self.config)
        with torch.no_grad():
            for layer in p.actors.values():
                for parameter in layer.parameters():
                    parameter.zero_()
        p.deterministic=True
        obs=np.zeros((1,64),np.float32)
        obs[0,17:19]=[.8,.2]
        self.assertEqual(p.act(obs,np.array([[False,True,True]]),"user",np.array([0]))[0][0],2)
        obs[0,17:19]=[.1,.5]
        self.assertEqual(p.act(obs,np.array([[False,True,True]]),"user",np.array([0]))[0][0],1)
        obs=np.zeros((1,64),np.float32)
        obs[0,:11]=.4
        obs[0,2]=.1
        obs[0,45]=.35
        self.assertEqual(p.act(obs,np.ones((1,11),bool),"route",np.array([self.c["users_per_region"]*3]))[0][0],2)
        mask=np.ones((1,11),bool); mask[0,2]=False
        self.assertNotEqual(p.act(obs,mask,"route",np.array([self.c["users_per_region"]*3]))[0][0],2)

    def test_evaluation_drains_and_finite(self):
        p=GreedyPolicy(self.c)
        rows,tasks,batches=evaluate(p,self.config,"Greedy",[61001],collect=True)
        self.assertEqual(len(tasks),rows[0]["arrivals"])
        self.assertTrue(0<=rows[0]["qos_rate"]<=1)

    def test_rejection_cost_and_energy_denominator(self):
        e=Environment(self.c,2)
        # Third simultaneous local admission exceeds 0.5 seconds of work.
        for i in range(3):
            j=Task(i,0,0,0,4e5,2.4e8,.35,status="selected")
            e.tasks[i]=j; e.arrivals+=1; e.measure_arrivals+=1
            e.enqueue_compute(j,0)
        while e.tasks: e.advance(e.time+.01)
        m=e.metrics()
        self.assertEqual(m["failed"],1)
        self.assertAlmostEqual(m["energy_j_per_arrival"],.48/3)
        self.assertAlmostEqual(m["qos_rate"],2/3)  # rejected third + late second
        self.assertAlmostEqual(e.cost_total,e.sum_cost,places=7)

    def test_no_publications_are_missing_not_zero(self):
        e=Environment(self.c,4)
        j=Task(0,0,0,0,4e5,2.4e8,.35,status="selected")
        e.tasks[0]=j; e.arrivals=e.measure_arrivals=1
        e.enqueue_compute(j,0)
        while e.tasks: e.advance(e.time+.01)
        m=e.metrics()
        self.assertIsNone(m["ql_m"]); self.assertEqual(m["publication_ratio"],0)

    def test_plan_counts(self):
        from run import plan
        expected=self.config["experiments"]
        for kind,n in (("privacy",len(expected["privacy_methods"])*len(expected["epsilons"])),
                       ("main",len(expected["main_methods"])*len(expected["users_per_region"])),
                       ("convergence",5)):
            jobs=plan(self.config,kind,2026)
            self.assertEqual(len(jobs),n); self.assertEqual(len({j["id"] for j in jobs}),n)


if __name__=="__main__": unittest.main()

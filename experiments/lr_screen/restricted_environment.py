"""Reviewed transfer privacy with the original RL offloading action pipeline.

QL is diagnostic only. Active waiting is conservatively masked until physical
benefits, not QL proxy gains, can justify it. No greedy controller is imported.
"""
from collections import defaultdict, deque
from functools import lru_cache
import math
import hashlib
import struct
from .runtime import np
from cross_slot_environment import CrossSlotWLOEnvironment
from .vendor.accountant import calibrate, delta_bound
from .mechanism import actual_batch, report_for


@lru_cache(maxsize=512)
def expected_quality(eps0, radius=500., shrink=.5):
    """Public uniform-disk prior integration, used ONLY for action guidance.

    Integrates the actual bounded density and postprocessing; no truth lookup.
    Reported reward/metrics always use the realized sampled location loss.
    """
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(24)
    radii, weights = (x+1)/2, w*(x+1)/2
    angle = (np.arange(96)+.5)*2*np.pi/96
    output_radius = radii[:, None]
    total = 0.
    for r, wi in zip(radii, weights):
        d = np.sqrt(np.maximum(0., r*r+output_radius**2-2*r*output_radius*np.cos(angle)))
        post_d = np.sqrt(np.maximum(0., r*r+(shrink*output_radius)**2-2*r*shrink*output_radius*np.cos(angle)))
        density = np.exp(-eps0*d/2) * weights[:, None]
        total += wi * float((post_d*density).sum()/density.sum())
    return float(total*radius)


class CandidateRestrictedEnvironment(CrossSlotWLOEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.cfg
        if (self.u != 300 or c["region_radius_m"] != 500. or c["epsilon_req"] != 1.
                or c["delta_req"] != 1e-6 or c["wait_max_s"] != .02
                or c["report_shrink"] != .5):
            raise ValueError("This study fixes N300, R500m, epsilon1, delta1e-6, wait20ms, shrink0.5")
        self.admission_history = defaultdict(deque)
        self.max_observed_wait = 0.
        self.draining = False
        self.arrival_stop_slot = None
        self.release_serial = 0
        self.wait_decision_rows = []

    def generate(self):
        first = self.serial
        super().generate()
        # Include task class/sizes/deadlines as well as counts in pairing proof.
        h = hashlib.sha256(bytes.fromhex(self.arrival_hash))
        for task_id in range(first, self.serial):
            j = self.tasks[task_id]
            h.update(struct.pack("<qqqdddd", j.id, j.user, j.arrival_slot,
                                 j.arrival, j.bits, j.cycles, j.deadline))
            h.update(j.profile.encode("utf-8")+b"\0")
        self.arrival_hash = h.hexdigest()

    def eps0(self, n):
        if n < 1 or int(n) != n or n > self.cfg["users_per_region"]:
            raise ValueError("n is actual within-region distinct batch count")
        return calibrate(int(n), self.cfg["epsilon_req"], self.cfg["delta_req"],
                         self.cfg["accountant_grid"], self.cfg["accountant_max_eps0"])

    def q_estimate(self, n):
        return expected_quality(self.eps0(n), self.cfg["region_radius_m"], self.cfg["report_shrink"])

    def report_quality_estimate(self, n):
        return self.q_estimate(n)

    def wait_information(self, tasks):
        """Conservative immediate mode: QL gain is not physical cost gain.

        No active wait until a causal physical-benefit model is validated.
        Offload/local and remote-node actions remain neural policy decisions.
        """
        c = self.cfg
        if c["active_wait_mode"] != "immediate_until_physical_benefit_validated":
            raise ValueError("Unreviewed active-wait mode")
        n = len({j.user for j in tasks})
        max_wait = max(j.active_wait for j in tasks)
        remaining = min(j.deadline-(self.time-j.arrival) for j in tasks)
        horizon = max(0., min(c["wait_max_s"]-max_wait, remaining))
        return dict(n=n, target=self.group_target, probability=0., horizon_s=horizon,
                    expected_quality_gain=0., expected_wait_cost=0., allow=False,
                    reason="physical_wait_benefit_not_validated")

    def _observations(self):
        context, glob, masks, users, ready = super()._observations()
        c = self.cfg
        for u in users:
            j = self.tasks[self.pending[u][0]]
            key = (j.region, j.profile)
            pool = [self.tasks[i] for i in self.class_staging.get(key, ())]
            information = self.wait_information(pool+[j])
            masks[u, 0] = information["allow"]
            context[u, 26] = min(context[u, 17], context[u, 18])+c["weight_time"]*c["dt_s"]/c["time_ref_s"]
            context[u, 24] = information["probability"]
        # Public pool counts must reflect class pools, not the unused legacy pool.
        for g in range(self.g):
            count = sum(len(ids) for (region, _), ids in self.class_staging.items() if region == g)
            glob[37+g] = count/c["users_per_region"]
        glob[46] = float(self.draining)
        glob[47] = (max(0, self.arrival_stop_slot-self.slot)/6000
                    if self.arrival_stop_slot is not None else 0.)
        return context, glob, masks, users, ready

    def _group_obs(self, base, region, profile, ids):
        c = self.cfg
        tasks = [self.tasks[i] for i in ids]
        info = self.wait_information(tasks)
        n = len(tasks)
        remaining = min(j.deadline-(self.time-j.arrival) for j in tasks)
        waited = max(j.active_wait for j in tasks)
        obs = base.copy()
        release_cost = c["weight_quality"]*self.q_estimate(n)/c["quality_ref_m"]
        obs[0:3] = [n/c["users_per_region"], waited/c["wait_max_s"], remaining/c["deadline_s"]]
        obs[17:19] = [release_cost-info["expected_quality_gain"]+info["expected_wait_cost"], release_cost]
        obs[10] = next((i+1 for i, p in enumerate(c["task_profiles"]) if p["name"] == profile), 0)
        obs[20:26] = [n/c["users_per_region"], self.eps0(n), info["horizon_s"]/c["wait_max_s"],
                       info["probability"], info["expected_quality_gain"], info["expected_wait_cost"]]
        obs[59:64] = [n/c["users_per_region"], waited/c["wait_max_s"], remaining/c["deadline_s"],
                      (n-(self.group_target or c["users_per_region"]+1))/c["users_per_region"], -1.]
        if self.collect:
            self.wait_decision_rows.append(dict(time_s=self.time, region=region, task_class=profile, **info))
        return obs, np.asarray([info["allow"], False, True], dtype=bool)

    def release(self, tasks, reason="adaptive_region_class"):
        if not tasks:
            return
        c = self.cfg
        n, eps0, bound = actual_batch(tasks, c)
        region = tasks[0].region
        center = np.asarray(c["region_centers_m"][region], dtype=float)
        beta = eps0/(2*c["region_radius_m"])
        self.max_class_group = max(self.max_class_group, n)
        self.release_serial += 1
        for j in tasks:
            if j.id not in getattr(self, "_staged_at_slot_start", set()):
                if not hasattr(self, "_slot_admissions"):
                    self._slot_admissions = []
                self._slot_admissions.append((j.region, j.profile, j.user))
            # Same task gets the same random stream across policies. Acceptance
            # can differ with actual-batch eps0; never use network N as n.
            truth = self.xy[j.user]
            if np.linalg.norm(truth-center) > c["region_radius_m"]+1e-7:
                raise AssertionError("True position outside its fixed public domain")
            raw, j.report = report_for(truth, center, c["region_radius_m"], eps0,
                                      self.seed, j.id, c["report_shrink"])
            j.ql, j.epsilon, j.batch_n = float(np.linalg.norm(j.report-truth)), beta, n
            j.status, j.release = "report", self.time
            j.report_ready = self.time+max(c["dt_s"], math.ceil(c["privacy_processing_s"]/c["dt_s"]-1e-9)*c["dt_s"])
            self.energy(j, c["report_energy_j"], "report")
            self.noise_publications[j.user] += 1
            self.sum_class_group_wait += j.active_wait
            self.max_observed_wait = max(self.max_observed_wait, j.active_wait)
            if j.active_wait > c["wait_max_s"]+1e-9:
                raise AssertionError("Active waiting exceeded 20ms")
            charge = c["weight_quality"]*j.ql/c["quality_ref_m"]
            self.step_cost += charge
            self.cost_total += charge
            self.user_step_cost[j.user] += charge
            if self.measured(j):
                self.measure_published += 1
                self.measure_amplified += int(eps0 > c["epsilon_req"]+1e-12)
                self.sum_ql += j.ql
                self.sum_ql_theory += self.q_estimate(n)
                self.sum_k += n
        if self.collect:
            self.batch_rows.append(dict(batch_id=self.release_serial, time_s=self.time, region=region,
                task_class=tasks[0].profile, n_distinct=n, epsilon0=eps0, beta_per_m=beta,
                delta_bound=bound, shrink=c["report_shrink"], release_reason=reason,
                task_ids=";".join(str(j.id) for j in tasks), user_ids=";".join(str(j.user) for j in tasks)))

    def step(self, policy, arrivals=True):
        exact_time = self.slot*self.cfg["dt_s"]
        if abs(self.time-exact_time) > 1e-9:
            raise AssertionError("Unexpected slot clock drift")
        self.time = exact_time
        self.draining = not arrivals
        before = {i for i in self.tasks if self.tasks[i].status == "staged"}
        self._staged_at_slot_start = before
        # Validate masks for ALL algorithms, including centralized control.
        class CheckedPolicy:
            def __getattr__(_, name):
                return getattr(policy, name)
            def act(_, obs, mask, stage, agents):
                acts, records = policy.act(obs, mask, stage, agents)
                if len(acts) != len(mask) or not np.isfinite(acts).all():
                    raise AssertionError("Malformed actions")
                if np.any(np.asarray(acts) != np.asarray(acts, dtype=int)):
                    raise AssertionError("Noninteger actions")
                if np.any(np.asarray(acts) < 0) or np.any(np.asarray(acts) >= mask.shape[1]):
                    raise AssertionError("Action out of range")
                if not mask[np.arange(len(acts)), np.asarray(acts, dtype=int)].all():
                    raise AssertionError("Illegal masked action")
                return acts, records
        result = super().step(CheckedPolicy(), arrivals)
        # Admissions are recorded after the slot: a current gate cannot read
        # future actions or the admissions that will occur next slot.
        for j in self.tasks.values():
            if j.status == "staged" and j.id not in before:
                self.admission_history[(j.region, j.profile)].append((self.time, j.user))
        # Immediate releases must also count in observed historical admissions.
        for batch in getattr(self, "_slot_admissions", []):
            self.admission_history[(batch[0], batch[1])].append((self.time, batch[2]))
        self._slot_admissions = []
        for j in self.tasks.values():
            self.max_observed_wait = max(self.max_observed_wait, j.active_wait)
            if j.active_wait > self.cfg["wait_max_s"]+1e-9:
                raise AssertionError("Active wait budget exceeded")
        return result

    def metrics(self):
        result = super().metrics()
        result["quality_cost_is_in_reward"] = self.cfg["weight_quality"] != 0
        result["active_wait_mode"] = self.cfg["active_wait_mode"]
        return result

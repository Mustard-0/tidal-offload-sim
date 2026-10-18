"""Two-stage decisions, continuous transmission/FCFS compute within each slot.

One user HOL task may be released per slot. A regional actor sees only the
shuffled location report and public resource state. Physics reads true XY.
Service cost is integrated while tasks live; QL never enters reward/cost.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import math
import numpy as np

from physics import candidate_nodes, propagation, rate
from privacy import amplification_threshold, local_epsilon
from workload import expected_batch_fill_s, tide_factors, validate_environment

OBS_DIM = 64
N_ROUTE = 11


@dataclass
class Task:
    id: int
    user: int
    region: int
    arrival: float
    bits: float
    cycles: float
    deadline: float
    profile: str = "homogeneous"
    arrival_slot: int = -1
    arrival_tide_factor: float = 1.0
    status: str = "pending"
    node: int = -1
    remaining: float = 0.0
    due: float = math.inf
    active_wait: float = 0.0
    hol_time: float = -1.0
    release: float = -1.0
    report_ready: float = -1.0
    node_choice_time: float = -1.0
    compute_ready: float = -1.0
    compute_start: float = -1.0
    end: float = -1.0
    energy: float = 0.0
    transmit_energy: float = 0.0
    compute_energy: float = 0.0
    report_energy: float = 0.0
    ql: float = 0.0
    epsilon: float = 0.0
    batch_n: int = 0
    report: object = None
    violated: bool = False
    failed: bool = False


class Environment:
    def __init__(self, cfg, seed, method="TMAPO", phase_s=0., measure_start=0.,
                 measure_end=math.inf, collect=False):
        self.cfg, self.seed, self.method = cfg, int(seed), method
        validate_environment(cfg)
        self.phase_s = phase_s
        self.measure_start, self.measure_end = measure_start, measure_end
        dt = cfg["dt_s"]
        self.measure_start_slot = math.ceil(measure_start / dt - 1e-9)
        self.measure_end_slot = (math.ceil(measure_end / dt - 1e-9)
                                 if math.isfinite(measure_end) else None)
        self.collect = collect
        self.g = len(cfg["region_centers_m"])
        self.u = cfg["users_per_region"] * self.g
        self.b = len(cfg["bs_xy_m"])
        self.s = len(cfg["sat_overhead_s"])
        if self.b + self.s != N_ROUTE or self.g != 3:
            raise ValueError("This protocol fixes 3 regions, 8 BSs and 3 satellites")
        self.region = np.repeat(np.arange(self.g), cfg["users_per_region"])
        positions = []
        for g, center in enumerate(cfg["region_centers_m"]):
            draws = np.random.default_rng(np.random.SeedSequence([seed, 1, g])).random((cfg["users_per_region"], 2))
            angle, radius = 2*np.pi*draws[:, 0], cfg["region_radius_m"]*np.sqrt(draws[:, 1])
            positions.append(np.asarray(center) + np.c_[radius*np.cos(angle), radius*np.sin(angle)])
        self.xy = np.concatenate(positions)
        self.pending = [deque() for _ in range(self.u)]
        self.staging = [deque() for _ in range(self.g)]
        self.staged_users = set()
        self.min_batch = amplification_threshold(cfg)
        self.compute = [deque() for _ in range(self.u + N_ROUTE)]
        self.tasks = {}
        self.time, self.slot, self.serial = 0., 0, 0
        self.arrivals = self.ended = 0
        self.arrival_hash = hashlib.sha256(b"").hexdigest()
        self.noise_publications = np.zeros(self.u, dtype=int)
        self.cost_total = self.step_cost = 0.
        self.user_step_cost = np.zeros(self.u)
        self.measure_arrivals = self.measure_ended = self.measure_completed = self.measure_failed = 0
        self.measure_qos = self.measure_published = 0
        self.measure_late_completed = self.measure_expired = 0
        self.sum_delay = self.sum_energy = self.sum_cost = 0.
        self.sum_ql = self.sum_ql_theory = self.sum_k = self.sum_wait = 0.
        self.sum_transmit_energy = self.sum_compute_energy = self.sum_report_energy = 0.
        self.sum_resolution_time = 0.
        self.queue_pressure = np.zeros(self.g)
        self.high_tide = np.zeros(self.g, dtype=bool)
        self.pressure_samples = 0
        self.sum_queue_pressure = np.zeros(self.g)
        self.sum_high_tide = np.zeros(self.g)
        self.sum_compute_busy = np.zeros(N_ROUTE)
        self.measure_amplified = 0
        self.rows, self.batch_rows = [], []
        self.window_counts = np.zeros(self.g)
        self.window_index = 0
        self.level = np.zeros(self.g)
        self.trend = np.zeros(self.g)
        self.pred_error = deque(maxlen=cfg["prediction_error_windows"])
        self.last_observed = np.zeros(self.g)
        self.previous_forecast = np.zeros(self.g)
        self.windows_seen = 0
        self.visibility = np.stack([candidate_nodes(cfg, g, phase_s) for g in range(self.g)])
        self.visibility_time = 0.

    def measured(self, task):
        if task.arrival_slot >= 0:
            return (task.arrival_slot >= self.measure_start_slot and
                    (self.measure_end_slot is None or task.arrival_slot < self.measure_end_slot))
        return self.measure_start <= task.arrival < self.measure_end

    def work(self, node):
        return sum(self.tasks[j].remaining for j in self.compute[node])

    def speed(self, node):
        return self.cfg["cpu_ue"] if node < self.u else self.cfg["cpu_bs"] if node < self.u+self.b else self.cfg["cpu_sat"]

    def node_kind(self, node):
        return 0 if node < self.u else 1 if node < self.u+self.b else 2

    def generate(self):
        c = self.cfg
        t = self.time + self.phase_s
        modulation = tide_factors(c, t)
        # Counter-keyed streams keep existing users' arrivals paired across scales.
        counts = np.concatenate([
            np.random.default_rng(np.random.SeedSequence([self.seed, 2, g, self.slot])).poisson(
                c["arrival_per_user_s"] * modulation[g] * c["dt_s"], c["users_per_region"])
            for g in range(self.g)
        ])
        self.arrival_hash = hashlib.sha256(bytes.fromhex(self.arrival_hash)+counts.astype("<i4").tobytes()).hexdigest()
        for user in np.flatnonzero(counts):
            region = int(self.region[user])
            self.window_counts[region] += counts[user]
            profiles = c.get("task_profiles", [])
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, 6, region, int(user) % c["users_per_region"], self.slot]))
            for _ in range(int(counts[user])):
                p = profiles[int(rng.choice(len(profiles), p=[x["probability"] for x in profiles]))] if profiles else {
                    "name":"homogeneous", "bits":c["task_bits"], "cycles":c["task_cycles"], "deadline_s":c["deadline_s"]}
                j = Task(self.serial, int(user), region, self.time, p["bits"], p["cycles"],
                         p["deadline_s"], profile=p["name"], arrival_slot=self.slot, arrival_tide_factor=float(modulation[region]))
                self.tasks[j.id] = j
                self.pending[user].append(j.id)
                self.serial += 1
                self.arrivals += 1
                self.measure_arrivals += int(self.measured(j))
        nwin = max(1, round(c["prediction_window_s"] / c["dt_s"]))
        if (self.slot + 1) % nwin == 0:
            observed = self.window_counts / (nwin*c["dt_s"])
            if self.windows_seen == 0:
                self.level = observed.copy()
            else:
                old = self.level.copy()
                self.pred_error.append(np.abs(observed-self.previous_forecast))
                self.level = c["holt_level"]*observed + (1-c["holt_level"])*(self.level+self.trend)
                self.trend = c["holt_trend"]*(self.level-old) + (1-c["holt_trend"])*self.trend
            self.previous_forecast = np.maximum(0, self.level+self.trend*c["prediction_horizon_s"]/c["prediction_window_s"])
            self.last_observed = observed
            self.windows_seen += 1
            self.window_counts[:] = 0

    def link_counts(self):
        active = [j for j in self.tasks.values() if j.status == "upload"]
        return Counter(j.node for j in active), Counter(j.user for j in active)

    def committed_remote_work(self):
        work = np.zeros(N_ROUTE)
        for j in self.tasks.values():
            if j.status in ("upload", "propagation"):
                work[j.node] += j.cycles
        return work

    def route_features(self, j, xy=None, reservations=None):
        c = self.cfg
        xy = j.report if xy is None else np.asarray(xy)
        snapshot = getattr(self, "_observation_snapshot", None)
        if snapshot is None:
            by_node, by_user = self.link_counts()
            committed_work = self.committed_remote_work()
            queued_work = [self.work(self.u+node) for node in range(N_ROUTE)]
            errors = np.mean(self.pred_error, axis=0) if self.pred_error else np.zeros(self.g)
        else:
            by_node, by_user, committed_work, queued_work, errors = snapshot
        times, energy, queue = np.zeros(N_ROUTE), np.zeros(N_ROUTE), np.zeros(N_ROUTE)
        reserved_count, reserved_work = reservations if reservations is not None else (
            np.zeros(N_ROUTE, dtype=int), np.zeros(N_ROUTE))
        capacity_mask = np.ones(N_ROUTE, dtype=bool)
        for node in range(N_ROUTE):
            tx = j.bits / rate(c, xy, node, self.time+self.phase_s,
                               by_node[node]+reserved_count[node]+1, by_user[j.user]+1)
            cpu = c["cpu_bs"] if node < self.b else c["cpu_sat"]
            queued_cycles = queued_work[node]+committed_work[node]+reserved_work[node]
            queue[node] = queued_cycles/cpu
            if reservations is not None:
                capacity_mask[node] = (queued_cycles+j.cycles <=
                    c["queue_cap_s"][self.node_kind(self.u+node)]*cpu+1e-5)
            prop = propagation(c, xy, node, self.time+self.phase_s)
            times[node] = tx+queue[node]+j.cycles/cpu+2*prop
            energy[node] = tx*c["user_power_w"]/max(1, by_user[j.user]+1)+j.cycles*c["energy_per_cycle"][1 if node<self.b else 2]
        age = self.time-j.arrival
        cost = c["weight_time"]*(age+times)/c["time_ref_s"]+c["weight_energy"]*energy/c["energy_ref_j"]
        cost += c["weight_violation"]*((age+times)>j.deadline)
        mask = self.visibility[j.region].copy()
        if reservations is not None and np.any(mask & capacity_mask):
            mask &= capacity_mask
        obs = np.zeros(OBS_DIM, np.float32)
        obs[:11] = np.minimum(times/c["time_ref_s"], 20)
        obs[11:22] = np.minimum(energy/c["energy_ref_j"], 20)
        obs[22:33] = queue/c["time_ref_s"]
        obs[33:44] = mask
        obs[44:51] = [age/c["time_ref_s"], (j.deadline-age)/c["time_ref_s"],
                      xy[0]/10000, xy[1]/10000, j.batch_n/c["users_per_region"],
                      j.epsilon*2*c["region_radius_m"], j.region/2]
        scale = 100*c["arrival_per_user_s"]
        obs[51:59] = [self.queue_pressure[j.region], float(self.high_tide[j.region]),
                      j.bits/c["task_bits"], j.cycles/c["task_cycles"],
                      len(self.staging[j.region])/c["users_per_region"],
                      self.last_observed[j.region]/scale, self.previous_forecast[j.region]/scale,
                      errors[j.region]/scale]
        obs[59:62] = self.previous_forecast/scale
        if self.method == "TMAPO-NoPred":
            obs[57:62] = 0
        return np.clip(obs, -100, 100), mask, cost

    def observations(self):
        # No routing, physical time advancement, or queue mutation occurs here.
        # Drop this snapshot before sequential assignments update the upload set.
        by_node, by_user = self.link_counts()
        self._observation_snapshot = (
            by_node, by_user, self.committed_remote_work(),
            [self.work(self.u+node) for node in range(N_ROUTE)],
            np.mean(self.pred_error, axis=0) if self.pred_error else np.zeros(self.g))
        try:
            return self._observations()
        finally:
            self._observation_snapshot = None

    def _observations(self):
        c = self.cfg
        if self.visibility_time != self.time:
            self.visibility = np.stack([candidate_nodes(c, g, self.time+self.phase_s) for g in range(self.g)])
            self.visibility_time = self.time
        obs = np.zeros((self.u+self.g, OBS_DIM), np.float32)
        masks = np.ones((self.u, 3), bool)
        active_users = []
        pools = np.zeros(self.g)
        for user, pending in enumerate(self.pending):
            if pending:
                pools[self.region[user]] += 1
        node_work = np.array([self.work(self.u+n)/self.speed(self.u+n) for n in range(N_ROUTE)])
        committed = self.committed_remote_work()
        occupancy = np.asarray([(self.work(self.u+n)+committed[n]) /
                                (c["queue_cap_s"][self.node_kind(self.u+n)]*self.speed(self.u+n))
                                for n in range(N_ROUTE)])
        self.queue_pressure = np.asarray([float(occupancy[mask].mean()) if mask.any() else 1.
                                          for mask in self.visibility])
        self.high_tide = np.where(self.high_tide, self.queue_pressure > c.get("low_queue_fraction", .3),
                                  self.queue_pressure >= c.get("high_queue_fraction", .6))
        rate_scale = 100*c["arrival_per_user_s"]
        predicted = self.previous_forecast
        pred_error = np.mean(self.pred_error, axis=0) if self.pred_error else np.zeros(self.g)
        public_cost = {}
        _, by_user = self.link_counts()
        for user, pending in enumerate(self.pending):
            g = self.region[user]
            own_work = self.work(user)/c["cpu_ue"]
            obs[user, :12] = [len(pending)/20, own_work/.5, pools[g]/c["users_per_region"],
                              self.last_observed[g]/rate_scale, predicted[g]/rate_scale, pred_error[g]/rate_scale,
                              g/2, node_work[:self.b].mean()/.1, node_work[:self.b].max()/.1,
                              node_work[self.b:].mean()/.1, 0, 0]
            if pending:
                j = self.tasks[pending[0]]
                if j.hol_time < 0:
                    j.hol_time = self.time
                age = self.time-j.arrival
                obs[user, 12:17] = [age/.5, j.active_wait/.1, (j.deadline-age)/.5, j.bits/4e5, j.cycles/2.4e8]
                local_time = own_work+j.cycles/c["cpu_ue"]
                local_energy = j.cycles*c["energy_per_cycle"][0]
                # Task sizes are observable; the user's true XY is not used here.
                key = (g, j.bits, j.cycles, by_user[user])
                if key not in public_cost:
                    public_cost[key] = self.route_features(j, c["region_centers_m"][g])[:2]
                pub, feasible = public_cost[key]
                remote_time = pub[:11]*c["time_ref_s"]+c["dt_s"]
                remote_energy = pub[11:22]*c["energy_ref_j"]+c["report_energy_j"]
                costs = c["weight_time"]*(age+remote_time)/c["time_ref_s"]+c["weight_energy"]*remote_energy/c["energy_ref_j"]+c["weight_violation"]*(age+remote_time>j.deadline)
                obs[user,17] = c["weight_time"]*(age+local_time)/c["time_ref_s"]+c["weight_energy"]*local_energy/c["energy_ref_j"]+c["weight_violation"]*(age+local_time>j.deadline)
                obs[user,18] = np.min(costs[feasible]) if feasible.any() else 100
                obs[user,19:21] = [self.queue_pressure[g], float(self.high_tide[g])]
                # Waiting is a regional gate decision, never an independent user action.
                masks[user, 0] = False
                masks[user, 1] = self.work(user)+j.cycles <= c["queue_cap_s"][0]*c["cpu_ue"]+1e-5
                masks[user, 2] = bool(self.visibility[g].any())
                if not masks[user].any():
                    masks[user, 1] = True  # Admission will explicitly record failure.
                if user not in self.staged_users:
                    active_users.append(user)
        ready = [j for j in self.tasks.values() if j.status == "report" and j.report_ready <= self.time+1e-9]
        for g in range(self.g):
            reports = [j for j in ready if j.region == g]
            if reports:
                obs[self.u+g] = np.mean([self.route_features(j)[0] for j in reports], axis=0)
            obs[self.u+g, 51:54] = [self.queue_pressure[g], float(self.high_tide[g]), 0]
            obs[self.u+g, 54:59] = [len(reports)/100, pools[g]/100, self.last_observed[g]/rate_scale, predicted[g]/rate_scale, pred_error[g]/rate_scale]
            obs[self.u+g, 59:64] = self.batch_state(g)
        global_obs = np.zeros(OBS_DIM, np.float32)
        global_obs[:11] = node_work/.1
        by_node, _ = self.link_counts()
        global_obs[11:22] = [by_node[n]/50 for n in range(N_ROUTE)]
        global_obs[22:25] = pools/c["users_per_region"]
        global_obs[25:28] = self.last_observed/rate_scale
        global_obs[28:31] = predicted/rate_scale
        global_obs[31:34] = pred_error/rate_scale
        global_obs[34:37] = [len(self.tasks)/max(1,self.u), np.mean(obs[:self.u,1]), np.max(obs[:self.u,1])]
        global_obs[37:40] = [len(group)/c["users_per_region"] for group in self.staging]
        global_obs[40:43] = self.queue_pressure
        global_obs[43:46] = self.high_tide
        if self.method == "TMAPO-NoPred":
            obs[:self.u, 4:6] = 0
            obs[self.u:, 57:59] = 0
            global_obs[28:34] = 0
        return obs, global_obs, masks, active_users, ready

    def batch_state(self, region):
        c = self.cfg
        ids = self.staging[region]
        if not ids:
            return np.zeros(5, np.float32)
        oldest = min((self.tasks[jid] for jid in ids), key=lambda j: j.arrival)
        target = self.min_batch or c["users_per_region"]
        return np.asarray([
            len(ids)/c["users_per_region"],
            oldest.active_wait/max(c["wait_max_s"], c["dt_s"]),
            (oldest.deadline-(self.time-oldest.arrival))/oldest.deadline,
            (len(ids)-target)/c["users_per_region"],
            float(self.min_batch is not None),
        ], dtype=np.float32)

    def batch_features(self, region, base):
        obs = base.copy()
        obs[59:64] = self.batch_state(region)
        tasks = [self.tasks[jid] for jid in self.staging[region]]
        if not tasks:
            return obs, np.asarray([False, True], dtype=bool)
        c = self.cfg
        pending = sum(bool(q) and u not in self.staged_users for u, q in enumerate(self.pending)
                      if self.region[u] == region)
        current_rate = self.last_observed[region]/c["users_per_region"] if self.windows_seen else c["arrival_per_user_s"]
        fill_time = expected_batch_fill_s(c["users_per_region"], len(tasks), pending,
                                         self.min_batch or c["users_per_region"]+1, current_rate)
        oldest = min(tasks, key=lambda j: j.arrival)
        feature, feasible, _ = self.route_features(oldest, c["region_centers_m"][region])
        service = float(np.min(feature[:N_ROUTE][feasible])*c["time_ref_s"]) if feasible.any() else math.inf
        report_delay = max(c["dt_s"], math.ceil(c["privacy_processing_s"]/c["dt_s"]-1e-9)*c["dt_s"])
        remaining = min(min(c["wait_max_s"]-j.active_wait for j in tasks),
                        min(j.deadline-(self.time-j.arrival) for j in tasks)-service-report_delay)
        can_hold = (self.method != "TMAPO-NoWait" and self.min_batch is not None
                    and len(tasks) < self.min_batch and remaining >= c["dt_s"]-1e-9
                    and max(c["dt_s"], fill_time) <= remaining+1e-9)
        wait_scale = max(c["wait_max_s"], c["dt_s"])
        obs[51:54] = [min(fill_time/wait_scale, 100), remaining/wait_scale, self.queue_pressure[region]]
        # A below-target batch still satisfies epsilon_req using its own accountant.
        return obs, np.asarray([can_hold, True], dtype=bool)

    def energy(self, j, amount, component):
        j.energy += amount
        setattr(j, component+"_energy", getattr(j, component+"_energy")+amount)
        charge = self.cfg["weight_energy"]*amount/self.cfg["energy_ref_j"]
        self.step_cost += charge
        self.cost_total += charge
        self.user_step_cost[j.user] += charge

    def violate(self, j):
        if not j.violated:
            j.violated = True
            self.step_cost += self.cfg["weight_violation"]
            self.cost_total += self.cfg["weight_violation"]
            self.user_step_cost[j.user] += self.cfg["weight_violation"]

    def end(self, j, failed=False):
        c = self.cfg
        if failed or self.time-j.arrival > j.deadline+1e-9:
            self.violate(j)
        j.end, j.failed = self.time, failed
        if j.status == "pending":
            self.pending[j.user].remove(j.id)
        if j.status == "staged":
            self.staging[j.region].remove(j.id)
            self.staged_users.remove(j.user)
        if j.status == "compute":
            self.compute[j.node].remove(j.id)
        self.ended += 1
        if self.measured(j):
            self.measure_ended += 1
            self.measure_failed += int(failed)
            self.measure_completed += int(not failed)
            self.measure_qos += int(j.violated)
            self.measure_late_completed += int(not failed and j.violated)
            self.measure_expired += int(failed and self.time >= j.arrival+c["lifetime_s"]-1e-9)
            self.sum_delay += (j.end-j.arrival) * int(not failed)
            self.sum_energy += j.energy
            self.sum_resolution_time += j.end-j.arrival
            self.sum_transmit_energy += j.transmit_energy
            self.sum_compute_energy += j.compute_energy
            self.sum_report_energy += j.report_energy
            self.sum_wait += j.active_wait
            self.sum_cost += c["weight_time"]*(j.end-j.arrival)/c["time_ref_s"]+c["weight_energy"]*j.energy/c["energy_ref_j"]+c["weight_violation"]*j.violated
            if self.collect:
                self.rows.append({"task_id":j.id,"user":j.user,"region":j.region,"arrival_s":j.arrival,
                    "task_profile":j.profile,"task_bits":j.bits,"task_cycles":j.cycles,"deadline_s":j.deadline,
                    "arrival_tide_factor":j.arrival_tide_factor,
                    "end_s":j.end,"completed":int(not failed),"qos":int(j.violated),"energy_j":j.energy,
                    "transmit_energy_j":j.transmit_energy,"compute_energy_j":j.compute_energy,"report_energy_j":j.report_energy,
                    "active_wait_s":j.active_wait,"pre_hol_s":max(0,j.hol_time-j.arrival),
                    "published":int(j.report is not None),"ql_m":j.ql if j.report is not None else "",
                    "epsilon_loc_per_m":j.epsilon,"batch_n":j.batch_n,"release_s":j.release,
                    "choice_s":j.node_choice_time,"compute_queue_s":max(0,j.compute_start-j.compute_ready),"node":j.node})
        del self.tasks[j.id]

    def enqueue_compute(self, j, node):
        c = self.cfg
        capacity = c["queue_cap_s"][self.node_kind(node)]*self.speed(node)
        if self.work(node)+j.cycles > capacity+1e-5:
            self.end(j, failed=True)
            return
        j.status, j.node, j.remaining = "compute", node, j.cycles
        j.compute_ready = self.time
        if not self.compute[node]:
            j.compute_start = self.time
        self.compute[node].append(j.id)

    def release(self, tasks, reason="direct"):
        for g in range(self.g):
            group = [j for j in tasks if j.region == g]
            if not group:
                continue
            assert len({j.user for j in group}) == len(group)
            n = len(group)
            eps = local_epsilon(self.cfg, n, self.method == "TMAPO-NoAmp")
            for j in group:
                # Counter-keyed random numbers give paired noise across policies.
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, 3, j.id]))
                radius, angle = rng.gamma(2), rng.uniform(0,2*np.pi)
                noise = radius/eps*np.array([np.cos(angle),np.sin(angle)])
                j.report = self.xy[j.user]+noise
                j.ql, j.epsilon, j.batch_n = float(np.linalg.norm(noise)), eps, n
                j.status, j.release = "report", self.time
                j.report_ready = self.time+max(self.cfg["dt_s"], math.ceil(self.cfg["privacy_processing_s"]/self.cfg["dt_s"]-1e-9)*self.cfg["dt_s"])
                self.energy(j, self.cfg["report_energy_j"], "report")
                self.noise_publications[j.user] += 1
                if self.measured(j):
                    self.measure_published += 1
                    self.measure_amplified += int(eps > self.cfg["epsilon_req"]/(2*self.cfg["region_radius_m"]) + 1e-12)
                    self.sum_ql += j.ql
                    self.sum_ql_theory += 2/eps
                    self.sum_k += n
            if self.collect:
                self.batch_rows.append({"time_s":self.time,"region":g,"n_distinct":n,
                    "queue_pressure":float(self.queue_pressure[g]),"high_queue":int(self.high_tide[g]),
                    "epsilon_loc_per_m":eps,"epsilon0":eps*2*self.cfg["region_radius_m"],"release_reason":reason,
                    "task_ids":";".join(str(j.id) for j in group)})

    def assign(self, j, node):
        if not self.visibility[j.region, node]:
            raise ValueError("Selected a masked node")
        j.status, j.node, j.remaining = "upload", int(node), j.bits
        j.node_choice_time = self.time

    def step(self, policy, arrivals=True):
        self.step_cost = 0.
        self.user_step_cost[:] = 0.
        if arrivals:
            self.generate()
        context, glob, masks, users, ready = self.observations()
        if self.measure_start_slot <= self.slot and (self.measure_end_slot is None or self.slot < self.measure_end_slot):
            self.pressure_samples += 1
            self.sum_queue_pressure += self.queue_pressure
            self.sum_high_tide += self.high_tide
        policy.begin(context, glob, self.u)
        records = []
        if ready:
            # Order is public randomized mixing, never sorted by source user.
            mixing = np.random.default_rng(np.random.SeedSequence([self.seed,4,self.slot]))
            order = mixing.permutation(len(ready))
            ready = [ready[i] for i in order]
            zero_reservations = (np.zeros(N_ROUTE, dtype=int), np.zeros(N_ROUTE))
            for j in ready:
                # Earlier assignments in this loop are already in the upload set.
                feature, feasible, _ = self.route_features(j, reservations=zero_reservations)
                acts, rec = policy.act(feature[None, :], feasible[None, :], "route",
                                       np.array([self.u+j.region]))
                records.extend(rec)
                node = int(acts[0])
                self.assign(j,node)
        if users:
            acts, rec = policy.act(context[users],masks[users],"user",np.asarray(users))
            records.extend(rec)
            for u,a in zip(users,acts):
                j = self.tasks[self.pending[u][0]]
                if a == 0:
                    raise ValueError("User-level waiting is masked; regional gate owns waiting")
                else:
                    self.pending[u].popleft()
                    j.status = "selected"
                    if a == 1:
                        self.enqueue_compute(j,u)
                    else:
                        j.status = "staged"
                        self.staging[j.region].append(j.id)
                        self.staged_users.add(u)
        regions = [g for g in range(self.g) if self.staging[g]]
        if regions:
            packed = [self.batch_features(g,context[self.u+g]) for g in regions]
            acts, rec = policy.act(np.stack([x[0] for x in packed]),np.stack([x[1] for x in packed]),
                                   "batch",np.asarray([self.u+g for g in regions]))
            records.extend(rec)
            for g,a in zip(regions,acts):
                if a == 0:
                    for jid in self.staging[g]:
                        self.tasks[jid].active_wait += self.cfg["dt_s"]
                else:
                    ids = list(self.staging[g])
                    group = [self.tasks[jid] for jid in ids]
                    self.staging[g].clear()
                    self.staged_users.difference_update(j.user for j in group)
                    reason = "target_met" if self.min_batch and len(group) >= self.min_batch else (
                        "no_gain_possible" if self.min_batch is None else (
                            "no_wait_ablation" if self.method == "TMAPO-NoWait" else (
                                "policy_release" if packed[regions.index(g)][1][0] else "insufficient_wait_budget")))
                    self.release(group,reason)
        self.advance((self.slot+1)*self.cfg["dt_s"])
        self.slot += 1
        scale = self.u*self.cfg["arrival_per_user_s"]*self.cfg["dt_s"]
        return -self.step_cost/scale, records, context, glob

    def agent_rewards(self, local_fraction):
        """Conserved task-cost attribution with a common team component.

        UE controls receive their own tasks' accrued cost; regional controls
        receive regional cost. The mean of each group equals the team reward.
        No privacy proxy or favorable-method bonus is added to the objective.
        """
        scale = self.cfg["arrival_per_user_s"]*self.cfg["dt_s"]
        team = -self.step_cost/(self.u*scale)
        user = -self.user_step_cost/scale
        regional = user.reshape(self.g, self.cfg["users_per_region"]).mean(axis=1)
        return ((1-local_fraction)*team+local_fraction*np.r_[user, regional]).astype(np.float32)

    def advance(self, end_time):
        c = self.cfg
        while self.time < end_time-1e-11:
            # Resolve deterministic events before selecting the next interval.
            changed = True
            while changed:
                changed = False
                for j in list(self.tasks.values()):
                    if j.id not in self.tasks:
                        continue
                    if self.time >= j.arrival+c["lifetime_s"]-1e-10:
                        self.end(j,True); changed=True
                    elif j.status == "upload" and j.remaining <= 1e-4:
                        j.status = "propagation"
                        j.due = self.time+propagation(c,self.xy[j.user],j.node,self.time+self.phase_s)
                        changed=True
                    elif j.status == "propagation" and j.due <= self.time+1e-10:
                        self.enqueue_compute(j,self.u+j.node); changed=True
                    elif j.status == "compute" and j.remaining <= 1e-3:
                        self.compute[j.node].popleft()
                        j.status = "return"
                        j.due = self.time + (propagation(c,self.xy[j.user],j.node-self.u,self.time+self.phase_s) if j.node>=self.u else 0)
                        changed=True
                    elif j.status == "return" and j.due <= self.time+1e-10:
                        self.end(j); changed=True
                for node,q in enumerate(self.compute):
                    if q and self.tasks[q[0]].compute_start < 0:
                        self.tasks[q[0]].compute_start = self.time
            if not self.tasks:
                self.time = end_time
                break
            uploads = [j for j in self.tasks.values() if j.status == "upload"]
            by_node, by_user = Counter(j.node for j in uploads), Counter(j.user for j in uploads)
            rates = {j.id:rate(c,self.xy[j.user],j.node,self.time+self.phase_s,by_node[j.node],by_user[j.user]) for j in uploads}
            finish_times = [end_time]
            finish_times += [self.time+j.remaining/rates[j.id] for j in uploads]
            heads = [self.tasks[q[0]] for q in self.compute if q]
            finish_times += [self.time+j.remaining/self.speed(j.node) for j in heads]
            finish_times += [j.due for j in self.tasks.values() if j.status in ("propagation","return")]
            finish_times += [j.arrival+c["lifetime_s"] for j in self.tasks.values()]
            next_time = min(t for t in finish_times if t > self.time+1e-11)
            dt = next_time-self.time
            charge = len(self.tasks)*dt*c["weight_time"]/c["time_ref_s"]
            self.step_cost += charge
            self.cost_total += charge
            for j in self.tasks.values():
                self.user_step_cost[j.user] += dt*c["weight_time"]/c["time_ref_s"]
            for j in uploads:
                j.remaining = max(0,j.remaining-rates[j.id]*dt)
                self.energy(j,c["user_power_w"]*dt/by_user[j.user],"transmit")
            for j in heads:
                cycles = min(j.remaining,self.speed(j.node)*dt)
                j.remaining = max(0,j.remaining-cycles)
                self.energy(j,cycles*c["energy_per_cycle"][self.node_kind(j.node)],"compute")
                if j.node >= self.u:
                    overlap = max(0., min(next_time, self.measure_end)-max(self.time, self.measure_start))
                    self.sum_compute_busy[j.node-self.u] += overlap
            self.time = next_time
            for j in self.tasks.values():
                if self.time-j.arrival > j.deadline+1e-9:
                    self.violate(j)

    def metrics(self):
        if self.measure_ended != self.measure_arrivals:
            raise RuntimeError("Measured tasks must drain before reporting")
        n, p = self.measure_arrivals, self.measure_published
        if n == 0:
            raise ValueError("No measured arrivals")
        return {"arrivals":n,"completed":self.measure_completed,"failed":self.measure_failed,
            "delay_s":self.sum_delay/self.measure_completed if self.measure_completed else None,
            "energy_j_per_arrival":self.sum_energy/n,"qos_rate":self.measure_qos/n,
            "completion_rate":self.measure_completed/n,"mean_task_cost":self.sum_cost/n,
            "late_completed_rate":self.measure_late_completed/n,"expired_rate":self.measure_expired/n,
            "rejected_rate":(self.measure_failed-self.measure_expired)/n,
            "resolution_time_s_per_arrival":self.sum_resolution_time/n,
            "transmit_energy_j_per_arrival":self.sum_transmit_energy/n,
            "compute_energy_j_per_arrival":self.sum_compute_energy/n,
            "report_energy_j_per_arrival":self.sum_report_energy/n,
            "mean_queue_pressure":float(self.sum_queue_pressure.mean()/max(self.pressure_samples,1)),
            "high_queue_region_time_fraction":float(self.sum_high_tide.mean()/max(self.pressure_samples,1)),
            "remote_compute_utilization":float(self.sum_compute_busy.mean()/max(self.pressure_samples*self.cfg["dt_s"],1e-9)),
            **{f"region_{g}_high_queue_fraction":float(self.sum_high_tide[g]/max(self.pressure_samples,1)) for g in range(self.g)},
            "ql_m":self.sum_ql/p if p else None,"ql_theory_m":self.sum_ql_theory/p if p else None,
            "published":p,"publication_ratio":p/n,"report_weighted_batch_n":self.sum_k/p if p else None,
            "amplified_publication_ratio":self.measure_amplified/p if p else None,
            "amplification_threshold":self.min_batch,
            "active_wait_s":self.sum_wait/n,"max_user_publications":int(self.noise_publications.max()),
            "trace_hash":self.arrival_hash}


class GreedyPolicy:
    def __init__(self,cfg):
        self.cfg = cfg

    def begin(self,context,glob,n_user):
        self.context, self.n_user = context,n_user

    def act(self,obs,mask,stage,agents):
        c = self.cfg
        if stage == "batch":
            return np.where(mask[:,1],1,0),[]
        if stage == "route":
            cost = c["weight_time"]*obs[:,:11]+c["weight_energy"]*obs[:,11:22]
            cost += c["weight_violation"]*(obs[:,:11] > obs[:,45:46])
            return np.argmin(np.where(mask,cost,np.inf),axis=1),[]
        costs = np.stack([np.full(len(obs),np.inf),obs[:,17],obs[:,18]],axis=1)
        return np.argmin(np.where(mask,costs,np.inf),axis=1),[]


class LocalOnlyPolicy:
    """Execute every admitted task on its own UE; never publish a location."""

    def begin(self,context,glob,n_user):
        pass

    def act(self,obs,mask,stage,agents):
        if stage != "user":
            raise RuntimeError("Local-Only must not enter batch release or remote routing")
        return np.ones(len(obs),dtype=int),[]

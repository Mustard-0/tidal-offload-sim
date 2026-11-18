"""Independent baseline control pipeline over the frozen physical simulator."""
import hashlib
import json
import math


def environment_types():
    import numpy as np
    from environment import Environment as PhysicalEnvironment, GreedyPolicy, LocalOnlyPolicy, N_ROUTE
    from pressure9.environment import Environment as OriginalTMAPO

    class PreparedInput:
        def __init__(self, *args, **kwargs):
            self.prepared_slot = None
            self.full_input_hash = hashlib.sha256(b'').hexdigest()
            super().__init__(*args, **kwargs)

        def generate(self):
            self.time = self.slot * self.cfg['dt_s']
            if self.prepared_slot == self.slot:
                return
            first = self.serial
            super().generate()
            h = hashlib.sha256(bytes.fromhex(self.full_input_hash))
            for task_id in range(first, self.serial):
                j = self.tasks[task_id]
                row = dict(task_id=j.id, user=j.user, region=j.region, arrival_slot=j.arrival_slot,
                    arrival_s=j.arrival, task_bits=j.bits, task_cycles=j.cycles, deadline_s=j.deadline,
                    task_profile=j.profile, arrival_tide_factor=j.arrival_tide_factor,
                    truth_x_m=float(self.xy[j.user, 0]), truth_y_m=float(self.xy[j.user, 1]))
                h.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode() + b'\n')
            self.full_input_hash = h.hexdigest()
            self.prepared_slot = self.slot

        def prepare(self, arrivals):
            self.time = self.slot * self.cfg['dt_s']
            if arrivals:
                self.generate()
            return self.observations()[:2]

    class PlainEnvironment(PreparedInput, PhysicalEnvironment):
        """No TMAPO environment, batch gate, accountant, or credit inheritance."""
        def __init__(self, cfg, *, family, **kwargs):
            self.family = family
            self.reporter = cfg['mechanism']
            if self.reporter not in ('none', 'ordinary_planar_geoi_individual'):
                raise ValueError('Baseline cannot use TMAPO privacy mechanism')
            self.report_count = 0
            super().__init__(cfg, method=family, **kwargs)
            self.min_batch = None

        def route_features(self, *args, **kwargs):
            obs, mask, cost = super().route_features(*args, **kwargs)
            obs[57:62] = 0.
            return obs, mask, cost

        def _observations(self):
            context, glob, masks, users, ready = super()._observations()
            context[:self.u, 4:6] = 0.
            context[self.u:, 57:59] = 0.
            context[self.u:, 59:64] = 0.
            glob[28:34] = 0.
            glob[37:40] = 0.
            masks[:, 0] = False
            if self.family not in ('Greedy-NoPrivacy', 'Greedy-GeoI', 'Local-Only'):
                context[:self.u, 17:19] = 0.
                context[:self.u, 26] = 0.
            if self.family == 'Local-Only':
                # LOCAL means attempt admission. Full queues still reject and
                # charge a failure through the unchanged enqueue_compute path.
                masks[:, :] = False
                masks[:, 1] = True
            return context, glob, masks, users, ready

        def release(self, tasks, reason='individual_immediate'):
            for j in tasks:
                truth = self.xy[j.user]
                if self.reporter == 'ordinary_planar_geoi_individual':
                    beta = self.cfg['epsilon_req'] / (2 * self.cfg['region_radius_m'])
                    rng = np.random.default_rng(np.random.SeedSequence([self.seed, 3, j.id]))
                    radius, angle = rng.gamma(2., 1. / beta), rng.uniform(0., 2 * np.pi)
                    j.report = truth + radius * np.array([np.cos(angle), np.sin(angle)])
                    self.energy(j, self.cfg['report_energy_j'], 'report')
                    overhead = max(self.cfg['dt_s'], math.ceil(self.cfg['privacy_processing_s'] / self.cfg['dt_s'] - 1e-9) * self.cfg['dt_s'])
                    self.noise_publications[j.user] += 1
                    j.epsilon = beta
                else:
                    j.report, overhead, j.epsilon = truth.copy(), 0., 0.
                j.ql, j.batch_n = float(np.linalg.norm(j.report - truth)), 1
                j.status, j.release, j.report_ready = 'report', self.time, self.time + overhead
                self.report_count += 1
                if self.measured(j):
                    self.measure_published += 1
                    self.sum_ql += j.ql
                    self.sum_ql_theory += 4 * self.cfg['region_radius_m'] / self.cfg['epsilon_req'] if j.epsilon else 0.
                    self.sum_k += 1

        def agent_rewards(self, local_fraction):
            if local_fraction != 0:
                raise ValueError('No TMAPO local credit in a baseline')
            return super().agent_rewards(0.)

        def step(self, policy, arrivals=True):
            c = self.cfg
            self.time = self.slot * c['dt_s']
            self.step_cost, self.user_step_cost[:] = 0., 0.
            if arrivals:
                self.generate()
            context, glob, masks, users, ready = self.observations()
            policy.begin(context, glob, self.u)
            records = []

            def choose(obs, mask, stage, agents):
                actions, rows = policy.act(obs, mask, stage, agents)
                a = np.asarray(actions)
                if (a.shape != (len(obs),) or not np.isfinite(a).all() or
                    np.any(a != a.astype(int)) or np.any(a < 0) or np.any(a >= mask.shape[1]) or
                    not mask[np.arange(len(a)), a.astype(int)].all()):
                    raise AssertionError('Illegal baseline action')
                records.extend(rows)
                return a.astype(int)

            if ready:
                order = np.random.default_rng(np.random.SeedSequence([self.seed, 4, self.slot])).permutation(len(ready))
                reservations = (np.zeros(N_ROUTE, dtype=int), np.zeros(N_ROUTE))
                for ix in order:
                    j = ready[int(ix)]
                    obs, mask, _ = self.route_features(j, reservations=reservations)
                    a = choose(obs[None, :], mask[None, :], 'route', np.asarray([self.u + j.region]))
                    self.assign(j, int(a[0]))
            if users:
                actions = choose(context[users], masks[users], 'user', np.asarray(users))
                for user, a in zip(users, actions):
                    j = self.tasks[self.pending[user].popleft()]
                    j.status = 'selected'
                    if a == 1:
                        self.enqueue_compute(j, int(user))
                    elif a == 2:
                        self.release([j])
                    else:
                        raise AssertionError('Plain baseline cannot wait')
            self.advance((self.slot + 1) * c['dt_s'])
            self.slot += 1
            return -self.step_cost / (self.u * c['arrival_per_user_s'] * c['dt_s']), records, context, glob

        def metrics(self):
            result = super().metrics()
            result.update(number_queued_pending=len(self.tasks), privacy_mechanism=self.reporter,
                class_staging_left=0, class_gate_wait_actions=0, gate_allowed=0,
                maximum_distinct_class_batch=1 if self.report_count else 0,
                baseline_pipeline='individual release; no TMAPO inherited gate')
            return result

    class AblationEnvironment(OriginalTMAPO):
        def __init__(self, *args, ablation=None, **kwargs):
            self.ablation = ablation
            super().__init__(*args, family='TMAPO', **kwargs)

        def wait_information(self, tasks):
            info = super().wait_information(tasks)
            if self.ablation == 'NoWait':
                info.update(allow=False, reason='registered_no_wait_ablation')
            return info

        def _observations(self):
            context, glob, masks, users, ready = super()._observations()
            if self.ablation == 'NoForecastFeatures':
                context[:self.u, 4:6] = 0.
                context[self.u:, 57:59] = 0.
                glob[28:34] = 0.
            return context, glob, masks, users, ready

        def route_features(self, *args, **kwargs):
            obs, mask, cost = super().route_features(*args, **kwargs)
            if self.ablation == 'NoForecastFeatures':
                obs[57:62] = 0.
            return obs, mask, cost

    return PlainEnvironment, AblationEnvironment, GreedyPolicy, LocalOnlyPolicy

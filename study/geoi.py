"""Ordinary Geo-I ablation. Physics/queues/actions remain those of frozen source."""
import math
from experiments.lr_screen.runtime import np
from .environment import Environment
from population import user_region


class GeoIEnvironment(Environment):
    def eps0(self, n):
        if int(n) != n or not 1 <= n <= max(self.region_counts):
            raise ValueError('Invalid actual distinct-user count')
        return self.cfg['epsilon_req']

    def q_estimate(self, n):
        return 4 * self.cfg['region_radius_m'] / self.eps0(n)

    def release(self, tasks, reason='adaptive_region_class'):
        if not tasks:
            return
        c = self.cfg
        n = len(tasks)
        region = tasks[0].region
        if (len({(j.region, j.profile) for j in tasks}) != 1
                or len({j.user for j in tasks}) != n
                or len({j.id for j in tasks}) != n
                or n > self.region_counts[region]
                or any(user_region(c, j.user) != region for j in tasks)):
            raise ValueError('Invalid real public region/class batch')
        eps0 = self.eps0(n)
        beta = eps0 / (2 * c['region_radius_m'])
        self.max_class_group = max(self.max_class_group, n)
        self.release_serial += 1
        for j in tasks:
            if j.id not in getattr(self, '_staged_at_slot_start', set()):
                if not hasattr(self, '_slot_admissions'):
                    self._slot_admissions = []
                self._slot_admissions.append((j.region, j.profile, j.user))
            truth = self.xy[j.user]
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 3, j.id]))
            radius = rng.gamma(2., 1 / beta)
            angle = rng.uniform(0., 2 * np.pi)
            j.report = truth + radius * np.array([np.cos(angle), np.sin(angle)])
            j.ql, j.epsilon, j.batch_n = float(np.linalg.norm(j.report-truth)), beta, n
            j.status, j.release = 'report', self.time
            j.report_ready = self.time + max(c['dt_s'], math.ceil(c['privacy_processing_s']/c['dt_s']-1e-9)*c['dt_s'])
            self.energy(j, c['report_energy_j'], 'report')
            self.noise_publications[j.user] += 1
            self.sum_class_group_wait += j.active_wait
            self.max_observed_wait = max(self.max_observed_wait, j.active_wait)
            if j.active_wait > c['wait_max_s']+1e-9 or c['weight_quality'] != 0:
                raise AssertionError('Waiting or common reward contract changed')
            if self.measured(j):
                self.measure_published += 1
                self.sum_ql += j.ql
                self.sum_ql_theory += self.q_estimate(n)
                self.sum_k += n
        if self.collect:
            self.batch_rows.append(dict(batch_id=self.release_serial, time_s=self.time,
                region=region, task_class=tasks[0].profile, n_distinct=n,
                epsilon0=eps0, beta_per_m=beta, delta_bound=0., shrink=1.,
                privacy_basis='epsilon-local-DP on diameter2R; no shuffle amplification claimed',
                release_reason=reason, task_ids=';'.join(str(j.id) for j in tasks),
                user_ids=';'.join(str(j.user) for j in tasks)))


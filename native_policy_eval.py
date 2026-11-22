"""Native policy actions with read-only action and outcome instrumentation."""

from collections import Counter
from math import ceil

import numpy as np


def report_quote(cfg):
    dt = cfg['dt_s']
    if cfg['mechanism'] == 'none':
        # A new report cannot enter the already processed routing phase.
        return dt, 0.
    assert cfg['mechanism'] == 'ordinary_planar_geoi_individual'
    return max(dt, ceil(cfg['privacy_processing_s']/dt - 1e-9)*dt), cfg['report_energy_j']


def weighted_cost(delay, energy, deadline, cfg):
    return (cfg['weight_time'] * (delay / cfg['time_ref_s'])
            + cfg['weight_energy'] * (energy / cfg['energy_ref_j'])
            + cfg['weight_violation'] * (delay > deadline))


def with_greedy_quotes(base):
    class GreedyQuotes(base):
        def route_features(self, task, xy=None, reservations=None):
            obs, mask, cost = super().route_features(task, xy=xy, reservations=reservations)
            if xy is None and reservations is not None:
                # Uses the actual published coordinates, including for GeoI.
                self._route_quote = np.asarray(cost).copy()
            return obs, mask, cost

        def _observations(self):
            context, glob, masks, users, ready = super()._observations()
            cfg, snapshot = self.cfg, self._observation_snapshot
            dt, report_energy = report_quote(cfg)
            _, by_user, committed, queued, _ = snapshot
            cpu = np.asarray([self.speed(self.u + n) for n in range(self.b + self.s)])
            cap = cpu * np.asarray([cfg['queue_cap_s'][self.node_kind(self.u+n)] for n in range(len(cpu))])
            work = np.asarray(queued) + committed
            cached = {}
            for i in users:
                task = self.tasks[self.pending[i][0]]
                key = task.region, task.bits, task.cycles, by_user[i]
                if key not in cached:
                    # The two Greedys use exactly the same information before release.
                    public, visible, _ = self.route_features(task, xy=cfg['region_centers_m'][task.region])
                    fits = visible & (work + task.cycles <= cap + 1e-5)
                    n = len(visible)
                    cached[key] = public[:n] * cfg['time_ref_s'], public[n:2*n] * cfg['energy_ref_j'], fits
                delay, energy, fits = cached[key]
                costs = weighted_cost((self.time-task.arrival)+dt+delay, report_energy+energy, task.deadline, cfg)
                context[i, 18] = np.min(costs[fits]) if fits.any() else 100.
            return context, glob, masks, users, ready
    return GreedyQuotes


class QuotedGreedy:
    def __init__(self, env):
        self.env = env

    def begin(self, context, glob, n_user):
        pass

    def act(self, obs, mask, stage, agents):
        if stage == 'route':
            cost = self.env._route_quote
            assert len(obs) == 1 and cost.shape == mask[0].shape
            self.env._route_quote = None
            return np.argmin(np.where(mask, cost[None, :], np.inf), axis=1), []
        assert stage == 'user'
        cost = np.column_stack((np.full(len(obs), np.inf), obs[:, 17], obs[:, 18]))
        return np.argmin(np.where(mask, cost, np.inf), axis=1), []


class ActionMeter:
    def __init__(self, controller, env):
        self.controller = controller
        self.env = env
        self.actions = Counter()
        self.options = Counter()

    def begin(self, context, glob, n_user):
        return self.controller.begin(context, glob, n_user)

    def act(self, obs, mask, stage, agents):
        result = self.controller.act(obs, mask, stage, agents)
        if self.env.measure_start <= self.env.time < self.env.measure_end:
            selected = np.asarray(result[0], dtype=np.int64)
            assert selected.shape == (len(obs),) and mask[np.arange(len(obs)), selected].all()
            self.actions.update((stage, int(i)) for i in selected)
            self.options.update((stage, int(i)) for i in np.count_nonzero(mask, axis=1))
        return result

    def metrics(self):
        def rows(counts):
            return [[stage, index, count] for (stage, index), count in sorted(counts.items())]
        return dict(native_action_counts=rows(self.actions), native_legal_counts=rows(self.options))

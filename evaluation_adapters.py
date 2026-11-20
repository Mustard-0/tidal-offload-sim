"""Opt-in evaluation adapters. Do not use to overwrite a frozen run."""

from functools import cached_property


GREEDY_PAIR = ("Greedy-NoPrivacy", "Greedy-GeoI")
STRICT_DEADLINE_RATIO = 0.95
DEADLINE_TOLERANCE_S = 1e-9


def capacity_aware_greedy(base):
    """Screen the public offload quote, leaving actual routing and physics unchanged."""
    import numpy as np

    class CapacityAwareGreedy(base):
        @cached_property
        def remote_cycle_caps(self):
            return np.asarray([
                self.speed(node) * self.cfg["queue_cap_s"][self.node_kind(node)]
                for node in range(self.u, self.u + self.b + self.s)
            ])

        def route_features(self, task, xy=None, reservations=None):
            obs, visible, cost = super().route_features(task, xy=xy, reservations=reservations)
            if self.family not in GREEDY_PAIR or xy is None or reservations is not None:
                return obs, visible, cost
            snapshot = getattr(self, "_observation_snapshot", None)
            if snapshot is None:
                committed = self.committed_remote_work()
                queued = [self.work(node) for node in range(self.u, self.u + len(visible))]
            else:
                # The parent uses this same snapshot for the public quotes.
                committed, queued = snapshot[2:4]
            fits = np.asarray(queued) + committed + task.cycles <= self.remote_cycle_caps + 1e-5
            feasible = visible & fits
            obs[3 * len(visible):4 * len(visible)] = feasible
            return obs, feasible, cost

    return CapacityAwareGreedy


def with_strict_qos(base):
    """Count a reporting-only deadline without changing actions, reward, or drain."""
    class StrictQoSEvaluation(base):
        def __init__(self, *args, **kwargs):
            self._strict_violations = 0
            super().__init__(*args, **kwargs)

        def end(self, task, failed=False):
            strict = self.measured(task) and (
                failed or self.time - task.arrival >
                STRICT_DEADLINE_RATIO * task.deadline + DEADLINE_TOLERANCE_S
            )
            outcome = super().end(task, failed=failed)
            self._strict_violations += int(strict)
            return outcome

        def metrics(self):
            result = super().metrics()
            rate = self._strict_violations / result["arrivals"]
            if not result["qos_rate"] <= rate <= 1:
                raise ValueError("Strict QoS must include all original violations")
            return dict(result, qos_deadline_ratio=STRICT_DEADLINE_RATIO,
                        qos_95_count=self._strict_violations, qos_95_rate=rate)

    return StrictQoSEvaluation


def evaluation_environment_types(*, optimize_greedy=False):
    """Evaluation-only replacement for review_environment.environment_types."""
    from review_environment import environment_types

    plain, tmapo, greedy, local = environment_types()
    if optimize_greedy:
        plain = capacity_aware_greedy(plain)
    return with_strict_qos(plain), with_strict_qos(tmapo), greedy, local

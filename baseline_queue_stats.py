"""Restore measurements omitted by the isolated baseline step implementation."""

from math import isfinite


def with_queue_sampling(base):
    class Sampled(base):
        _sample_on_observation = False

        def step(self, policy, arrivals=True):
            self._sample_on_observation = True
            try:
                result = super().step(policy, arrivals=arrivals)
                assert not self._sample_on_observation
                return result
            finally:
                self._sample_on_observation = False

        def _observations(self):
            result = super()._observations()
            if self._sample_on_observation:
                self._sample_on_observation = False
                if self.measure_start_slot <= self.slot and (
                    self.measure_end_slot is None or self.slot < self.measure_end_slot
                ):
                    self.pressure_samples += 1
                    self.sum_queue_pressure += self.queue_pressure
                    self.sum_high_tide += self.high_tide
            return result

        def metrics(self):
            end = self.slot if self.measure_end_slot is None else min(self.slot, self.measure_end_slot)
            assert self.pressure_samples == max(0, end - self.measure_start_slot) > 0
            result = super().metrics()
            u = result["remote_compute_utilization"]
            assert isfinite(u) and 0. <= u <= 1. + 1e-9
            return dict(result, queue_samples=self.pressure_samples)

    return Sampled

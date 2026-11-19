"""Opt-in telemetry correction; the frozen controller and physics are unchanged."""

from review_environment import environment_types


def diagnostic_environment_types():
    plain, tt, greedy, local = environment_types()

    class DiagnosticPlain(plain):
        def step(self, policy, arrivals=True):
            slot = self.slot
            result = super().step(policy, arrivals=arrivals)
            if self.measure_start_slot <= slot and (self.measure_end_slot is None or slot < self.measure_end_slot):
                # PlainEnvironment's queue state here is the pre-decision observation.
                self.pressure_samples += 1
                self.sum_queue_pressure += self.queue_pressure
                self.sum_high_tide += self.high_tide
            return result

    return plain, DiagnosticPlain, tt, greedy, local

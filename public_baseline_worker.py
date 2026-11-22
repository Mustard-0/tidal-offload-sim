"""Run the frozen archived simulator with the new learner updates only."""

import fresh_baseline_learners
import fresh_baseline_worker as frozen
import public_baseline_protocol as spec
from public_baseline_updates import make_controller


if __name__ == "__main__":
    frozen.spec = spec
    fresh_baseline_learners.controller = make_controller
    raise SystemExit(0 if frozen.main() else 2)

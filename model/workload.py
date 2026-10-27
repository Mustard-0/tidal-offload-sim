"""Common workload equations and analytical capacity checks, without training."""
from __future__ import annotations

import numpy as np


def tide_factors(cfg, time_s):
    phase = 2 * np.pi * time_s / cfg["tide_period_s"]
    common = cfg.get("global_tide_amplitude", 0.) * np.sin(phase + cfg.get("global_tide_phase", 0.))
    factors = 1 + common + np.asarray(cfg["tide_amplitudes"]) * np.sin(phase + np.asarray(cfg["tide_phases"]))
    if np.any(factors < 0):
        raise ValueError("Tidal arrival rates must stay nonnegative")
    return factors


def expected_batch_fill_s(population, staged, waiting_users, target, per_user_rate):
    """Expected distinct-user fill time under the current Poisson-rate estimate.

    Already pending unstaged users are available next slot. For the others,
    successive first arrivals have rates (available-k)*per_user_rate.
    This is an optimistic feasibility estimate: future offload choices may differ.
    """
    available = population - staged
    ready = min(waiting_users, available)
    missing = max(0, target - staged - ready)
    if missing == 0:
        return 0.
    if per_user_rate <= 0 or missing > available - ready:
        return float("inf")
    return float(sum(1. / ((available - ready - k) * per_user_rate) for k in range(missing)))


def validate_environment(cfg):
    for key in ("users_per_region", "dt_s", "task_bits", "task_cycles", "deadline_s",
                "lifetime_s", "arrival_per_user_s", "tide_period_s", "cpu_ue", "cpu_bs", "cpu_sat",
                "prediction_window_s", "prediction_horizon_s", "region_radius_m"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["lifetime_s"] < cfg["deadline_s"]:
        raise ValueError("Task lifetime must cover the QoS deadline")
    profiles = cfg.get("task_profiles", [])
    if profiles:
        if not np.isclose(sum(p["probability"] for p in profiles), 1.):
            raise ValueError("Task-profile probabilities must sum to one")
        for p in profiles:
            if any(p[k] <= 0 for k in ("probability", "bits", "cycles", "deadline_s")):
                raise ValueError("Task profiles require positive sizes, deadlines and probabilities")
            if p["deadline_s"] > cfg["lifetime_s"]:
                raise ValueError("Task-profile deadline exceeds lifetime")
    if cfg.get("global_tide_amplitude", 0.) + max(cfg["tide_amplitudes"]) > 1:
        raise ValueError("Combined tide amplitudes can produce negative arrival rates")
    if not 0 <= cfg.get("low_queue_fraction", .3) < cfg.get("high_queue_fraction", .6) <= 1:
        raise ValueError("Queue hysteresis must satisfy 0 <= low < high <= 1")


def analytical_load_rows(cfg, scales):
    """Offered work / nominal capacity; these are NOT measured utilization."""
    phase = np.linspace(0, cfg["tide_period_s"], 1200, endpoint=False)
    factors = np.stack([tide_factors(cfg, t) for t in phase])
    remote_capacity = (len(cfg["bs_xy_m"]) * cfg["cpu_bs"] +
                       len(cfg["sat_overhead_s"]) * cfg["cpu_sat"]) / cfg["task_cycles"]
    local_capacity = cfg["cpu_ue"] / cfg["task_cycles"]
    rows = []
    for users in scales:
        total = users * cfg["arrival_per_user_s"] * factors.sum(axis=1)
        rows.append(dict(users_per_region=users, total_users=users * len(cfg["region_centers_m"]),
                         arrivals_mean_per_s=float(total.mean()), arrivals_min_per_s=float(total.min()),
                         arrivals_peak_per_s=float(total.max()), all_remote_capacity_per_s=remote_capacity,
                         all_remote_offered_mean=float(total.mean() / remote_capacity),
                         all_remote_offered_peak=float(total.max() / remote_capacity),
                         ue_offered_mean=cfg["arrival_per_user_s"] / local_capacity,
                         ue_offered_peak=float(factors.max() * cfg["arrival_per_user_s"] / local_capacity)))
    return rows


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    validate_environment(config["environment"])
    print(json.dumps(analytical_load_rows(config["environment"], config["experiments"]["users_per_region"]), indent=2))

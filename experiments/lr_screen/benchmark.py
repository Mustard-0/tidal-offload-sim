"""Explicit bounded component timing, not an LR screen or convergence result."""
import argparse
import copy
import time
from .runtime import np, torch, ROOT, seed_all, json_write, csv_write, runtime_info, source_hashes, available_ram_gib
from .protocol import configuration
from .restricted_environment import CandidateRestrictedEnvironment
from .algorithms import learner_for
from .runner import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--out", type=str, required=True)
    a = p.parse_args()
    timings = []
    for method in ("TMAPO", "IPPO", "PPO"):
        c, t = configuration(method, 2)
        seed_all(t["seed"])
        learner = learner_for(method, t, c, a.device)
        env = CandidateRestrictedEnvironment(c, seed=20261025, phase_s=30.)
        for update in range(1, 4):
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(128):
                reward, records, _, _ = env.step(learner)
                learner.remember(env.agent_rewards(0.), records, False)
            ctx, glob, *_ = env.observations()
            stats = learner.update(ctx, glob)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append(dict(method=method, update=update, training_slots=128,
                wall_s=time.monotonic()-start, free_ram_gib=available_ram_gib(), **stats))
        # Actual requested validation window; measure one of the six cohorts.
        learner.deterministic = True
        start = time.monotonic()
        evaluate(learner, c, 814001, 30., 300, 1000)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        val_wall = time.monotonic()-start
        for row in timings:
            if row["method"] == method:
                row["one_validation_cohort_wall_s"] = val_wall
        print("TIMING", method, a.device, "train update seconds", [round(r["wall_s"], 3) for r in timings if r["method"]==method],
              "validation cohort seconds", round(val_wall, 3), flush=True)
    out = ROOT/a.out
    out.mkdir(parents=True, exist_ok=False)
    csv_write(out/"timings.csv", timings)
    # Short-run extrapolation only. Large backlogs and checkpoint IO can change
    # these numbers; no purchase, cloud dispatch or formal training is started.
    eta = {}
    for method, count in (("TMAPO", 5), ("IPPO", 3), ("PPO", 3)):
        rows = [r for r in timings if r["method"] == method]
        per_cell = float(np.median([r["wall_s"] for r in rows[1:]]))*2000 + rows[-1]["one_validation_cohort_wall_s"]*6*21
        eta[method] = dict(cells=count, seconds_per_cell=per_cell, total_hours=per_cell*count/3600)
    json_write(out/"benchmark.json", dict(component_only=True, formal_training_started=False,
        runtime=runtime_info(a.device), source_hashes=source_hashes(), estimate=eta,
        sequential_total_hours=sum(r["total_hours"] for r in eta.values()),
        limitation="reviewed privacy integration; 3x128 timing excludes checkpoint IO, long-run backlog and full-cycle behavior; estimate is not guaranteed"))


if __name__ == "__main__":
    main()

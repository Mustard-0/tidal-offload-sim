"""Bounded throughput measurements, never paper results or checkpoint selection."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_convergence as pc
from four_episode_curves import settings
from cross_slot_learners import CrossSlotPPO


def measure(device, threads, updates=8, optimized=False):
    if threads not in (1, 2, 4) or not 1 <= updates <= 16:
        raise ValueError("Bounded performance probe only")
    torch.set_num_threads(threads)
    random.seed(87021)
    np.random.seed(87021)
    torch.manual_seed(87021)
    cfg, train = pc.configuration(settings("TMPPO", 1))
    cfg["cache_route_physics"] = optimized
    learner = CrossSlotPPO("TMAPO", train, cfg, device=device)
    # Exercise the trained-actor path, not just the initial critic burn-in.
    learner.steps = 2048
    env = pc.new_environment(cfg, 87021, 0, 5)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    stepping, optimizing = 0., 0.
    profiler.enable()
    for update in range(updates):
        tick = time.perf_counter()
        for _ in range(128):
            learner.set_training_clock(6000-env.slot, max(0, 6000-env.slot-201), 6000)
            _, records, _, _ = env.step(learner)
            learner.remember(env.agent_rewards(0.), records, False)
        if device == "cuda":
            torch.cuda.synchronize()
        stepping += time.perf_counter()-tick
        tick = time.perf_counter()
        context, glob, *_ = env.observations()
        learner.update(context, glob)
        if device == "cuda":
            torch.cuda.synchronize()
        optimizing += time.perf_counter()-tick
    profiler.disable()
    wall = time.perf_counter()-started
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
    return dict(device=device, threads=threads, optimized=optimized, updates=updates, env_steps=128*updates,
                wall_s=wall, env_and_inference_s=stepping, optimization_s=optimizing,
                steps_per_second=128*updates/wall, profile=stream.getvalue(),
                purpose="Timing only; altered warm-up counter; never scientific results",
                torch=torch.__version__, cuda_memory_peak_bytes=(
                    torch.cuda.max_memory_allocated() if device == "cuda" else 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Do not overwrite performance evidence")
    value = measure(args.device, args.threads, args.updates, args.optimized)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in value.items() if k != "profile"}, indent=2))

"""Bounded hardware/profile test; never creates a formal training cell."""
import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path
from .runtime import ROOT, np, torch, seed_all, json_write, available_ram_gib
from .protocol import configuration
from .algorithms import learner_for
from .restricted_environment import CandidateRestrictedEnvironment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu','cuda:0'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if available_ram_gib() < 1.5:
        raise RuntimeError('Insufficient available RAM for hardware test')
    cfg, training = configuration('TMAPO', 2)
    seed_all(87021)
    learner = learner_for('TMAPO', training, cfg, args.device)
    env = CandidateRestrictedEnvironment(cfg, seed=20261026, phase_s=30.)
    profiler = cProfile.Profile()
    rows = []
    for update in range(3):
        profiler.enable()
        start = time.monotonic()
        for _ in range(128):
            _, records, _, _ = env.step(learner)
            learner.remember(env.agent_rewards(0.), records, False)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        rollout = time.monotonic()-start
        ctx, glob, *_ = env.observations()
        start = time.monotonic()
        stats = learner.update(ctx, glob)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        optimizer = time.monotonic()-start
        profiler.disable()
        assert all(np.isfinite(v) for v in stats.values())
        rows.append(dict(update=update+1, rollout_s=rollout, optimizer_s=optimizer,
                         total_s=rollout+optimizer, **stats))
    args.out.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(45)
    (args.out/'profile.txt').write_text(stream.getvalue(), encoding='utf-8')
    json_write(args.out/'timing.json', dict(component_only=True, device=args.device, slots=384,
        trace_hash=env.arrival_hash, rows=rows,
        mean_rollout_s=float(np.mean([r['rollout_s'] for r in rows])),
        mean_optimizer_s=float(np.mean([r['optimizer_s'] for r in rows])),
        warning='Profiler overhead included; diagnose bottlenecks, not full-run ETA'))
    print(args.device, [(round(r['rollout_s'],3), round(r['optimizer_s'],3)) for r in rows], flush=True)


if __name__ == '__main__':
    main()

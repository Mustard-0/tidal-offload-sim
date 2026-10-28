# Shared long-horizon pressure discovery, first step only

No learning, no ranking guarantee, no changes to old results or frozen physics.
Retain every original strong Greedy/GeoI-Greedy result, including favorable ones.
This is explicitly environment development guided by heuristic evaluation;
later learned algorithms must use the SAME selected conditions and fresh tests.

Fixed grid before scores: N100/150, per-user arrival intensity3/6/9 per second,
two fixed development traces7298101/7298109, both original strong controllers.
For each: empty-start1s admission/reference and5s causal warmup +300s measured
admission + natural settlement. Phase0; full300s tide, no future information.
Short-to-long also changes warmup; do NOT interpret it as duration-only causality.

Do not change task sizes, deadlines, TTL, CPUs, energy/cycle, link resources,
queue caps, rewards, or privacy to penalize a particular method. Only change
arrival intensity, with the exact same generator for ALL methods. This grid is
not final evidence of a reinforcement-learning advantage; no RL is run here.
The20-150 population scope is not expanded to200-400.

12 rule workers, at most8 standard public free CPU jobs, one NumPy process each,
maximum90min per worker,30min per scene, no failed/unknown reruns. No Torch/GPU.
Raw task/batch/wait records streamed to CSV and authenticated-encrypted only.
Summary/plots encrypted separately for lightweight delivery. Never upload keys,
plaintext results, checkpoints, papers or user files. No recurring monitor.

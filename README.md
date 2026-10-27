# N100: ONE mean-conserving credit review, basic baselines unchanged

This is the only additional review round authorized after the original6 cells
completed. Read HYPOTHESIS.md FIRST. TMAPO alone changes per-agent training
credit to the original .7 local fraction with exact mean conservation at
unequal region sizes. The physical episode objective and reported reward remain
unchanged; this is a declared learning-method change, NOT identical per-agent
reward vectors across algorithms. IPPO/PPO are unchanged identity controls.
New held-out test seeds6296101/6296109 are fixed before any new training.
At most6 additional2000-episode cells, no further reviews or ranking loop.
Original evidence and old final-test scores/curves are not overwritten.

Independent version, not a patch to existing trained results. The original
model/environment/actual-batch bounded privacy was copied from the completed
N100 version at7409d4c0214ae5db19dac09fb9cf562935cfe057, with explicit new
subclasses and a new finite driver. No private Git history or old data export.

Six training cells ONLY: TMAPO/IPPO/PPO x seeds87021/87022, each2000 completed
100-admission-slot + natural-drain episodes. N100 actual34/33/33. One common
actor1e-4 / critic3e-4 LR, no grid or favorable-seed choice. Standard authorized
public CPU runners only, at most6 parallel,3h training budget per cell, no reruns.
Results/checkpoints/logs remain AES-GCM encrypted with the already-authorized key.

## Corrections after the baseline / waiting audit

- ALL methods: analytic actor logit prior removed (scale0), objective-weighted
  local/offload action-hint features removed. No stronger heuristic is appended
  to PPO/IPPO, and no TMAPO trained weights are copied into them.
- TMAPO: local actors, centralized local+global critic, causal forecast inputs,
  optional regional waiting gated by actual public region/class membership,
  lower-confidence historical distinct-user fill probability>=.8 and positive
  modeled PHYSICAL cost benefit including hold time and deadline penalties.
- Basic IPPO: local actor and local critic. Basic PPO: one centralized joint
  action controller and global critic. Neither gets forecast/error features or
  TMAPO's wait predictor. Both batch-release immediately, never deliberately wait
  blindly. They still learn complete local/offload/remote-node decisions.
- No user-queue WAIT; one regional gate owns the20ms budget. Enough users:
  release now. Shortfall unlikely, deadline tight, no history or no cost gain:
  release with safe local perturbation. No future arrivals or real XY read by
  the gate. Model samples use synthetic locations from the fixed PUBLIC disk.
- Old1s prediction window updated only at the final admission slot. New window
  is100ms with20ms forecast horizon; actual300s tidal task model is unchanged.
- Current accountant gives epsilon0=1 for n1..20. At N20-150, batch sizes may
  never make useful amplification reachable within20ms. ZERO waits can be the
  correct output; the code does not fabricate benefits to claim adaptivity.
- Cost prior removal applies to TMAPO too: the shared old prior was never a
  documented TMAPO-only architecture contribution. Actual CTDE/forecast/wait
  components are separated explicitly instead of inventing a favorable bonus.

## Unchanged scientific conditions

Same task sizes/classes/deadlines/lifetime, CPU, bandwidth, queues, physical
energy, conditional per-batch accountant and bounded R500/epsilon1/delta1e-6/
shrink0.5. Reward remains the original N100 physical cost:
.25*resolution_seconds/.5 + .25*system_energy_j/.24 + QoS_failure; QL diagnostic
only. Delay includes queue/report/transmit/compute naturally; failed or late
tasks count in QoS/all arrivals. No resources/deadlines changed for ranking.

Evaluation ONLY actual N20/40/60/80/100/125/150. New validation/test seeds are
separate and registered in study/protocol.py; no selection on the old inspected
test. Primary inference samples the policy as trained, secondary argmax is
reported for ALL algorithms, not whichever mode favors TMAPO. Initial policy
and terminal validation are preserved; terminal checkpoints only. Greedy and
ordinary GeoI-Greedy are evaluated without training, with no TMAPO module.

No recent-paper checkpoint or unfinished ablation from a different protocol is
mixed into this simple study. No training-return ranking is called a physical
metric advantage. Short episodes are not full300s tidal trajectories. Batch
privacy is not a repeated-user trajectory DP proof. No convergence/win promise.

## Reliable finite execution

Local7 standard-library tests; cloud preflight additionally checks ALL three
actor updates, forecast isolation, true-coordinate non-leakage, report-aware
routing, task pairing, physical-gate truth independence, and exact2+2 vs4-episode
checkpoint resume before any formal jobs can start. All6 jobs use the identical
source commit. No hot edits or retrying failed scientific cells.

Checkpoints contain actors/critic/optimizers/normalizer/RNG/full episode CSV at
fully drained boundaries, saved every10 episodes/120s and at resource/wall
limits. A failure/partial completion is retained, not renamed as complete.
The final free CPU job checks the full grid and produces encrypted raw CSV,
physical20-150 plots and MA50 return curves with both fixed seeds retained.

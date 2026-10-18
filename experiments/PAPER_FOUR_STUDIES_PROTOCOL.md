# Reproducible simulation protocol

## Common environment

- CrossSlotWLOEnvironment; three regions, 8 base stations and 3 satellites.
- Default total users 60; 3 tasks per user per second; global slot 10ms.
- Three public task classes; genuine cross-slot WAIT/LOCAL/OFFLOAD, shuffle and routing.
- Physical task arrivals, resource sharing, deadlines and workload seeds are common.
- Epsilon .2, delta 1e-6; raw unprojected planar Laplace reports.
- Objective time/energy/location weights .25 each; QoS penalty 1.
- Delay is per completed task, energy/QoS per generated task, QL per published report.
- Report publication ratio, actual distinct batch size and amplified report fraction alongside QL.
- Strict Local-Only cannot upload. NoTidal hides forecast inputs but preserves real tidal traffic.

## Convergence budget

TMPPO, IPPO and MARATD3-Adapted each use two actor/critic learning rates:
1e-4/3e-4 and 3e-4/1e-3. Training seeds are 87021, 87035 and 87049.
Validation uses seeds 814001/814019 and phases 30/130/230 seconds.

The finite screen uses six cells, one training seed, 128 updates x128 global
slots. The formal budget uses eighteen cells, 2000 updates x128 slots each.
Both use the predeclared 2000-update learning-rate schedule. Full replay
checkpoints are saved every25 updates, validation, pause and completion;
scalar progress and training CSVs are saved every update.

The team_v2 profile uses shared team rewards and a discrete categorical actor
surrogate for the TD3 adaptation. Validation selection uses total measured
task cost divided by total generated tasks, including the update0 candidate.
Random-action diagnostics do not select checkpoints. All training seeds have
equal weight in final seed-level means and standard deviations.

The screen gate checks raw task accounting, pairing, source/config/runtime
identity and checkpoint selection. All six cells must finish and remain
within the registered cost and QoS stability thresholds. Winning an algorithm
comparison is not a gate requirement. Failed screens are retained without
automatically repeating parameter searches.

## Further studies and limits

Scale points are total N30/45/60/75/90. Privacy mechanism diagnostics span
N30/60/90/150/210/300 and epsilon .1/.2/.5/1/2. Sensitivity uses three pairs of
objective weights with five settings each. ATO-SLA adaptation and matched
publication-set privacy evaluation remain pending, not silently substituted.

The convergence workflow does not run the held-out test streams
9848031/9848047/9848063/9848089. Short validation is not full300s tidal-cycle
evaluation. The legacy conditional accountant provides no complete-transcript
DP proof; QL is report error, not an attacker's inference error.

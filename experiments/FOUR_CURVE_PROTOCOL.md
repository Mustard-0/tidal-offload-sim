# Four Complete-Episode Learning-Rate Curves

This separate campaign follows the user's new request after the previous three
bounded diagnostic cycles ended. It does not reopen or overwrite that ledger.
The requested display name TMAPO maps to the existing external TMPPO key and
internal TMAPO learner. There are exactly four cells: TMAPO and IPPO, each with
Actor/Critic LR1=1e-4/3e-4 and LR2=3e-4/1e-3, training seed87021.

## Explicit training protocol change

All four policies learn from scratch: no hand-coded actor cost prior, local
NumPy Xavier-uniform actor initialization with output gain.01 and zero bias.
The same actual initial actor tensors are retained and compared across the
four runners. Nonzero random output weights avoid an artificial all-WAIT
argmax tie at initialization. The previous strong-prior curves remain intact;
these new curves must be labelled FROM SCRATCH, not substituted into old results.
No Greedy, Local-Only, workload, objective, privacy, resource or physical baseline
is weakened. No user-program termination or global power change is needed.

The original CrossSlotWLOEnvironment, 8BS+3LEO, N60, W/L/O, class Shuffle and
counter-keyed traffic are reused. Gamma1, lambda1, entropy0, team reward,
PopArt normalization, actor burn-in2048 slots and critic-only clock are common.
Both PPO actors remain trainable. TMAPO retains its centralized critic, IPPO
its decentralized critic. There is no actor freezing or shared-value pooling.

Each cell has128 COMPLETE episodes of6000 global slots:768000 interactions,
6000 policy updates of128 slots, with a common6000-update LR decay. The four
cells total3072000 registered primary training interactions. This is not2000
episodes, not the old2000-update schedule, and not the formal18-cell study.
Each episode drains its cohort; complete-episode logs reconcile accumulated
training reward to physical cost. Update counts are never relabelled episodes.

## Figures and selection

The requested training plot uses mean per-slot return of each complete episode
and a trailing50-episode arithmetic mean. Before episode50, min_periods=1 uses
all available earlier episodes, explicitly disclosed. No future data, invented
points or modified CSV values are used. All four full raw traces and the full
smoothed plot are retained; smoothing does not establish convergence.

Independent validation is performed at update0 and every375 updates (eight
complete episodes), on seeds814001/814019 and phases30/130/230, with300 warmup
and1000 measured slots per cohort. Raw validation points are not smoothed as
if they were episodes. Each method's preferred LR minimizes pooled task cost
over the FINAL THREE validation checkpoints, using the same full training
budget. This is single-seed exploratory model selection, not statistical proof.
Checkpoint selection separately retains the minimum validation cost including0.

A supplemental exploratory zoom uses ONE COMMON50-episode interval for all
four lines around the validation-selected TMAPO checkpoint. The full plots and
selection rule are supplied beside it. Different per-line best segments may
not be spliced or used to claim overall superiority. If IPPO is better or both
TMAPO LRs regress, report that instead of presupposing the requested conclusion.

## Execution and limits

Four ubuntu22.04 standard public CPU runners, one torch thread each; free-only,
no paid large runner, GPU, Actions artifact or cache. Worker wall15000 seconds,
job timeout285 minutes. Resource/time failure pauses at update boundaries and
retains evidence; no automatic rerun or cross-runner resume. Immutable Release
progress JSONs include actual completed episode aggregates, not checkpoint
attestation. Final Release ZIPs retain raw CSV, source, logs and checkpoints.

Source/config/plan hashes, runtime dependencies, paired trajectories, initial
actors, task-cost accounting,128 completed episodes and validation selection
must be audited. No final blind test, extra seed/LR, fourth diagnostic cycle or
other study is automatic. The previous automation stays paused until this new
campaign is actually launched and its monitor is explicitly updated.

Raw unprojected QL is report-quality loss, not an attacker's inference error.
Legacy one-batch conditional accounting is exploratory, not a full-transcript
or adaptive/longitudinal DP guarantee. Short validation is not a full300-second
tidal evaluation. All failed and zero-improvement outcomes must remain visible.

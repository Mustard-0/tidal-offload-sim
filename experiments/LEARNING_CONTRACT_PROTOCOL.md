# Bounded learning-contract audit

The latest user explicitly authorized continued correctness checks and bounded
training diagnostics. This does not revive any old campaign or permit an
unbounded search until TMPPO wins. Preserve all failed outcomes.

1. Test analytic PPO action-gradient direction, terminal GAE, unchanged
   diagnostics, and the actual environment's single-task WAIT cost. Existing
   TD3 executed-action/target and all prior tests must still pass.
2. Keep gamma=1. Discounting a deferred privacy cost changes the objective and
   can reward postponement, so do not use it as a numerical shortcut.
3. A four-cell TMPPO diagnostic factorial: GAE lambda .98/1, entropy .01/0.
   All four use LR1, seed87021, 128 updates x128 slots, the same 6000-slot
   episodes, actor burn-in, priors, physical tasks, W/L/O/Shuffle, raw noise,
   rewards, validation seeds and 2000-update LR decay schedule. Not an algorithm
   ranking. Changing lambda changes the estimator, not the physical objective;
   removing entropy removes an optimization regularizer and is explicitly
   labelled, not a hidden reward/metric change.
4. Validate at0/32/64/96/128 on both seeds and all three registered phases;
   warmup300/measure1000 slots, drain every cohort. Sampled endpoint diagnostics
   never select models. No final held-out test. All raw traces/checkpoints saved.
5. Exactly four standard public ubuntu22.04 CPU jobs, no local training, GPU,
   paid runner, Actions artifact/cache, or duplicate dispatch. Each worker starts
   only above1.5GiB RAM and pauses below.5GiB at update boundaries. Worker budget
   1200s, runner timeout30min; each result goes to the same immutable Release.
   No automatic formal/2000-update run or blind cloud rerun/resume.
6. Every cell must pass raw integrity accounting independently. A potential
   follow-up is limited to control and one candidate on seed87035, only if the
   candidate's last-three-point mean is >=1% below initialization and its mean
   QoS increase <=1 percentage point. This is a resource gate, not proof of
   superiority. If nothing qualifies, preserve the results and examine the
   next concrete causal hypothesis; never select a winner merely by appearance.

New optional advantage diagnostics must be numerically inert, unit-tested, and
record raw per-role GAE scales. No scientific model or old source is overwritten.

## Second diagnostic: shared centralized team value

The completed first ablation found no candidate eligible for independent-seed
confirmation. Lambda1/entropy0 reduced regression but did not beat the common
prior. This second study is a new mechanistic diagnostic, NOT formal expansion
or a claim that the first gate passed. Decision and evidence are recorded in
the first run's cycle2_decision.json, under the same finite iteration ledger.

Exactly two TMPPO cells, team_value_control/team_value_shared, each128x128,
seed87021, LR1, lambda1/entropy0 and all original physical settings. In the
second cell only, the centralized critic receives the mean of agent contexts
(broadcast identically to every agent) plus the same global observation and
training clock. Actor inputs, architecture and reward stay unchanged. The
representation remains partially observed: pooling is a regularization
hypothesis, not proof that existing per-agent value baselines are incorrect.

Unit contracts check identical team-value inputs/predictions, invariance to
agent row order, unchanged actor outputs, unchanged physical settings, and
rejection of incompatible value-input checkpoints. Both cells must pass raw
accounting; learning gate and final-test exclusions remain identical. Two free
standard CPU cloud jobs only, no local training or automatic long training.
# Final Bounded Cycle: Actor-Component Credit Diagnostic

The second cycle's shared-value arm briefly improved at update64 but then
regressed. A 24-cohort paired replay on the already-registered validation traces
reproduced its initialization/latest endpoints. Replacing the W/L/O actor with
its preserved initialization restored cost from .630675 to .383574 and QoS from
14.196% to zero; replacing only the router left cost .628789 and QoS14.078%.
These are frozen-checkpoint interventions, not new checkpoint selection or a
proof of an incorrect policy-gradient formula.

The final cycle registers role_joint, role_route_only and role_wlo_only, each
128x128 interactions, seed87021. All use shared_team_value, gamma1, lambda1,
entropy0 and the same original task/physical/privacy/objective/validation/LR
protocol. Only optimizer updates for the named frozen actor are skipped. The
frozen actor still uses the common cost prior; it is not a forced LOCAL or
forced RELEASE policy. All critic updates remain enabled. Checkpoints record
the actor-freeze contract and reject incompatible loads. Unit tests require
the frozen parameters to be bit-identical, active actor/critic to update,
empty-freeze behavior to match legacy updates, and all physical configs to agree.

The joint control must reproduce the previous shared-value arm's training
prefix, excluding timing/RAM/new diagnostic columns. All outcomes, including
failures and regressions, are retained. A frozen-actor ablation is NOT the final
fully trained TMPPO algorithm and cannot establish a paper algorithm ranking.
The same sustained-improvement criterion is retained: nonzero selected update,
tail3 cost at most .99 of initialization, QoS growth at most .01. This is the
third and final diagnostic cycle: total registered training interactions147456
of the196608 cap. No fourth cycle, 2000-update expansion, heldout evaluation or
formal grid is automatic, even if an ablation looks promising.


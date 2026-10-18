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

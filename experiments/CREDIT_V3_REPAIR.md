# One bounded credit_v3 repair pilot

## Evidence, Not Cosmetic Smoothing

The preceding team_v2 screen completed six cells but failed its fixed gate.
At 128 updates, TMPPO LR1's deterministic cost rose from 0.378072 to 0.481487,
while its independently sampled diagnostic changed from 0.418992 to 0.420035.
LR2 likewise had a large deterministic regression with nearly flat sampled
performance. This is a policy-mode gap, not proof that PPO's expected return
improved. TD3 deteriorated in both modes, so a mode change cannot repair it.

The physical reward audit passed: raw task time, energy, publication QL and QoS
sum to the training objective after the cohort drains. There is no demonstrated
sign error that justifies changing weights, deadlines, traffic or privacy noise.
The prior screen and checkpoints remain untouched in a separate frozen checkout.

## Changes And Their Limits

1. The undiscounted finite training return depends on the remaining episode and
   arrival horizon, but neither was available to the value networks. Add these
   two known training-clock features to critics only. Do not overwrite any
   existing feature or expose the clock, raw positions or future arrivals to
   actors. Preserve the event-specific current/next clocks in TD3 replay.
2. Freeze all three actors during the first 2048 global interaction slots while
   fitting their critics. Continue target-critic updates during TD3 burn-in.
   This is a declared stabilization hypothesis, NOT a proven cause or cure of
   the preceding regression. Burn-in consumes the normal interaction budget.
3. Keep deterministic argmax as the selection policy. At every validation point
   also evaluate a counter-seeded sampled policy on the SAME exogenous workload,
   labelled diagnostic only. Never choose whichever policy mode looks better.
4. A checkpoint at initialization has no demonstrated improvement speed. Report
   its 95%-improvement time as unavailable, not zero-step convergence.
5. Repair Windows long-path archive extraction and shorten transfer directory
   names. Preserve traversal/symlink/duplicate checks and immutable assets.

The PPO objective, clipping, physical reward, common prior and learning-rate
candidates are unchanged. The TD3 method remains a discrete categorical
recurrent/attention adaptation, not an exact reproduction of continuous TD3.
These changes cannot establish full-transcript DP or guarantee TMPPO wins.

## Predeclared Budget And Stop Rule

Exactly six cells: TMPPO/IPPO/MARATD3-Adapted x LR1/LR2; seed 87021; each 64
updates x 128 slots = 8192 interactions, 49,152 total. Validate at 0/16/32/48/64
using both seeds 814001/814019 and phases 30/130/230 seconds. Warmup and measured
windows are the same 300/500 slots as team_v2. Use its first 64 updates for
equal-budget descriptive comparisons, not its 128-update endpoint.

The previous stability thresholds stay fixed: last three validation costs
average <= 1.25 times initialization, QoS growth <= 5 percentage points, and
mean QoS <= 10%, in every cell. Passing is not a learning or superiority claim.
Retain all failures. No automatic formal dispatch, extra seed, parameter search,
test-stream evaluation or second repair run follows this pilot.

Local TMPPO uses at most two CPU workers. Four disjoint IPPO/TD3 cells may use
free standard public GitHub CPU runners. No larger paid runner, cache or Actions
artifact storage is enabled. Source, runtime, plan and outputs are frozen before
launch. Checkpoints, raw task/batch CSV, logs and policy-mode curves are retained.

## Verification

Regression tests cover critic clock separation from actor inputs, all-method
actor burn-in with changing critics, event-specific TD3 clocks, checkpoint
continuity, exact pause/resume, unchanged physical settings and long-path import.
Code/preflight verification must pass on the exact frozen revision before launch.

References: the trained stochastic-policy objective and critic baselines follow
[Spinning Up policy gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).
Separate repeated evaluation and explicit policy-mode choices are discussed in
[SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
Neither source promises monotone validation curves or improvement in 64 updates.

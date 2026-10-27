# One permitted credit-assignment review, registered before viewing old test scores

## Evidence used, not a desired ranking

Original campaign40569582412 at3f0acd11890a4e8b81a9c811b97058cf92000906
completed all6 x2000 episodes. Only TRAINING and independent VALIDATION metrics
were inspected to choose this hypothesis; the old final-test physical scores
and figures had not been opened when this document and change were prepared.
Original results are immutable, including unfavorable outcomes.

TMAPO seed87021 validation physical cost rose0.2706912924->0.2902371769;
seed87022 fell0.2958636471->0.2595281892. Last100 training KL is about6e-9
and policies remain high-entropy. Both IPPO seeds improve more consistently.
This does not prove a bug or that TMAPO ought to win. It motivates testing whether
weak, globally broadcast credit prevents useful learning of local decisions.

The model's base config contains local_credit_fraction=.7, but the completed
entry explicitly overrode it to0 and called agent_rewards(0). The omission was
a declared common-team variant, not evidence of corrupted training. This new
version makes the proposed TMAPO credit component explicit rather than calling
the old outcome invalid. Waiting is NOT identified as a bug: all0 waits are
consistent with the accountant's unavailable small-batch gains.

## Single prospective change

TMAPO training only: restore the original .7 local-credit fraction as a
declared learning-method change. Redistribute the SAME physically accrued
delay/energy/QoS cost, with no new reward terms, task penalties or true-location
inputs. Global reported episode_return stays IDENTICAL in definition. This does
not claim every agent receives the same reward vector as the baseline.

For N actual users, G=3 regions, slot scale a=lambda*dt and per-user physical
cost c_i, team reward is -sum(c_i)/(N*a). UE local credit is -c_i/a. Region
local credit is -G*sum(c_i in region)/(N*a). Thus BOTH the UE mean and the
regional-controller mean equal the original team reward, including unequal
34/33/33 regions. Mix .3*team+.7*local. The old per-region unweighted mean
would not conserve this identity at unequal populations; do not blindly reuse
that formula. Test conservation and full physical-cost accounting.

IPPO/PPO retain exactly their original team-reward training and basic modules.
All6 configurations are trained again with the same2 seeds, same one LR,
same2000-episode/100-admission-plus-drain budget. Baselines are identity controls,
not modified to make the revised TMAPO look better. No checkpoint warm-start.

## Frozen scope and stop

This consumes the ONLY additionally authorized review round: at most6 new
training cells, each3h, no failed-job rerun. Total original+review maximum12
cells/24000 completed episodes. No more reviews if this fails to help.
The7-day deadline remains anchored to the original2026-10-28 03:29:30
Asia/Shanghai campaign, not reset at this review.

Tasks, resources, reward objective, privacy, masks, forecast and wait gates,
network architectures/optimizer/LR/seed/budget stay unchanged. No force-waiting.
Initial and terminal validation keep seeds6294101/6294109. New held-out test
seeds6296101/6296109 are fixed here BEFORE any review training. All declared
seeds/modes/populations20..150 are retained; sampled is primary, argmax secondary
for EVERY method. Compare the credit hypothesis on validation and retain old
and new final tests separately. No final-test tuning or claim of significance.

A full source-locked real-gradient/resume preflight must pass before new formal
jobs. Preserve failed evidence and stop on unknown/numerical failure. After this
finite round, deliver the true results and end monitoring; do not loop to a win.

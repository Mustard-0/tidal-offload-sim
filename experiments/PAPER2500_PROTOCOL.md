# Simple sequential paper experiments, exact complete-episode budget

This new authorization follows the finished128-episode pilot. It is not a
continuation with a changed schedule:2500-episode training starts from scratch,
using the same seed87021 and the same actor initialization, task flow, physics,
privacy, reward, WLO/Shuffle and8BS+3LEO. TMPPO is displayed as TMAPO.
The pilot's strong phase-dependent raw oscillation and smoothing prefix are
not evidence that a learner converged; do not modify the task distribution or
choose different intervals per method to make curves separate.

## Four experiments only

1. TMAPO and IPPO, each LR1=1e-4/3e-4 and LR2=3e-4/1e-3. Each cell is
   exactly2500 complete6000-slot episodes,15000000 interactions,117188 updates
   of at most128 slots (last update64). Independent validation every3750
   updates plus initialization/final. Preferred LR minimizes pooled task cost
   over the last three validation points. Best checkpoint separately includes0.
   Full50-episode trailing plots retain the first49 as a shorter dotted prefix,
   raw episodes and independent validation; no splicing or forced superiority.
2. User-scale generalization at total N30/45/60/75/90 using frozenN60 policies.
   This is explicitly zero-shot scale transfer, not an undisclosed retraining
   at everyN. Compare TMAPO, IPPO, Greedy, Fixed-GeoI+Greedy, NoTidal, NoAmp,
   NoWait, MARATD3-Adapted and MATORA-DQN-Adapted. Local-Only is a diagnostic
   reference. NoTidal is separately trained with only forecast observations
   masked; NoAmp and NoWait are labelled frozen-policy mechanism ablations.
   NoAmp changes calibration only; NoWait removes the WAIT action for the same
   frozen policy. Fixed-GeoI+Greedy uses the same greedy cost-based offloading
   and route policy, immediate release and fixed local Geo-I calibration.
3. Privacy C1 is ONE simple controlled quality figure: total300 synthetic users,
   three public-region batches of100 distinct members, one report/member per
   independent repetition, eps.1/.2/.3/.5/1, delta1e-6,100 repetitions. Geo-I
   and Geo-I+Shuffle reuse the actual local accountant/noise implementation and
   paired standardized noise. No public-region projection. This is a mechanism
   quality benchmark with a specified100-member batch, NOT evidence that the
   deployment's small observed task batches always amplify; no delay/energy or
   full-transcriptDP claim. Old service batches are reference only, not relabelled
   or silently replaced by the population count. Pasted example numbers are not
   experiment inputs. C1 is not a second epsilon-by-population training grid.
4. Three two-weight trade-offs: time-energy, time-quality, energy-quality.
   Each pair takes(.1,.5),(.2,.4),(.3,.3),(.4,.2),(.5,.1); the other weight=.2,
   QoS weight=1. Each setting gets its own2500-episode policy at the chosen
   TMAPO LR, rather than rescoring one policy. Plot raw physical metrics using
   fixed physical reference units, not arbitrary axes chosen to force crossing.

Frozen validation streams814001/814019, phases30/130/230;300warmup+1000
measurement slots per cohort. Four reserved final test streams
9848031/9848047/9848063/9848089,30s warmup+300s measurement, drain afterwards.
No heldout test result selects LR, seeds, weights, method variants or checkpoint.
This simplified batch has one training seed. Test-trace variability is NOT
training-seed uncertainty; these are exploratory curves, not final three-seed
statistical evidence. All failures, overlap and regressions are retained.

## Real recent-method adaptations

MARATD3 (Computer Networks272:111680,2025):
https://doi.org/10.1016/j.comnet.2025.111680
The existing recurrent event-time history, twin attention critics, delayed
actor updates and categorical discrete surrogate are reused. Original
continuous resource allocation is not reproduced. All methods share the same
FCFS/resource physics and the same objective.

MATORA (IEEE IoT Journal12(4):3832-3845,2025; online2024):
https://doi.org/10.1109/JIOT.2024.3482290
The distributed DQN offloading component is also described by the authors'
group in https://patents.google.com/patent/CN116405962B/en . This implementation
has actual Q networks, epsilon-greedy replay, target-network bootstrapping and
event-aligned returns, not renamed PPO. Its two adapted WLO/route heads retain
the common masked action set. Original convex CPU allocation is replaced by
the common fixed resource model for every comparator. Name it MATORA-DQN-Adapted,
not an exact author reproduction. Q-network LR uses the critic candidate; there
is no actor LR in DQN. Two Q-LR candidates use the same validation selection rule.
These two2025 methods satisfy the requested recent-method comparison without
inventing a2026 reproduction from an abstract. ATO-SLA2024 is related-work
reference, not required in the latest main baseline list. Do not upload its PDF.

## Performance, execution and finite gates

Engineering only: append scalar training CSV instead of quadratic full rewrites;
reuse same-slot identical channel calculations and vectorize conservative
coverage across regions. Raw tasks/metrics must match the uncached path exactly.
CPU/CUDA timing probes are explicitly not scientific training. The measured
small sequential simulator is slower on CUDA; do not equateVRAM occupancy with
throughput. Same1torch-thread CPU contract for local/cloud, no paid resource.

Local at most4 independent worker processes; public standard ubuntu22.04 at
most16 cloud jobs, never more jobs than independent registered cells. Start
RAM>=1.5GiB, runningRAM<.5GiB saves an update boundary, then reduces local
concurrency. No killing user applications, clearing caches, global power edits
or deliberate OOM. Temporary idle-sleep prevention is released at queue exit.
Each child exits before its memory/context is reused. Resource waits stop after12h.

Each segment at most3750 updates/10800seconds, well before the6-hour hosted
job cap. Parent/child exit, settings/source/runtime identity, CSV and latest
checkpoint agreement, unique asset digests/CRC and a paused boundary are required
before resume. Immutable Release assets, not paid artifact/cache storage. Cloud
account/quota errors may transfer a NOT-STARTED job or a verified paused prefix
to local. Unknown/unpublished outcomes and code/numerical errors never trigger
blind rerun. Training cannot be split into independent chunks and concatenated.

The learning gate for the selected TMAPO LR is: final3 validation cost<=.99 of
initial, QoS growth<=.01, QoS<=.10, tail costCV<=.05, selected checkpoint>0.
It does not require TMAPO to beat IPPO. If it fails after2500 episodes, stop
expansion and retain/report the outcome rather than repeat until successful.

Only after passing:5 auxiliary cells (NoTidal and twoLRs for each recent method),
then scale evaluation, simpleC1,15 sensitivity cells and their evaluation.
Total registered training cap24cells/360000000 interactions, maximum48segments
per cell including resource pauses, maximum7days. No auto extra seeds or retries.

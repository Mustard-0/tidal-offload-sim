# Moderate shared-load comparison, 2026-10-28

This NEW finite experiment implements the human's request after the completed
Greedy pressure study. It never resumes an old owner, queue, model or test set.
The repository-root README belongs to the unchanged historical source;
this file and protocol.py define the active experiment.

## Frozen common conditions

- Actual training N150=50/50/50, arrival6 tasks/user/s, same task profiles,
  deadlines, lifetime, CPUs, links, queues and energy coefficients as the
  completed rule pressure study. This is its moderate2x load, not severe3x.
- Four complete300s admission cycles;2s fixed natural settlement per cycle.
  Exactly120800 environment interactions/1208 updates at100 slots/update.
  Queue/predictor history persists across updates. These are FOUR cycles,
  NOT1208 empty-start episodes or2000 full300s episodes.
- Five methods x fixed seeds87021/87022 = TEN cells, ONE LR per method.
  TMAPO/IPPO/PPO actor1e-4,critic3e-4; SAC3e-4/3e-4;
  MARATD3 MLP1e-4/1e-4,LSTM.1x. Hidden32, common slot gamma.999.
  Replay methods retain6 gradients per100 slots; equal interactions are
  not a claim of identical optimizer workload or independently optimized LRs.
- Same physical cost .25*time/.5 +.25*system_joules/.24 + QoS_failure;
  QL diagnostic only. TMAPO has the already-reviewed mean-conserving.7 local
  credit; other learners use scalar team rewards. This is a declared method
  difference, not identical per-agent reward vectors.
- Every learned method has analytic actor cost prior0 and user weighted-cost
  hints masked. Basic PPO/IPPO and recent adapters receive no TMAPO forecast.
  Strong Greedy retains its explicit public physical model as a reference.
- Same bounded R500/epsilon1/delta1e-6/actual region-class distinct-user batch/
  shrink.5 mechanism for learned methods. Ordinary GeoI-Greedy is an explicitly
  different privacy reference. No truthful XY enters an executing policy.
- TMAPO retains causal forecast and a physically guarded optional20ms regional
  wait; baselines release immediately. Zero waits may be correct, never force
  batching or claim a waiting advantage without observed positive effects.

## Real method differences and recent papers

TMAPO: local actors, centralized local+global critic and declared local credit.
IPPO: local actors/local critic. PPO: centralized joint-action likelihood and
global critic, NOT a renamed update class or release-only pilot.
SD-SAC-Adapted: central masked discrete actor, independent twin Q, targets,
decision-event replay with slot-time discount, learned entropy temperature.
MARATD3-Adapted: two-layer LSTM actor/critics, two-head attention excluding self,
twin critics, delayed updates, target smoothing and continuous-intent decoding.
Its regional route intent uses aggregated reports rather than a per-task
continuous CPU/bandwidth allocation. This limitation is retained and disclosed.

The recent adapters are copied from the previously reviewed highload baseline
module, not from the unrelated Atari ZIP. They are NOT author-code reproductions:
MARATD3: Computer Networks2025, DOI10.1016/j.comnet.2025.111680.
SD-SAC: Computer Communications2026, DOI10.1016/j.comcom.2026.108623.
The task topology, physical objective, masks, discount and finite replay schedule
are adapted to this shared simulation and are not claimed identical to papers.

## Independent validation and held-out evaluation

Before updates and after each cycle, preserve fixed validation at N150,
5s causal warmup +10s measured, phases30/150, seeds8298101/8298109.
No best checkpoint or seed is selected; use the terminal120800-step policy.
Held-out test seeds9298101/9298109 were fixed BEFORE these new runs and differ
from the inspected pressure-development traces7298101/7298109.
Evaluate actual N20/40/60/80/100/125/150,5s warmup +300s/full tide +settlement.
Native sampled categorical policies and deterministic TD3 intent are declared
in advance; never switch modes to favor a method. All seven final curves share
the same newly generated exogenous inputs and physical conditions.

## Reliability and finite resource budget

New source uses exact slot*time arrival stamps and a one-slot prepared cache.
Full input records, raw task/batch/wait CSV chunks and rolling prefix hashes
are preserved; no old raw data or clock hashes are rewritten. The new input
logger is diagnostic only, never feeds truthful XY to a policy.
Every10 updates save model/critic/targets/optimizers/normalizer/replay/pending
actions/environment/queues/prepared state/RNG and complete log prefix.
No file handle is placed inside an environment checkpoint.
Five-method real actor/critic updates and exact2+4 vs6-update resume MUST pass
in the source-locked gate before formal dispatch. Profile actual N150/rate6.
The component gate is not formal training and never uses final test seeds.

At most12 standard public free CPU jobs (10 independent learner pipelines and
two rule evaluations), one Torch thread per worker, no GPU/paid runner/local
training. Each cell has at most TWO9000s training segments, total5h; only a
verified ordinary pause may resume. A completed segment is referenced, not
trained again. Unknown/numerical/scientific failures halt further expansion;
already dispatched finite segments preserve evidence. No failed-job rerun.
Each terminal evaluation has a3h bound. The campaign is finite, at most7days.
Full outputs/checkpoints use the already-authorized AES-GCM Release transport;
never upload plaintext results, papers, user files or secrets. Desktop low RAM
does not stop healthy cloud jobs; there is no desktop owner dependency.

Deliver actual scalability, fixed validation and MA50-update return plots,
raw CSV/config/logs/checkpoints, including all unfavorable/incomplete cells.
Training plots disclose full-cycle phase/reset boundaries; smooth appearance
is not convergence proof. Two-seed SD is not CI. Batch privacy is not trajectory
DP. No rank-triggered environment changes, extra seeds, LR grid or training.

# High-load causal RL and matched ablations, 2026-10-27

Independent NEW experiment. Old N100/3000-episode sources, weights and results
are preserved. No promise of TMAPO superiority or stable convergence.

## Actual corrections

-100 steps is a ROLLOUT/UPDATE, not a drained episode. Queues and causal Holt
  history persist across updates. Each physical cycle is300s/30000 steps, with
 29800 admission steps and200 fixed drainage steps. Ten cycles = exactly300000
  environment interactions/3000 updates for EVERY cell, including drainage.
- Real populations by cycle:300,200,400,300,100,400,200,300,400,300, split equally
  across3 regions. Same seeded tasks for every method; no phantom batch users.
- R500m, epsilon1, delta1e-6, improved bounded actual-region/class distinct-user
  calibration and shrink.5 retained. Same tasks, deadlines, CPU and links.
- Active WAIT is ZERO for every main method. The20ms ceiling remains a hard
  limit, not mandatory delay. FCFS compute queueing is separately measured.
- Categorical learned policies are sampled at deployment as trained. Hard
  argmax is a secondary diagnosis, never used only for weak baselines. TD3
  uses deterministic continuous intents with no extra evaluation exploration.
- Latency-first common cost: delay/.1s + .1*energy/.1J +5*QoS failure. A100ms
  delay costs1,100mJ costs.1, a violation costs5. QL is diagnostic only.
  This is a NEW declared objective, not relabeling old rewards/results.
- Correct nonterminal bootstrap uses the next CURRENT post-arrival state.
  A prepared-slot cache prevents duplicate generation on resume. Checkpoints
  include queues, predictor, prepared state, optimizer, replay/pending actions,
  full RNG and counters. Exact mid-cycle resume is tested.
- Administrative observation-window end flags are hidden from every learned
  policy/critic. No future workload or true coordinates enter decision inputs.
- MARATD3 uses compact chronological replay, excludes missing evicted context,
  and buckets minibatches by real agent count across population changes.

## Finite matrix and validation

31 screen cells: TMAPO5LR x3seeds; IPPO/PPO each2LR x2seeds; SD-SAC-Adapted and
MARATD3-Adapted each2LR x2seeds. Rates are listed in protocol.py. TMAPO-family
critic/actor ratio3; discrete SAC equal actor/critic rates. MARATD3 LSTM rates
are.1 x its listed MLP rate. Seeds are87021/87022/87023, not selected by outcome.

After the main screen, all methods select LR by the SAME terminal validation
pooled physical cost, retaining all seeds and terminal budget. No lucky
checkpoint selection. Initial policy and blocks1000/2000/3000 are evaluated.
Validation uses seeds814001/814019, actual N200/300/400, phases30/150,3s actual
causal warmup+3s measured cohort with full drain. Short validation is a screening
tool, not full-cycle scientific evidence.

Then3 ablations x3seeds use the selected full TMAPO LR: NoForecast, NoPrior,
LocalCritic. At identical LR/seed, LocalCritic is exactly IPPO; compatible
completed IPPO cells are referenced with their original origin, not retrained
or passed off as new executions. At most40 training cells (38 if two exact
aliases can be reused). NoWait is not a separate curve because active WAIT is
already zero. Ablations change policy information/components, NOT true loads.

Final scalability: real N100/200/300/400, four NEW held-out seeds,5s causal
warmup+300s measurement/full tidal period+natural drainage. Compare all main
methods, Greedy, GeoI-Greedy and3 ablations. GeoI-Greedy explicitly changes the
privacy mechanism; it is not a same-noise algorithm-only ablation. Store raw
task/batch CSV gzips, failures, delay/energy/QoS, arrival hashes and provenance.

The recent-paper methods are architecture adaptations, not exact reproductions
of the authors' physical topology, resource-allocation or SDN implementation.
They are not yet trained merely because component tests passed.

## Meaning of Greedy

Greedy picks the feasible node with minimum CURRENT reported-state physical
cost (queue, transmission, computation, energy and deadline estimate). It does
not look at future arrivals. Those SAME estimates enter every learner's route
observation and common analytic cost prior. The learned actor can change the
choice. A strong current-state heuristic is still greedy; it is not a global
future-aware optimum. Do not weaken it, add selective control-plane overhead,
or alter the workload after observing rankings.

## Execution / safety

Code and component tests are separate from formal training. New formal models
start from scratch; no old checkpoint is compatible with this new objective.
Finite limits:7days,96 cumulative segments/cell,3h training segment, no retries
of unknown/numerical/code failures. A local owner holds one exclusive lock,
checks RAM, never kills user processes and saves at update boundaries.

Human explicitly approved public pure code and encrypted simulation checkpoints,
logs and results in an independent Release on Mustard-0/tidal-offload-sim.
Only AES-GCM ciphertext and non-performance ownership receipts are published;
no plaintext results, PDFs, user files or keys. The local saved197-update prefix
is adopted only after both local processes exit and immutable hashes match.
One lightweight cloud owner handles at most16 standard public CPU jobs, with
96 cumulative segments per cell and the ORIGINAL seven-day start. No reruns of
failed/unknown jobs. The local segment counts as segment1, not reset to zero.
Use ops.test_cloud_contract before ops.deploy_cloud prepare/publish and the
exclusive ops/launch_cloud.ps1. Publishing code is not proof of job execution;
verify real workflow SHA/attempt and actual started receipts.

```powershell
$env:PYTHONPATH='F:/stin_project/experiment_runtime/torch_cpu_2_5_1;F:/stin_project/tmapo_lr_speed_20261026/.runtime'
& D:/anaconda/envs/stin/python.exe -m experiments.highload.preflight
# Only after the saved final preflight/source hashes match:
& D:/anaconda/envs/stin/python.exe ops/local_owner.py --out runs/highload_local_01 --first-only
```

Plots must use cumulative environment steps/updates. Do NOT label3000 rollout
updates as3000 drained episodes. Population-cycle boundaries must be disclosed;
training-return movement across loads alone is not convergence evidence.

## Execution-only acceleration revision

The human subsequently requested maximum free-resource utilization. The frozen
scientific plan/source/budgets above remain unchanged. An explicit execution
override uses a conservative20 standard-job cap (the account API did not return
a plan) and up to4 INDEPENDENT one-thread processes per four-vCPU Ubuntu runner.
No GPU/larger/paid runner, extra configurations, seeds or interactions are added.
All RNG, queues, models, CSVs and logs remain process-isolated per cell. Existing
single-cell jobs retain their ORIGINAL public commit and keep running normally.
New waves pin their own execution commit, not the old wave's commit.
The local dispatch planner also applies a12GiB conservative per-runner estimate:
SAC training6GiB / final-test checkpoint loading10GiB, recurrent TD3 2.5GiB,
other processes1.5GiB. Heavy replay models are NOT blindly packed fourfold.
The worker commit that passed the cloud probe is pinned independently of the
local scheduler commit; memory grouping only reduces that tested process count.
The measured four-process aggregate throughput was1.83x, not4x; individual
processes slowed from CPU contention. Therefore dispatch uses the SMALLEST
process count needed for the ready queue/free machine slots, up to4. If enough
machines are free, each cell gets its own runner. One pack per wave allows1/2/3/4
processes with the same tested worker; polling is60s to leave API-rate headroom.

Activation requires an actual cloud component-only serial-vs-four-process probe:
numeric training rows and COMPLETE learner/optimizer/replay state hashes must
match, throughput must improve by at least15%, and summed peak RSS stays below
12GiB. Only the small benchmark report is encrypted/uploaded; component models
are not published or counted as formal results. See ops/benchmark_packed.py.
The probe also privately records a CPU profile before single-process changes.

The original lightweight owner exited on RemoteDisconnected during a GET jobs
request. Audit evidence preserves the old state, all wave ownership, actual
SHA/attempt/jobs, and absence of partial imports. ops/packed_owner.py adopts it
only after verified owner exit and the matching actual cloud probe. Read-only
network operations now retry; uncertain writes never blindly repeat. Exclusive
lock and an execution_control.json action=quiesce support future safe handoff.
The base plan's16 cap is historical; actual resource override and probe provenance
are recorded separately in coordinator.json, not silently relabeled.

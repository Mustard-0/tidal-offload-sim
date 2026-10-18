# 2500 complete-episode queue

Latest authorization: one exploratory seed87021, four TMAPO/IPPO learning-rate
cells,2500 complete6000-slot episodes each. Preserve free CPU only. Do not resume
v75/v76, old diagnostics or the old128-episode pilot. Use PAPER2500_PROTOCOL.md
for the frozen scientific and finite execution contract.

Development verification: runs/paper2500_verification_01/verification.json.
It includes54 model tests,89 experiment tests, syntax and physical preflight.
The short timing probes under runs/perf_01 are performance diagnostics, not
paper results. Local CUDA was slower than one-thread CPU for this simulator.

Do not call training started until runs/paper2500_01/launcher.json and actual
PIDs/logs demonstrate progress. The coordinator is standard-library only;
scientific code is isolated in exiting children. Start it only once. Windows
launches must be hidden with separate stdout/stderr and a retained PID/session.
Use the pinned CPU runtime described in the protocol, not environment-default
CUDA Torch. The coordinator owns a temporary idle-sleep request.

Keep all source/settings/plan/raw/checkpoints frozen once the plan is created.
Verify parent/children have exited, paused/latest/raw agree and the exact source
and dependencies match before any resume. Unknown failures do not authorize a
rerun. Never split independent chunks and splice a2500-episode curve. Segment
checkpoints retain learner, environment, RNG, optimizer, replay and CSV prefix.

After the declared learning gate passes, the same finite queue runs auxiliary
comparators, paired zero-shot user-scale evaluation, simple C1 quality and
pairwise weight studies. Max24 training cells/360M interactions/7days/48 segments
per cell. No automatic retry until TMAPO wins. Full heldout tests do not tune
learning rates or variants. Single-seed figures are not final statistical proof.

Privacy C1, not C2: the pasted table was a reference, not measured values.
Old v76 reference task batches were1-15 distinct members, not100 just because
there were300 users. Do not upload that private old CSV or relabel it. New C1
is a clearly labelled controlled one-report/member comparison with3x100-member
public-region batches. It measures raw QL without projection, not service delay,
attacker error or full-transcript DP. Reuse existing noise/accounting code; do
not invent a broad privacy training matrix or claim deployment batch gains.

Recent baselines are explicitly architecture adaptations MARATD3(2025) and
MATORA-DQN(2025 issue, online2024), with common resource physics/reward. Do not
rename another PPO to either method. ATO-SLA2024 PDF is reference only; it must
not be uploaded to the public repository.

Automation v71 should stay quiet on healthy unchanged states, notify meaningful
stage transitions, failures or actionable risks, and pause at the finite queue
end or needs_review. Never launch a duplicate coordinator/cloud wave.

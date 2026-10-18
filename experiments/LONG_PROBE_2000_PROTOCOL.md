# Explicitly authorized 2000-update exploratory run

The user requested long GitHub training after reviewing the failed credit_v3
short screen. This new authorization tests whether longer training helps. It
does not retroactively pass that screen or authorize the formal 18-cell matrix.

- Six independent jobs: TMPPO, IPPO, MARATD3-Adapted x LR1/LR2.
- One predeclared seed 87021, no seed search or automatic repetition.
- Each job starts from initialization for 2000 updates x128 global slots.
  Total 1,536,000 training interactions. The changed episode horizon is 6000
  slots, as already registered for long training; do not splice short curves in.
- Keep credit_v3 critic clocks, shared rewards and 2048-slot actor burn-in,
  actor/critic LR1=1e-4/3e-4, LR2=3e-4/1e-3; fixed 2000-update LR schedule.
- Keep N60, actual cross-slot W/L/O, Shuffle, 8 BS +3 LEO, raw location noise,
  task profiles, arrivals, physical resources and reward weights unchanged.
- Deterministic paired validation at update0 and every100 updates uses both
  seeds 814001/814019 and phases30/130/230, with 300 warmup and1000 measured
  slots. Minimum pooled per-task validation cost selects the checkpoint,
  including initialization. Sampled diagnostics only at0/2000 reduce overhead
  without changing primary evaluation or training RNG. No held-out test runs.
- At most six standard ubuntu22.04 public CPU runners. No paid larger runner,
  GPU, Actions artifact/cache storage, or local long training. Every job has
  a 15000-second worker budget and a285-minute runner timeout. A worker budget
  or RAM stop saves a paused checkpoint; do not silently restart it elsewhere.
- Keep CSV, raw validation task/batch traces, logs, frozen source, best/latest
  checkpoints and provisional plots in immutable Release ZIP assets. Publish
  partial results on failure/pause. Windows atomic-file sharing errors get only
  bounded retries; permanent errors still fail visibly.

Code tests and physical preflight must pass before dispatch. The raw-data audit
is separate from learning success; retain prior stability ratios and failures,
and never call an intact but deteriorating policy a successful convergence run.
All six outcomes remain visible. This single-seed, short-validation exploration
does not choose a final learning rate, prove superiority or full-transcript DP.
No subsequent study, extra seed or new run is automatic after completion.

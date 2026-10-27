"""One seed, three algorithms, eleven cells; no other experimental matrix."""
from pathlib import Path
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "85b5e3ccd9fd2d5f8786e5d89daf04c8a471d5ed"
PRIVACY_COMMIT = "e2e23b20b67e2100acfc33d33c744a22bd78dd38"
LRS = {
    "TMAPO": ((3e-5, 9e-5), (1e-4, 3e-4), (2e-4, 6e-4),
              (3e-4, 9e-4), (5e-4, 1.5e-3)),
    "IPPO": ((3e-5, 9e-5), (1e-4, 3e-4), (3e-4, 9e-4)),
    "PPO": ((3e-5, 9e-5), (1e-4, 3e-4), (3e-4, 9e-4)),
}
METHODS = {
    "TMAPO": "shared local W/L/O+route actors; V(local, public global) per agent; team reward",
    "IPPO": "shared local W/L/O+route actors; V(local) per agent; team reward",
    "PPO": "one central controller; pi(decision,public global), joint slot likelihood; V(public global); team reward",
}


def configuration(method, lr):
    if method not in LRS or not 1 <= lr <= len(LRS[method]):
        raise ValueError("Unknown algorithm or algorithm-specific LR index")
    base = json.loads((ROOT / "model/config.json").read_text(encoding="utf-8"))
    c, t = copy.deepcopy(base["environment"]), copy.deepcopy(base["training"])
    c.update(users_per_region=100, region_radius_m=500., epsilon_req=1., delta_req=1e-6,
             wait_max_s=.02, prediction_horizon_s=.02, project_public_region=False,
             mechanism="bounded_tight_actual_batch_shrink0.5", report_shrink=.5,
             weight_time=.25, weight_energy=.25, weight_quality=0., weight_violation=1.,
             quality_ref_m=500., min_service_reserve_s=.04,
             wait_fill_probability=.8, wait_min_history_s=1.,
             active_wait_mode="immediate_until_physical_benefit_validated")
    t.update(seed=87021, updates=2000, rollout_steps=128, episode_steps=6000,
             hidden=32, epochs=2, minibatch=256, actor_cost_prior_scale=2.,
             gamma=1., gae_lambda=.98, value_normalization=True, local_credit_fraction=0.,
             linear_lr_decay=True, lr_final_fraction=.1, torch_threads=1,
             validation_interval=100, checkpoint_interval=25,
             validation_seeds=[814001, 814019], validation_phases=[30., 130., 230.],
             validation_warmup_s=3., validation_measure_s=10.)
    t["actor_lr"], t["critic_lr"] = LRS[method][lr-1]
    return c, t


def plan():
    c, t = configuration("TMAPO", 1)
    cells = [dict(id=f"{m}_LR{i+1}_seed{t['seed']}", algorithm=m, lr_index=i+1,
                  actor_lr=a, critic_lr=v, env_steps=t["updates"]*t["rollout_steps"],
                  state="not_started_pending_preflight", started=False, complete=False)
             for m, rates in LRS.items() for i, (a, v) in enumerate(rates)]
    return dict(schema="lr-screen-20261026-v2", status="pending_source_locked_preflight",
                base_commit=BASE_COMMIT, standalone_privacy_commit=PRIVACY_COMMIT,
                environment=c, common_training=t, cells=cells, algorithms=METHODS,
                lr_rule="actor and critic jointly scaled at fixed critic/actor ratio 3",
                lr_reason="include conservative 3e-5 and prior 1e-4/3e-4 actor anchors; TMAPO adds two intermediate/high points; no result-based grid edits",
                total_training_env_steps=sum(r["env_steps"] for r in cells),
                selection="checkpoint: minimum pooled validation cost including step 0; LR: minimize last-five validation mean+SD among stable candidates; if none stable, provisional only",
                stability=dict(minimum_validation_points=8, window=5, max_tail_cv=.1,
                    max_relative_linear_drift=.05, max_tail_cost_over_initial=1.25,
                    max_tail_qos_increase=.05, finite_diagnostics_required=True,
                    interpretation="operational screening rule only; cannot prove convergence"),
                reward="negative accrued team cost / (300*3 tasks/s*0.01s); same for all methods",
                cost=".25*resolution_seconds/.5 + .25*energy_j/.24 + 1*QoS_violation",
                quality_scale_reason="QL is diagnostic only, explicitly confirmed by user; not a reward or selection term",
                waiting_limitation="20ms hard cap; active WAIT masked until a causal physical-cost benefit estimator is validated; immediate release, no forced batching",
                denominators=dict(delay="completed", energy="arrivals", qos="arrivals", cost="arrivals", ql="published"),
                reserved_test_seeds=[481953, 729604, 814113, 814229], test_set_used=False,
                smoothing="none in primary plots; any optional smoothing must keep raw CSV and state window",
                evidence_limit="one training seed; short phase-stratified validation; not multi-seed paper evidence or full-trajectory DP",
                post_run_rule="analyze all 11 outcomes; any necessary code repair creates a new source revision and reruns all affected cells at equal budgets; retain old runs")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

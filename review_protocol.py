"""Finite correctness gate, not a performance-ranking search."""
import hashlib
import json
from pathlib import Path

CAMPAIGN = 'baseline-isolation-preflight-20261119'
BRANCH = 'experiment/' + CAMPAIGN
BASE_COMMIT = 'db615a445eb5a41f2f4cccaf55764750e5ca9920'
POPULATIONS = (20, 40, 60, 80, 100, 125, 150)
DEVELOPMENT_SEEDS = (871901, 871909)
DEVELOPMENT_TRACES = (1191901, 1191909)
LEARNERS = ('TMAPO', 'IPPO', 'PPO', 'SD-SAC-Adapted', 'MARATD3-Adapted')
ABLATIONS = ('NoForecastFeatures', 'NoWait', 'NoLocalCredit')
RULES = ('Greedy-NoPrivacy', 'Greedy-GeoI', 'Local-Only')
FEATURE_SLICES = {'user': ((4, 6),), 'region': ((57, 59),),
                  'route': ((57, 62),), 'global': ((28, 34),)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def settings(method, seed=DEVELOPMENT_SEEDS[0], n=20, ablation=None):
    if method not in LEARNERS + RULES or seed not in DEVELOPMENT_SEEDS or n not in POPULATIONS:
        raise ValueError('Outside finite correctness-gate scope')
    if ablation is not None and (method != 'TMAPO' or ablation not in ABLATIONS):
        raise ValueError('Ablations apply only to TMAPO')
    # Reading JSON does not import NumPy, Torch, checkpoints, or training engines.
    source = Path(__file__).resolve().parent / 'model/config.json'
    if not source.exists():
        source = Path(__file__).resolve().parent.parent / 'tmapo_pressure9_20261028/model/config.json'
    base = json.loads(source.read_text(encoding='utf-8-sig'))
    c, t = base['environment'], base['training']
    counts = [n // 3 + int(i < n % 3) for i in range(3)]
    c.update(users_per_region=max(counts), region_user_counts=counts,
        region_radius_m=500., arrival_per_user_s=9., epsilon_req=1., delta_req=1e-6,
        report_shrink=.5, project_public_region=False,
        mechanism='bounded_tight_actual_batch_shrink0.5',
        weight_time=.25, weight_energy=.25, weight_quality=0., weight_violation=1.,
        time_ref_s=.5, energy_ref_j=.24, quality_ref_m=500., wait_max_s=.02,
        prediction_window_s=.1, prediction_horizon_s=.02,
        active_wait_mode='causal_physical_gain_guard_v1', observation_horizon_slots=30000,
        min_service_reserve_s=.04, wait_min_history_s=.1, wait_history_window_s=.2,
        wait_fill_probability=.8, wait_model_samples=16, wait_model_margin=.002,
        wait_confidence_z=1.645, baseline_release='immediate')
    t.update(seed=seed, hidden=32, actor_lr=1e-4, critic_lr=3e-4,
        gamma=.999, gae_lambda=.98, epochs=2, minibatch=256, ppo_clip=.2,
        target_kl=.02, entropy=.01, value_normalization=True,
        local_credit_fraction=.7 if method == 'TMAPO' and ablation != 'NoLocalCredit' else 0.,
        max_grad_norm=.5, actor_cost_prior_scale=0., linear_lr_decay=True,
        lr_final_fraction=.1, torch_threads=1, offpolicy_updates_per_block=24,
        history=8, batch=8, replay_capacity=2048)
    if method == 'SD-SAC-Adapted':
        t.update(actor_lr=3e-4, critic_lr=3e-4, batch=128, replay_capacity=100000, initial_alpha=.01)
    elif method == 'MARATD3-Adapted':
        t.update(critic_lr=1e-4)
    if method != 'TMAPO':
        # Baseline reporters never use TMAPO contraction or batch amplification.
        c['report_shrink'] = 1.
        c['mechanism'] = 'ordinary_planar_geoi_individual' if method != 'Greedy-NoPrivacy' and method != 'Local-Only' else 'none'
        if c['mechanism'] == 'none':
            c.update(privacy_processing_s=0., report_energy_j=0.)
    return c, t


def plan():
    return dict(schema='baseline-isolation-correctness-gate-v1', source_base=BASE_COMMIT,
        populations=POPULATIONS, development_seeds=DEVELOPMENT_SEEDS,
        development_traces=DEVELOPMENT_TRACES, learners=LEARNERS, rules=RULES,
        ablations=ABLATIONS, ranking_is_not_a_gate=True, formal_experiments_started=False,
        gate=dict(maximum_jobs=1, wall_seconds=1500, maximum_updates_per_learned_case=6,
            rollout_slots=100, learner_population=20, environment_steps_per_smoke_scene=80,
            all_population_rule_scenes=21, smoke_trace=DEVELOPMENT_TRACES[0],
            automatic_training_or_second_run=False, free_cpu_only=True),
        scientific_controls='Unchanged tasks, queues, links, compute, deadlines and physical objective; new development inputs fixed before outcomes',
        information='No baseline forecast, local-credit vector, weighted logit prior, active wait, shuffle amplification or report contraction',
        comparison_scope='Full TMAPO versus ordinary-GeoI controllers is a SYSTEM comparison with DIFFERENT privacy mechanisms, not isolated policy superiority',
        rules_scope='NoPrivacy and Local are explicitly non-private references, not guaranteed worst cases',
        recent_scope='SD-SAC and MARATD3 architecture adaptations, not author-code reproductions',
        selection='All development outcomes retained. A best single seed may be labeled exploratory only; no seed or input chosen to force method order',
        next_stage='No formal scale-up until this correctness gate is verified; formal validation needs new disjoint seeds fixed before training')

"""One finite exploratory training/scale-up pilot; no formal-test claims."""
import hashlib
from pathlib import Path
from review_protocol import ABLATIONS, DEVELOPMENT_SEEDS, DEVELOPMENT_TRACES, LEARNERS, POPULATIONS, RULES, settings as reviewed_settings

CAMPAIGN = 'baseline-isolation-pilot-20261119'
BRANCH = 'experiment/' + CAMPAIGN
BASE_COMMIT = 'ccd981cb30c2e0763b7a41816923757f74f4a53e'
CONFIGS = LEARNERS + tuple('TMAPO-' + a for a in ABLATIONS)
CELLS = tuple(m + '_seed' + str(s) for m in CONFIGS for s in DEVELOPMENT_SEEDS) + RULES


def settings(label, seed, n):
    variant = label[len('TMAPO-'):] if label.startswith('TMAPO-') else None
    method = 'TMAPO' if variant else label
    c, t = reviewed_settings(method, seed, n, variant)
    if method == 'TMAPO':
        t.update(actor_lr=3e-4, critic_lr=1e-3, gae_lambda=.995, entropy=.001)
    return method, variant, c, t


def plan():
    root = Path(__file__).resolve().parent
    return dict(schema='finite-isolated-baseline-exploratory-pilot-v1', source_base=BASE_COMMIT,
        cells=CELLS, development_seeds=DEVELOPMENT_SEEDS, development_traces=DEVELOPMENT_TRACES,
        populations=POPULATIONS, train_population=150, admission_slots=4800,
        maximum_drain_slots=203, update_block_slots=100, maximum_training_updates=51,
        learning_rate_schedule_horizon=1208, evaluation_warmup_s=5., evaluation_measure_s=30.,
        evaluation_phase_s=0., maximum_wall_seconds_per_cell=5100, maximum_parallel=8,
        finite_rounds=1, paid=False, gpu=False, formal=False,
        review_code_sha256={n: hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for n in ('review_protocol.py','review_environment.py','review_checks.py','review_worker.py')},
        privacy_scope='System bundles: TMAPO bounded amplified/shrunk reporter versus individual ordinary-GeoI baseline reporters. NoPrivacy/Local explicitly have no privacy guarantee.',
        fairness='All learned controllers freshly initialized, same admission slots and natural drainage; actual steps and optimizer updates reported. TMAPO settings chosen in earlier development, not equal tuning budget.',
        ablations='NoForecastFeatures, NoWait, NoLocalCredit match the full TMAPO optimizer settings; each is freshly retrained, not a renamed checkpoint.',
        limitation='30-second evaluation is a short exploratory window, not a full 300-second tide or a formal result. No convergence or ordered-ranking guarantee.',
        seed_policy='Keep both fixed development seeds. Best single-seed observation may be labeled exploratory, never replace the two-seed aggregate.',
        next_stage='Stop after this one finite pilot. No automatic rerun, ranking-triggered data change, or formal training. New held-out seeds must be preregistered before a later formal stage.')

"""Fixed development-only pressure probe; the held-out scale-up is untouched."""
import hashlib
from pathlib import Path
from review_protocol import LEARNERS, RULES, DEVELOPMENT_SEEDS, digest

CAMPAIGN = 'eight-controller-pressure-probe-20261119'
BRANCH = 'experiment/' + CAMPAIGN
BASE_COMMIT = 'b28d915c157cb3606051063f887e1b962a2ec458'
PRIOR_CAMPAIGN = 'baseline-isolation-pilot-20261119'
PRIOR_RUN = 44394509688
PRIOR_RELEASE = 450537009
PRIOR_INPUTS = '278c201771835e8063cabc614847159b7a8b8788373becb35200d75cdb746fe2'
SEEDS = (1196711, 1196719, 1196731)
POPULATIONS = (20, 80, 150)
WARMUP, MEASURE, PHASE = 1., 3., 30.
FACTORS = {'reference': (1., 1.), 'arrivals_x1.25': (1.25, 1.),
    'remote_cpu_x0.65': (1., .65), 'combined': (1.25, .65)}
CELLS = tuple(f'{m}_seed{s}' for m in LEARNERS for s in DEVELOPMENT_SEEDS) + RULES
WALL_S = 3200
INHERITED = ('pilot_protocol.py', 'review_protocol.py', 'review_environment.py', 'review_checks.py')


def plan():
    root = Path(__file__).resolve().parent
    return dict(schema='eight-controller-development-probe-v1', cells=CELLS,
        inherited_code={n: hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in INHERITED},
        policies=dict(run=PRIOR_RUN, commit=BASE_COMMIT, inputs=PRIOR_INPUTS, retraining=False),
        seeds=SEEDS, representative_trace=SEEDS[1], sample_selection='Fixed illustrative seed before outcomes; all kept in main summaries.',
        populations=POPULATIONS, factors=FACTORS, warmup_s=WARMUP, measure_s=MEASURE, phase_s=PHASE,
        only_development=True, closed_formal_tests_untouched=True, max_parallel=4, wall_s=WALL_S,
        result='All scenes with per-profile completion and channel/workload statistics; no model export.',
        limitations='Shared 2x2 scenario changes, short phase window, fixed pilot policies and no rank-based selection.')

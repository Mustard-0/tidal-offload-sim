"""One bounded development screen for route-only physical logit guidance."""
import hashlib
from pathlib import Path
from statistics import mean

from optimization2_selection import aggregate
from review_protocol import DEVELOPMENT_SEEDS, digest

CAMPAIGN = 'tt-mappo-route-residual-development-20261120'
BRANCH = 'experiment/' + CAMPAIGN
BASE_COMMIT = 'd7491d5be11d7737c4a4320bae51877189b7b36c'
VARIANTS = {'control': 0., 'route025': .25, 'route100': 1.}
TRACES = (1198701, 1198709)
POPULATIONS = (40,100,150)
PHASES = (30.,150.)
WARMUP, MEASURE = 5., 10.
CELLS = tuple(f'{v}_seed{s}' for v in VARIANTS for s in DEVELOPMENT_SEEDS)
EXPECTED = {(n,t,p) for n in POPULATIONS for t in TRACES for p in PHASES}
LIMIT = 3500
INHERITED = ('optimization2_worker.py','optimization2_core.py','optimization2_selection.py',
    'optimization2_protocol.py','review_environment.py','review_checks.py','review_protocol.py',
    'pilot_protocol.py','pilot_worker.py','scalability_worker.py','scalability_protocol.py')


def plan():
    root = Path(__file__).resolve().parent
    return dict(schema='route-only-development-v1', source_base=BASE_COMMIT, cells=CELLS,
        inherited_review_hashes={n: hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in INHERITED},
        variants=VARIANTS, weights='Only the two authenticated full35 endpoints from run 44466767988',
        traces=TRACES, seeds=DEVELOPMENT_SEEDS, populations=POPULATIONS, phases=PHASES,
        warmup_s=WARMUP, measure_s=MEASURE, unchanged='Training, weights, waiting, privacy, physics, deadlines and reward',
        route='Subtract scale times the existing public physical route-cost estimate before legal-mask argmax',
        minimum_mean_cost_gain=.02, maximum_per_population_qos_increase=.005,
        maximum_per_seed_completion_decrease=.002, wall_s=LIMIT, parallel=3,
        final_test_used=False, retrain=False, auto_expand=False)


def select(cells):
    missing = [c for c in CELLS if not cells.get(c,{}).get('completed')]
    if missing:
        return dict(complete=False, missing=missing, selected=None, final_test_used=False)
    signatures, model_hashes, stats, by_n = {}, {}, {}, {}
    for c in CELLS:
        value = cells[c]
        seed = int(c.rsplit('_seed',1)[1])
        model = value['model_hash']
        assert model and model_hashes.setdefault(seed,model) == model
        rows = value['rows']
        assert len(rows)==len(EXPECTED) and {(r['users'],r['trace_seed'],r['phase_s']) for r in rows}==EXPECTED
        for r in rows:
            key = (r['users'],r['trace_seed'],r['phase_s'])
            fingerprint = (r['full_input_hash'],r['arrivals'])
            assert signatures.setdefault(key,fingerprint)==fingerprint
            assert r['number_queued_pending']==0
        stats[c] = aggregate(rows)
        by_n[c] = {n:aggregate([r for r in rows if r['users']==n]) for n in POPULATIONS}
    checks = []
    control_mean = mean(stats[f'control_seed{s}']['cost'] for s in DEVELOPMENT_SEEDS)
    criteria = plan()
    for variant in tuple(VARIANTS)[1:]:
        pairs = []
        for s in DEVELOPMENT_SEEDS:
            b,k = f'control_seed{s}',f'{variant}_seed{s}'
            prior,new = stats[b],stats[k]
            qos_diff = max(by_n[k][n]['qos']-by_n[b][n]['qos'] for n in POPULATIONS)
            okay = new['cost']<prior['cost'] and all(new[m] is not None and prior[m] is not None and
                new[m]<=prior[m]+1e-12 for m in ('qos','energy','delay'))
            okay &= new['completion']>=prior['completion']-criteria['maximum_per_seed_completion_decrease']
            okay &= qos_diff<=criteria['maximum_per_population_qos_increase']
            pairs.append(dict(seed=s,passed=okay,worst_population_qos_delta=qos_diff,control=prior,candidate=new))
        gain=(control_mean-mean(p['candidate']['cost'] for p in pairs))/control_mean
        checks.append(dict(variant=variant,cost_gain=gain,passed=all(p['passed'] for p in pairs) and
            gain>=criteria['minimum_mean_cost_gain'],per_seed=pairs))
    eligible=[c for c in checks if c['passed']]
    return dict(complete=True, final_test_used=False, paired_cohorts=len(signatures),
        selected=max(eligible,key=lambda c:c['cost_gain'])['variant'] if eligible else None,
        checks=checks, by_cell=stats)

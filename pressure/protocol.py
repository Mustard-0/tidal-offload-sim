"""Declared before any stress scores are observed."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plan():
    return dict(schema='greedy-common-pressure-discovery-v1',
        frozen_source_commit='0d1553cb789c399f3186cb1327d143684060032e',
        source_hashes_hash='8033aa8d9f8f528455f770a915766901b53cdb098ebab59b39e9d4d3ab0850e2',
        methods=['Greedy', 'GeoI-Greedy'], populations=[100, 150],
        arrival_rates_per_user_s=[3., 6., 9.], traces=[7298101, 7298109],
        scenes=[dict(name='short_empty', warmup_s=0., measure_s=1.),
                dict(name='long_cycle', warmup_s=5., measure_s=300.)],
        phase_s=0., dt_s=.01, natural_drain=True,
        cells=[dict(id=f'{method}_N{n}_q{q}', method=method, users=n, rate=q)
               for method in ['Greedy', 'GeoI-Greedy'] for n in [100, 150] for q in [3, 6, 9]],
        changed=['arrival intensity, identically for all methods', 'measurement duration and initial warmup'],
        unchanged=['task size/deadline/lifetime', 'CPU/bandwidth/coverage/energy model',
                   'queue capacities', 'reward', 'privacy mechanism', 'original strong Greedy controller'],
        distinction='Short vs long changes both warmup and duration, NOT a pure duration-only causal estimate.',
        purpose='Find common load regimes where myopic controllers have limitations; no guaranteed ranking.',
        selection='Retain EVERY cell. This is environment-development data, NOT final independent test evidence.',
        future='Any selected stress environment must be shared by every algorithm. Use NEW untouched evaluation traces if training later.',
        budget=dict(maximum_cells=12, maximum_parallel=8, maximum_seconds_per_worker=5400,
                    rerun=False, training=False, paid=False, gpu=False))

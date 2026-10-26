"""Focused existing task-conservation audit, without old matrix entry points."""
import math
from experiments.lr_screen.runtime import np


def audit_tasks(env,m):
    c=env.cfg
    if len(env.rows)!=m['arrivals'] or len({r['task_id'] for r in env.rows})!=len(env.rows):
        raise AssertionError('Missing/duplicate cohort task')
    costs=[]
    for r in env.rows:
        energy=sum(r[k] for k in ('transmit_energy_j','compute_energy_j','report_energy_j'))
        if not math.isclose(energy,r['energy_j'],abs_tol=1e-9):raise AssertionError('Energy mismatch')
        value=c['weight_time']*(r['end_s']-r['arrival_s'])/c['time_ref_s']
        value+=c['weight_energy']*energy/c['energy_ref_j']+c['weight_violation']*r['qos']
        if r['published']:value+=c['weight_quality']*r['ql_m']/c['quality_ref_m']
        costs.append(value)
    if not math.isclose(float(np.mean(costs)),m['mean_task_cost'],abs_tol=1e-8):
        raise AssertionError('Physical cost mismatch')
    if m['arrivals']!=m['completed']+m['failed']:raise AssertionError('Task conservation')
    for b in env.batch_rows:
        ids=b['user_ids'].split(';')
        if len(set(ids))!=b['n_distinct'] or len(ids)!=len(set(ids)):
            raise AssertionError('Actual distinct user batch mismatch')


def pooled(rows,key):
    denominator='completed' if key=='delay_s' else 'published' if key in (
        'ql_m','report_weighted_batch_n','amplified_publication_ratio') else 'arrivals'
    rows=[r for r in rows if r.get(key) is not None]
    n=sum(r[denominator] for r in rows)
    return sum(r[key]*r[denominator] for r in rows)/n if n else None

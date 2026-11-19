"""Validation-only frozen multi-objective guard, preserving every fixed seed."""
import math
import statistics
from optimization2_protocol import (CELLS, DEVELOPMENT_SEEDS, VARIANTS, plan,
    CHECKPOINT_UPDATES, VALIDATION_PHASES, VALIDATION_POPULATIONS, VALIDATION_TRACES)


def aggregate(rows):
    def weighted(key, weight):
        values = [r for r in rows if r[weight] > 0]
        return sum(r[key]*r[weight] for r in values) / sum(r[weight] for r in values) if values else None
    return dict(cost=weighted('mean_task_cost','arrivals'),
        energy=weighted('energy_j_per_arrival','arrivals'), delay=weighted('delay_s','completed'),
        qos=weighted('qos_rate','arrivals'),
        completion=sum(r['completed'] for r in rows)/sum(r['arrivals'] for r in rows))


def select(cells):
    criteria = plan()['selection']
    summary, fingerprints = {}, {}
    for cell in CELLS:
        value = cells[cell]
        if not value['completed']:
            return dict(completed=False, selected_variant=None, stop_reason='finite budget incomplete',
                        incomplete_cells=[c for c in CELLS if not cells[c]['completed']])
        assert value['updates'] == 302 and value['env_steps'] == 30200 and value['pending_tasks'] == 0
        rr = value['validation_rows']
        assert len(rr) == 24
        assert {(r['update'],r['users'],r['trace_seed'],r['phase_s']) for r in rr} == {
            (u,n,t,p) for u in CHECKPOINT_UPDATES for n in VALIDATION_POPULATIONS
            for t in VALIDATION_TRACES for p in VALIDATION_PHASES}
        for r in rr:
            key = (r['users'],r['trace_seed'],r['phase_s'])
            assert fingerprints.setdefault(key, (r['full_input_hash'],r['arrivals'])) == (r['full_input_hash'],r['arrivals'])
            assert r['number_queued_pending'] == 0
            assert r['completed'] + r['failed'] == r['arrivals']
            assert math.isclose(r['energy_j_per_arrival'], sum(r[k+'_energy_j_per_arrival'] for k in ('transmit','compute','report')), rel_tol=1e-8, abs_tol=1e-12)
            assert math.isclose(r['qos_rate'],r['failed']/r['arrivals']+r['late_completed_rate'],abs_tol=1e-12)
            assert math.isclose(r['mean_task_cost'],.5*r['resolution_time_s_per_arrival']+.25*r['energy_j_per_arrival']/.24+r['qos_rate'],rel_tol=1e-8,abs_tol=1e-10)
        summary[cell] = {u: aggregate([r for r in rr if r['update']==u]) for u in (0,151,302)}
    checks = []
    for variant in VARIANTS:
        if variant == criteria['reference']:
            continue
        per_seed = []
        for s in DEVELOPMENT_SEEDS:
            candidate = summary[variant + '_seed' + str(s)]
            reference = summary[criteria['reference'] + '_seed' + str(s)][302]
            end, middle = candidate[302], candidate[151]
            ok = (end['cost'] < reference['cost'] and
                end['qos'] <= reference['qos'] + criteria['maximum_qos_increase'] and
                end['completion'] >= reference['completion'] - criteria['maximum_completion_decrease'] and
                end['energy'] <= reference['energy'] + 1e-12 and
                end['delay'] is not None and reference['delay'] is not None and
                end['delay'] <= reference['delay'] + 1e-12 and
                end['cost'] <= middle['cost'] * criteria['maximum_late_cost_ratio'])
            per_seed.append(dict(seed=s, eligible=ok, terminal=end, control_terminal=reference))
        reference_cost = statistics.mean(summary['full70_seed'+str(s)][302]['cost'] for s in DEVELOPMENT_SEEDS)
        candidate_cost = statistics.mean(summary[variant+'_seed'+str(s)][302]['cost'] for s in DEVELOPMENT_SEEDS)
        improvement = (reference_cost - candidate_cost) / reference_cost
        eligible = improvement >= criteria['minimum_pooled_cost_improvement'] and all(r['eligible'] for r in per_seed)
        checks.append(dict(variant=variant, eligible=eligible, mean_cost=candidate_cost,
            improvement_vs_control=improvement, per_seed=per_seed))
    eligible = [r for r in checks if r['eligible']]
    return dict(completed=True, final_test_used=False, all_fixed_seeds_retained=True,
        selected_variant=min(eligible,key=lambda r:r['mean_cost'])['variant'] if eligible else None,
        no_eligible_candidate_keeps_control=True, independent_formal_validation_still_required=True,
        checks=checks, summaries=summary, paired_validation_inputs=len(fingerprints))


def main():
    from optimization2_worker import Transport
    t = Transport()
    cells = {}
    for cell in CELLS:
        part = 2 if cell+'.part2.enc' in t.assets else 1
        cells[cell] = t.read_small(cell+'.part'+str(part)+'.enc', cell+'.part'+str(part))
    result = select(cells)
    t.small('selection.enc', result, 'selection')
    print('Validation-only decision encrypted and saved. Complete inputs:', result['completed'])


if __name__ == '__main__':
    main()


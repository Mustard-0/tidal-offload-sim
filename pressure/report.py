"""All declared loads and both controllers, never a ranking-selected subset."""
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .experiment import csv_write, json_write


def report(raw, out):
    rows = []
    for method in ('Greedy', 'GeoI-Greedy'):
        for n in (100,150):
            for rate in (3,6,9):
                for scene in ('short_empty','long_cycle'):
                    g = [r for r in raw if r['method']==method and int(r['users'])==n
                         and int(float(r['arrival_rate_per_user_s']))==rate and r['scene']==scene]
                    assert len(g)==2
                    a = sum(float(r['arrivals']) for r in g)
                    c = sum(float(r['completed']) for r in g)
                    values = dict(method=method, users=n, arrival_rate_per_user_s=rate, scene=scene,
                        arrivals=int(a), completed=int(c),
                        delay_ms=sum(float(r['delay_s'])*float(r['completed']) for r in g if float(r['completed'])>0)/c*1000 if c else None,
                        energy_j=sum(float(r['energy_j_per_arrival'])*float(r['arrivals']) for r in g)/a,
                        qos_percent=100*sum(float(r['qos_rate'])*float(r['arrivals']) for r in g)/a,
                        completion_percent=100*c/a,
                        queue_ms=sum(float(r['queue_ms'])*float(r['arrivals']) for r in g)/a,
                        remote_compute_percent=100*statistics.mean(float(r['remote_compute_utilization']) for r in g),
                        mean_task_cost=sum(float(r['mean_task_cost'])*float(r['arrivals']) for r in g)/a,
                        failed_percent=100*sum(float(r['failed']) for r in g)/a,
                        late_completed_percent=100*sum(float(r['late_completed_rate'])*float(r['arrivals']) for r in g)/a)
                    rows.append(values)
    csv_write(out/'physical_metrics.csv', rows)
    fig, axes = plt.subplots(2,3, figsize=(13,8), layout='constrained')
    for axrow,n in zip(axes,(100,150)):
        for ax,key,label in zip(axrow, ('delay_ms','energy_j','qos_percent'),
                ('Completed-task latency (ms)','System energy (J / arrival)','QoS failure (%)')):
            for method,color in (('Greedy','#26734e'),('GeoI-Greedy','#b57d1a')):
                for scene,style,title in (('short_empty','--','1s empty start'),('long_cycle','-','5s warmup +300s')):
                    g = [r for r in rows if r['method']==method and r['users']==n and r['scene']==scene]
                    ax.plot([r['arrival_rate_per_user_s'] for r in g], [r[key] for r in g],
                            style, marker='o', color=color, label=method+' | '+title)
            ax.set(xlabel='Arrivals per user per second', ylabel=label, title=f'N={n}')
            ax.set_xticks([3,6,9]);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    fig.suptitle('Common environment pressure discovery | unchanged CPUs, tasks, deadlines, energy and controllers')
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='outside lower center', ncol=2, fontsize=9)
    fig.savefig(out/'greedy_pressure.png',dpi=160)
    fig.savefig(out/'greedy_pressure.pdf');plt.close(fig)
    fig, axes = plt.subplots(1,3, figsize=(13,4.5), layout='constrained')
    for ax,key,label in zip(axes,('delay_ms','energy_j','qos_percent'),
        ('Completed-task latency (ms)','System energy (J / arrival)','QoS failure (%)')):
        for method,color in (('Greedy','#26734e'),('GeoI-Greedy','#b57d1a')):
            for rate,style in ((3,'-'),(6,'--'),(9,':')):
                g = [r for r in rows if r['method']==method and r['scene']=='long_cycle' and r['arrival_rate_per_user_s']==rate]
                ax.plot([r['users'] for r in g],[r[key] for r in g],style,marker='o',color=color,label=f'{method} | {rate}/s/user')
        ax.set(xlabel='Actual users',ylabel=label);ax.set_xticks([100,150]);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    fig.suptitle('Full-cycle long scenes ONLY | two fixed development traces | no learning performed')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(out/'long_scenes.png',dpi=160);plt.close(fig)
    json_write(out/'limitations.json',dict(development_not_final_test=True,
        stronger_load_may_hurt_every_method=True,no_claim_rl_will_win=True,
        delay_denominator='completed tasks; ALWAYS read together with failure and completion',
        qos='failed or completed beyond per-task deadline, divided by measured arrivals',
        energy='same original system dynamic compute+TX+report energy per measured arrival',
        no_prior_short_rl_curves_mixed=True))

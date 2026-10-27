"""Only observed physical metrics and episode returns; no ranking manipulation."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.lr_screen.runtime import csv_write,json_write
from .protocol import POPULATIONS


def rows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def report(folders,out):
    raw=[];returns=[]
    for folder in folders:
        done=json.loads((folder/'complete.json').read_text())
        r=rows(folder/'episode_returns.csv')
        if len(r)!=2000 or int(r[-1]['episode'])!=2000 or not done['all_tasks_resolved']:
            raise ValueError('Incomplete training budget')
        returns.extend([dict(method=done['method'],seed=done['seed'],**v) for v in r])
        raw+=rows(folder/'test_metrics.csv')
        if (folder/'rules_metrics.csv').exists():raw+=rows(folder/'rules_metrics.csv')
    methods=('TMAPO','IPPO','PPO','Greedy','GeoI-Greedy')
    # Pair the complete generated input trace, not only total arrival counts.
    pairs={}
    for r in raw:
        key=(r['users'],r['trace_seed'],r['phase_s'])
        sig=(r['trace_hash'],r['arrivals'])
        if key in pairs and pairs[key]!=sig:raise ValueError('Unpaired exogenous evaluation inputs')
        pairs[key]=sig
    summary=[]
    for mode in ('sampled','argmax'):
        for method in methods:
            for n in POPULATIONS:
                g=[r for r in raw if r['method']==method and int(r['users'])==n and r['mode'] in (mode,'rule')]
                expected=20 if method in ('TMAPO','IPPO','PPO') else 10
                if len(g)!=expected:raise ValueError('Incomplete evaluation grid')
                seed_values=[]
                for seed in sorted({r['training_seed'] for r in g}):
                    s=[r for r in g if r['training_seed']==seed]
                    a=sum(float(r['arrivals']) for r in s);c=sum(float(r['completed']) for r in s)
                    seed_values.append(dict(delay_ms=sum(float(r['delay_s'])*float(r['completed']) for r in s)/c*1000 if c else np.nan,
                        energy_j=sum(float(r['energy_j_per_arrival'])*float(r['arrivals']) for r in s)/a,
                        qos_percent=sum(float(r['qos_rate'])*float(r['arrivals']) for r in s)/a*100,
                        wait_ms=sum(float(r['active_wait_s'])*float(r['arrivals']) for r in s)/a*1000))
                item=dict(method=method,users=n,mode=mode,seeds=len(seed_values))
                for metric in seed_values[0]:
                    values=[r[metric] for r in seed_values]
                    item[metric]=float(np.mean(values));item[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
                summary.append(item)
    csv_write(out/'raw_scene_metrics.csv',raw);csv_write(out/'episode_returns.csv',returns)
    csv_write(out/'physical_metrics.csv',summary)
    colors=('#bc2e47','#227e9a','#7251a0','#38824c','#ca8a24')
    for mode in ('sampled','argmax'):
        fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
        for ax,key,label in zip(axes,('delay_ms','energy_j','qos_percent'),
            ('Completed-task latency (ms)','System energy (J / arrival)','QoS failure (%)')):
            for method,color in zip(methods,colors):
                g=[r for r in summary if r['method']==method and r['mode']==mode]
                x=np.array([r['users'] for r in g]);y=np.array([r[key] for r in g]);sd=np.array([r[key+'_sd'] for r in g])
                ax.plot(x,y,color=color,marker='o',ms=4,label=method);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.08)
            ax.set(xlabel='Actual total users',ylabel=label);ax.set_xticks(POPULATIONS);ax.grid(alpha=.2)
        fig.suptitle(f'Basic baselines vs guarded TMAPO | {mode} | same tasks/resources/privacy/reward')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=5)
        fig.savefig(out/f'scalability_{mode}.png',dpi=180);fig.savefig(out/f'scalability_{mode}.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for method,color in zip(methods[:3],colors):
        for seed in sorted({r['seed'] for r in returns if r['method']==method}):
            g=[r for r in returns if r['method']==method and r['seed']==seed]
            y=np.array([float(r['episode_return']) for r in g]);smooth=np.convolve(y,np.ones(50)/50,mode='valid')
            ax.plot(np.arange(50,len(y)+1),smooth,color=color,alpha=.8,label=f'{method} seed{seed}')
    ax.set(xlabel='Completed episode',ylabel='Episode return (MA50)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'returns_ma50.png',dpi=170);plt.close(fig)
    json_write(out/'complete.json',dict(completed=True,training_cells=6,episodes_per_cell=2000,
        evaluation_populations=POPULATIONS,paired_scenes=len(pairs),primary='sampled',
        secondary='argmax for every method',smoothing='returnMA50 only; raw kept; physical curves unsmoothed',
        uncertainty='two fixed training-seed SD, not CI',no_superiority_assumption=True))

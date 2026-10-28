"""Terminal comparison with every fixed seed, no test-time mode selection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .protocol import METHODS,SEEDS,POPULATIONS,plan,digest
from experiments.lr_screen.runtime import csv_write,json_write,read_csv


def report(folders,out):
    raw=[];training=[];validation=[];cycles=[]
    methods=(*METHODS,'Greedy','GeoI-Greedy')
    for folder in folders:
        done=json.loads((folder/'complete.json').read_text())
        assert done['completed'] and done['scenes']==14
        raw+=read_csv(folder/'test_metrics.csv')
        if done['method'] in METHODS:
            r=read_csv(folder/'training.csv')
            assert len(r)==1208 and int(r[-1]['env_steps'])==120800
            training.extend([dict(method=done['method'],seed=done['seed'],**x) for x in r])
            c=read_csv(folder/'cycles.csv');assert len(c)==4
            cycles.extend([dict(method=done['method'],seed=done['seed'],**x) for x in c])
            for file in folder.glob('validation_*.json'):
                v=json.loads(file.read_text())
                validation.append(dict(method=done['method'],seed=done['seed'],**{k:x for k,x in v.items() if k!='rows'}))
    assert len(raw)==168 and len(training)==12080 and len(cycles)==40 and len(validation)==50
    pairs={};trainpairs={}
    for r in raw:
        k=(r['users'],r['trace_seed']);sig=(r['full_input_hash'],r['arrivals'])
        assert pairs.setdefault(k,sig)==sig,'Different test task inputs'
        assert r['all_tasks_resolved']=='True'
    for r in cycles:
        k=(r['seed'],r['cycle']);sig=(r['full_input_hash'],r['arrivals'])
        assert trainpairs.setdefault(k,sig)==sig,'Different training task inputs'
    summary=[]
    for method in methods:
        for n in POPULATIONS:
            g=[r for r in raw if r['method']==method and int(r['users'])==n]
            assert len(g)==(4 if method in METHODS else 2)
            seed_values=[]
            for seed in sorted({r['training_seed'] for r in g}):
                s=[r for r in g if r['training_seed']==seed]
                a=sum(float(r['arrivals']) for r in s);c=sum(float(r['completed']) for r in s)
                seed_values.append(dict(delay_ms=sum(float(r['delay_s'])*float(r['completed']) for r in s if float(r['completed']))/c*1000 if c else np.nan,
                    energy_j=sum(float(r['energy_j_per_arrival'])*float(r['arrivals']) for r in s)/a,
                    qos_percent=100*sum(float(r['qos_rate'])*float(r['arrivals']) for r in s)/a,
                    completion_percent=100*c/a,
                    mean_task_cost=sum(float(r['mean_task_cost'])*float(r['arrivals']) for r in s)/a))
            item=dict(method=method,users=n,training_seeds=len(seed_values) if method in METHODS else 0)
            for key in seed_values[0]:
                v=[r[key] for r in seed_values];item[key]=float(np.mean(v));item[key+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else 0.
            summary.append(item)
    for name,rows in (('physical_metrics',summary),('scene_metrics',raw),('training',training),('cycles',cycles),('validation',validation)):
        csv_write(out/(name+'.csv'),rows)
    colors=('#bc2e47','#227e9a','#7251a0','#307f68','#ad6b31','#27843e','#b48c22')
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),layout='constrained')
    for ax,key,label in zip(axes,('delay_ms','energy_j','qos_percent'),
        ('Completed-task latency (ms)','System energy (J / arrival)','QoS failure (%)')):
        for method,color in zip(methods,colors):
            g=[r for r in summary if r['method']==method]
            x=np.array([r['users'] for r in g]);y=np.array([r[key] for r in g]);sd=np.array([r[key+'_sd'] for r in g])
            ax.plot(x,y,marker='o',ms=4,color=color,label=method)
            if method in METHODS:
                ax.fill_between(x,np.maximum(0.,y-sd),y+sd,color=color,alpha=.08)
        ax.set(xlabel='Actual users',ylabel=label);ax.set_xticks(POPULATIONS);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    fig.suptitle('Common moderate load: 6 arrivals/user/s | 5s warmup +300s | terminal policies | 2-seed SD, not CI')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=8)
    fig.savefig(out/'scalability.png',dpi=170);fig.savefig(out/'scalability.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    for method,color in zip(METHODS,colors):
        for seed in SEEDS:
            g=[r for r in training if r['method']==method and int(r['seed'])==seed]
            y=np.array([float(r['rollout_return']) for r in g]);smooth=np.convolve(y,np.ones(50)/50,mode='valid')
            ax.plot([int(r['env_steps']) for r in g][49:],smooth,color=color,
                    alpha=.85 if seed==SEEDS[0] else .5,linestyle='-' if seed==SEEDS[0] else '--',label=f'{method} seed{seed}')
    for x in (30200,60400,90600):
        ax.axvline(x,color='#888888',alpha=.25,lw=.7)
    ax.set(xlabel='Cumulative environment interactions',ylabel='100-slot return (MA50 updates, NOT episodes)')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.savefig(out/'returns_ma50_updates.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,key,label in zip(axes,('mean_task_cost','qos_rate'),('Fixed validation physical cost','Fixed validation QoS failure')):
        for method,color in zip(METHODS,colors):
            g=[r for r in validation if r['method']==method];xs=sorted({r['env_steps'] for r in g})
            ax.plot(xs,[np.mean([r[key] for r in g if r['env_steps']==x]) for x in xs],'o-',color=color,label=method)
        ax.set(xlabel='Environment interactions',ylabel=label);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(out/'fixed_validation.png',dpi=170);plt.close(fig)
    json_write(out/'complete.json',dict(completed=True,training_cells=10,env_steps_per_cell=120800,
        rows=12080,cycles=40,test_scenes=168,paired_test_groups=len(pairs),paired_training_cycle_groups=len(trainpairs),
        no_seed_selection=True,no_test_tuning=True,plan=plan(),plan_hash=digest(plan()),
        limitations='2 fixed seeds; architecture-adapted recent methods; SD not CI; batch privacy not trajectory DP; no superiority/convergence guarantee'))

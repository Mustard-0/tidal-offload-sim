"""One raw curve from actual completed, naturally drained episode records."""
import csv
from pathlib import Path


def plot_cell(folder,label):
    folder=Path(folder)
    with (folder/'episode_returns.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    if not rows or any(int(r['episode'])!=i+1 or int(r['arrival_steps'])!=100
                       or int(r['arrivals'])!=int(r['ended']) for i,r in enumerate(rows)):
        raise ValueError('Missing or undrained episode evidence')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.plot([int(r['episode']) for r in rows],[float(r['episode_return']) for r in rows],lw=.8)
    ax.axvline(2000,color='#495057',ls='--',lw=1,label='Budget extension at2000')
    ax.legend()
    ax.set(xlabel='Completed episode',ylabel='Episode return (higher is better)',
        title=label+' | actual N=100 (34/33/33), seed87021')
    ax.grid(alpha=.2)
    fig.text(.5,.02,'100 arrival slots per episode + natural drainage; all tail costs included. Raw, no smoothing.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(folder/'episode_return.png',dpi=160)
    plt.close(fig)

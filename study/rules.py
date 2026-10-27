"""The two untrained reference controllers, evaluated without TMAPO modules."""
from .protocol import configuration,POPULATIONS,TEST_SEEDS,PHASES
from .environment import Environment
from .geoi import GeoIEnvironment
from .runner import episode
from experiments.lr_screen.runtime import csv_write,json_write
from environment import GreedyPolicy


def evaluate_rules(out):
    rows=[]
    for method,cls in (('Greedy',Environment),('GeoI-Greedy',GeoIEnvironment)):
        for n in POPULATIONS:
            cfg,_=configuration('TMAPO',87021,n)
            for trace in TEST_SEEDS:
                for phase in PHASES:
                    e=cls(cfg,seed=trace,phase_s=phase,measure_end=1.,family=method,collect=True)
                    e.arrival_stop_slot=100
                    episode(e,GreedyPolicy(cfg))
                    m=e.metrics()
                    folder=out/'rule_evaluation'/f'{method}_N{n}_t{trace}_p{int(phase)}'
                    folder.mkdir(parents=True,exist_ok=False)
                    csv_write(folder/'tasks.csv',e.rows);csv_write(folder/'batches.csv',e.batch_rows)
                    json_write(folder/'metrics.json',m)
                    rows.append(dict(method=method,training_seed=0,users=n,mode='rule',
                                     trace_seed=trace,phase_s=phase,**m))
    csv_write(out/'rules_metrics.csv',rows)

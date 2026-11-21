"""Measure genuine deployment edits against the frozen ten-user grid."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import population_grid_protocol as grid


BASE='45b986a9485a84d3dd96e4eb26400f85153827bb'
PARENT_RUN=44925862192
WORKFLOW='.github/workflows/tt_mappo_dispatch_trial_20261122.yml'
POPS={'mid':tuple(range(20,101,10)),'high':tuple(range(110,151,10))}
BRANCHES={k:'experiment/tt-mappo-dispatch-'+k+'-20261122' for k in POPS}
INHERITED=grid.INHERITED+('population_grid_protocol.py','population_grid_worker.py')
MIN_ENERGY_GAIN=.01
MAX_POINT_COMPLETION_LOSS=.005
MAX_POINT_QOS_INCREASE=.005
MAX_TOTAL_RESOLUTION_INCREASE=.005


def stage(name):
    prior=grid.stage(name)
    branch=BRANCHES[name]
    campaign=branch.split('/',1)[1]
    pops=POPS[name]
    cases=tuple(('original',n,t,p) for n in pops for t,p in prior.TRACES)

    def plan(project=None,overlay=None):
        root=Path(project) if project else Path(__file__).resolve().parent
        def source(p):
            f=Path(overlay)/p if overlay else root/p
            return f if f.is_file() else root/p
        return dict(schema='paired-physical-dispatch-v1',stage=name,parent=BASE,anchor_run=PARENT_RUN,
            cases=cases,arms=prior.ARMS,seed=prior.SEED,mix=prior.MIXES,warm=prior.WARM,
            measure=prior.MEASURE,ratios=prior.RATIOS,wall_s=3600,train=False,gpu=False,paid=False,
            changes='Only replace routes predicted better in time and energy. '
                    'Non-preemptive remote queue compare FIFO, EDF, SPT, minimize predicted deadline misses then time.',
            eligible=dict(min_energy_gain=MIN_ENERGY_GAIN,max_completion_loss=MAX_POINT_COMPLETION_LOSS,
                          max_qos_increase=MAX_POINT_QOS_INCREASE,
                          max_resolution_increase=MAX_TOTAL_RESOLUTION_INCREASE),
            inherited_review_hashes={p:hashlib.sha256(source(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                     for p in INHERITED})

    def read_packet(release,commit):
        assert release['tag_name']==campaign
        return prior.read_packet(dict(release,tag_name=prior.CAMPAIGN),commit)

    return SimpleNamespace(**{**vars(prior),'CAMPAIGN':campaign,'BRANCH':branch,'WORKFLOW':WORKFLOW,'BASE':BASE,
        'POPS':pops,'CASES':cases,'INHERITED':INHERITED,'plan':plan,'read_packet':read_packet})

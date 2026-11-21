"""Eight-method main comparison over all fourteen load settings."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import dispatch_trial_protocol as prior


BASE='3368d0e83873b48f8f480ea7879595fd72e36240'
PARENT_RUN=44936553892
WORKFLOW='.github/workflows/tt_mappo_rescue_main_20261122.yml'
BRANCHES={k:f'experiment/tt-mappo-rescue-main-v2-{k}-20261122' for k in ('mid','high')}
INHERITED=prior.INHERITED+('physical_dispatch.py','dispatch_trial_protocol.py','dispatch_trial_worker.py',
                          'dispatch_component_protocol.py','dispatch_component_worker.py')


def stage(name):
    inherited=prior.stage(name)
    branch=BRANCHES[name]
    campaign=branch.split('/',1)[1]

    def plan(project=None,overlay=None):
        root=Path(project) if project else Path(__file__).resolve().parent
        def source(p):
            f=Path(overlay)/p if overlay else root/p
            return f if f.is_file() else root/p
        return dict(schema='main-deadline-rescue-v2',stage=name,parent=BASE,anchor=PARENT_RUN,
            cases=inherited.CASES,arms=inherited.ARMS,seed=inherited.SEED,mix=inherited.MIXES,
            warmup=inherited.WARM,window=inherited.MEASURE,ratios=inherited.RATIOS,wall_s=inherited.WALL,
            train=False,gpu=False,paid=False,parallel_per_batch=6,
            rescue='Retain original choice if predicted to fit and finish on time. '
                   'Otherwise replace only with a visible, capacity-feasible route projected on time, faster and no more expensive.',
            queue='Previously verified non-preemptive remote FIFO/EDF/SPT selection. No changes to Local-Only.',
            compare='Fixed full 20-150 exploratory grid; preserve every completed case and original controls, regardless of direction.',
            inherited_review_hashes={p:hashlib.sha256(source(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                     for p in INHERITED})

    def read_packet(release,commit):
        assert release['tag_name']==campaign
        return inherited.read_packet(dict(release,tag_name=inherited.CAMPAIGN),commit)

    return SimpleNamespace(**{**vars(inherited),'CAMPAIGN':campaign,'BRANCH':branch,'BASE':BASE,
        'WORKFLOW':WORKFLOW,'INHERITED':INHERITED,'plan':plan,'read_packet':read_packet})

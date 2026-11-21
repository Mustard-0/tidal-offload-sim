"""Split the dispatcher into two independent, fixed-input candidates."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import dispatch_trial_protocol as previous


BASE='2681e79bd53e53bcaf400027fa3d705063683c63'
PARENT_RUN=44927521799
WORKFLOW='.github/workflows/tt_mappo_dispatch_components_20261122.yml'
COMPONENTS=('route','queue')
BRANCHES={(part,span):f'experiment/tt-mappo-dispatch-{part}-{span}-20261122'
          for part in COMPONENTS for span in ('mid',)}
INHERITED=previous.INHERITED+('dispatch_trial_protocol.py','dispatch_trial_worker.py','physical_dispatch.py')


def stage(part,span):
    prior=previous.stage(span)
    branch=BRANCHES[part,span]
    campaign=branch.split('/',1)[1]
    arms=tuple((m,a) for m,a in prior.ARMS if m!='Local-Only')

    def plan(project=None,overlay=None):
        root=Path(project) if project else Path(__file__).resolve().parent
        def source(path):
            local=Path(overlay)/path if overlay else root/path
            return local if local.is_file() else root/path
        return dict(schema='single-dispatch-component-v1',stage=span,component=part,parent=BASE,anchor=PARENT_RUN,
            cases=prior.CASES,arms=arms,seed=prior.SEED,mix=prior.MIXES,warm=prior.WARM,measure=prior.MEASURE,
            ratios=prior.RATIOS,wall_s=prior.WALL,train=False,gpu=False,paid=False,parallel=6,
            local_only='No-op for either remote change. Reuse previously verified Local-Only test.',
            gate='Require the existing energy and completion guard, and no more near-ties in summed latency.',
            inherited_review_hashes={p:hashlib.sha256(source(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                     for p in INHERITED})

    def read_packet(release,commit):
        assert release['tag_name']==campaign
        return prior.read_packet(dict(release,tag_name=prior.CAMPAIGN),commit)

    return SimpleNamespace(**{**vars(prior),'CAMPAIGN':campaign,'BRANCH':branch,'WORKFLOW':WORKFLOW,'BASE':BASE,
        'ARMS':arms,'INHERITED':INHERITED,'plan':plan,'read_packet':read_packet})

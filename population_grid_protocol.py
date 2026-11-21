"""Ten-user evaluation grid continuing the verified original workload."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import lowload_protocol as low
from focus_protocol import POPS as EXISTING_POPS


BASE='4102110e0f8720f7a995649ae9a3b16747b49cf0'
PARENT_RUN=44924091627
WORKFLOW='.github/workflows/tt_mappo_population_grid_20261122.yml'
POPS={'mid':tuple(range(50,101,10)),'high':tuple(range(110,151,10))}
BRANCHES={k:'experiment/tt-mappo-grid-'+k+'-20261122' for k in POPS}
RATIOS=low.RATIOS
REPORT_RATIO=40
NEAR_QOS=.005
MAX_NEAR_PAIRS=7  # At most one third of the 21 offloading pairs.
INHERITED=low.INHERITED+('lowload_protocol.py','lowload_worker.py')


def settings(method,seed,n):
    assert n in sum(POPS.values(),())+low.POPS
    if n in EXISTING_POPS or n==30:
        return low.settings(method,seed,n)
    cfg,train=low.settings(method,seed,20)
    counts=[n//3+int(i<n%3) for i in range(3)]
    return dict(cfg,users_per_region=max(counts),region_user_counts=counts),train


def stage(name):
    pops=POPS[name]
    branch=BRANCHES[name]
    campaign=branch.split('/',1)[1]
    mixes={'original':low.MIXES['original']}
    cases=tuple(('original',n,t,p) for n in pops for t,p in low.TRACES)

    def plan(project=None,overlay=None):
        root=Path(project) if project else Path(__file__).resolve().parent
        def source(p):
            local=Path(overlay)/p if overlay else root/p
            return local if local.is_file() else root/p
        return dict(schema='ten-user-population-grid-v1',stage=name,parent=BASE,anchor_run=PARENT_RUN,
            cases=cases,arms=low.ARMS,seed=low.SEED,mix=mixes,measure=low.MEASURE,warmup=low.WARM,
            ratios=RATIOS,report_ratio=REPORT_RATIO,wall_s=3600,train=False,gpu=False,paid=False,
            gate=dict(maximum_near_pairs=MAX_NEAR_PAIRS,near_qos=NEAR_QOS,
                unit='Proportion at the 40 percent sensitivity cut; seven offloading methods',
                require='20 to 100 complete, no non-increasing arrivals or falling energy per second, '
                        'report physical reasons for downward unit-energy segments'),
            inherited_review_hashes={p:hashlib.sha256(source(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                     for p in INHERITED})

    def read_packet(release,commit):
        assert release['tag_name']==campaign
        return low.read_packet(dict(release,tag_name=low.CAMPAIGN),commit)

    return SimpleNamespace(CAMPAIGN=campaign,BRANCH=branch,WORKFLOW=WORKFLOW,BASE=BASE,POPS=pops,
        RULES=low.RULES,SEED=low.SEED,MIXES=mixes,TRACES=low.TRACES,CASES=cases,ARMS=low.ARMS,
        RATIOS=RATIOS,WARM=low.WARM,MEASURE=low.MEASURE,WALL=3600,INHERITED=INHERITED,
        digest=low.digest,plan=plan,settings=settings,settings_for_mix=low.settings_for_mix,
        packet_body=low.packet_body,read_packet=read_packet)

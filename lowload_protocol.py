"""Low-load curve diagnostics, independent from the full population screen."""

import mixscreen_protocol as mix
from mixscreen_protocol import digest, RULES, WARM, MIXES, settings_for_mix, PROFILES
from scalability_protocol import settings as _settings


CAMPAIGN='tt-mappo-lowload-shape-20261122'
BRANCH='experiment/'+CAMPAIGN
WORKFLOW='.github/workflows/tt_mappo_lowload_shape_20261122.yml'
BASE='1b5b44218d59da8fba115b0f07e13f4ded8d7bc0'
PARENT_RUN=44922912974
SEED=mix.SEED
POPS=(20,30,40)
TRACES=((1210221,30.),(1210229,150.))
RATIOS=(100,95,90,85,80,75,70,60,50,40,30,25)
MEASURE=5.
WALL=2000
ARMS=tuple(arm for arm in mix.ARMS if arm!=('TMAPO','admission'))
CASES=tuple((k,n,t,p) for n in POPS for k in MIXES for t,p in TRACES)
INHERITED=mix.INHERITED+('mixscreen_protocol.py','mixscreen_admission.py','mixscreen_worker.py')


def settings(method,seed,n):
    cfg,train=_settings(method,seed,20 if n==30 else n)
    if n==30:
        cfg=dict(cfg,region_user_counts=[10,10,10],users_per_region=10)
    return cfg,train


def plan(project=None,overlay=None):
    import hashlib
    from pathlib import Path
    root=Path(project) if project else Path(__file__).resolve().parent
    def source(p):
        local=Path(overlay)/p if overlay else root/p
        return local if local.is_file() else root/p
    return dict(schema='lowload-shape-v1',parent=BASE,arms=ARMS,cases=CASES,seed=SEED,
        mixes=MIXES,profile_order=PROFILES,measure=MEASURE,warmup=WARM,wall_s=WALL,
        qos_deadline_percentages=RATIOS,train=False,gpu=False,paid=False,
        receipt_rule='Publish a compact authenticated subtotal after each population.',
        deadline_rule='Same relative thresholds count failures everywhere. Reporting only. '
                      'No policy, task deadlines, weights, reward or privacy changes.',
        inherited_review_hashes={p:hashlib.sha256(source(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in INHERITED})


def packet_body(packet,commit):
    return mix.packet_body(packet,commit)


def read_packet(release,commit):
    assert release['tag_name']==CAMPAIGN
    return mix.read_packet(dict(release,tag_name=mix.CAMPAIGN),commit)

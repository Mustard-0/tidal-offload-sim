"""Three deployment configurations within one exploratory TT-MAPPO seed."""

from base64 import b64decode,b64encode
import hashlib
import json
from pathlib import Path

from review_protocol import digest


CAMPAIGN = 'tt-mappo-seed871901-capacity-focus-20261121'
BRANCH = 'experiment/'+CAMPAIGN
WORKFLOW = '.github/workflows/tt_mappo_capacity_focus_20261121.yml'
BASE = 'cc24e6e6709f31383db5f8e526794dd933d05255'
SEED = 871901
VARIANTS = ('control','route100','route100_local')
WARM = 5.
WALL = 2900
POPS = (20,40,60,80,100,125,150)
DIAGNOSTIC = tuple(('diagnostic',n,t,p,12.) for n in (125,150) for t in (1202111,1202119) for p in (30.,150.))
FRESH = tuple(('fresh',n,t,p,5.) for n in POPS for t,p in ((1204051,30.),(1204059,150.)))
CASES = DIAGNOSTIC+FRESH
INHERITED = ('energy_qos_worker.py','energy_qos_protocol.py','energy_qos_telemetry.py',
             'evaluation_adapters.py','review_environment.py','review_checks.py','review_protocol.py',
             'pilot_protocol.py','pilot_worker.py','scalability_worker.py','scalability_protocol.py',
             'optimization2_worker.py','optimization2_protocol.py','optimization2_core.py')
MARKER = 'STIN encrypted configuration: '


def plan(root=None,overlay=None):
    root=Path(__file__).resolve().parent if root is None else Path(root)
    def source(p):
        candidate=Path(overlay)/p if overlay is not None else root/p
        return candidate if candidate.is_file() else root/p
    return dict(schema='single-training-seed-capacity-v1',seed=SEED, variants=VARIANTS,
        cases=CASES, warmup_s=WARM, wall_s=WALL, stage_budget_hours=24, jobs=3,
        scope='Same frozen TT-MAPPO full35; no training. Same model, tasks, resources, deadline, privacy and reward.',
        route_scale=dict(control=0.,route100=1.,route100_local=1.),
        local_fallback_only='Compute-heavy, simultaneous all-remote capacity screen and feasible local deadline.',
        choose='Lowest fresh mean cost with improvement over control and no population QoS rise over 0.5 pp. '
               'Use complete fresh paired cases only. Diagnose on matching old high-load windows separately.',
        inherited_review_hashes={p:hashlib.sha256(source(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in INHERITED})


def encode(packet,commit):
    return ('Single seed exploratory evaluation; up to three CPU jobs. No training.\n\n'+
        MARKER+json.dumps(dict(sha=hashlib.sha256(packet).hexdigest(),commit=commit,
                               packet=b64encode(packet).decode()),separators=(',',':')))


def decode(release,commit):
    assert release['tag_name']==CAMPAIGN and release['target_commitish']==commit
    assert len(release['body'])<32768
    line,=[s[len(MARKER):] for s in release['body'].splitlines() if s.startswith(MARKER)]
    data=json.loads(line)
    cipher=b64decode(data['packet'],validate=True)
    assert 0<len(cipher)<16384 and data['commit']==commit and data['sha']==hashlib.sha256(cipher).hexdigest()
    return cipher

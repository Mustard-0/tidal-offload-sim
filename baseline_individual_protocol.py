"""Fixed one-seed screen with an independent window pair for all methods."""

from base64 import b64decode,b64encode
import hashlib
import json
from pathlib import Path

from focus_protocol import POPS,SEED,WARM
from review_protocol import digest,LEARNERS
from scalability_protocol import RULES


CAMPAIGN='baseline-individual-rework-20261121'
BRANCH='experiment/'+CAMPAIGN
WORKFLOW='.github/workflows/baseline_individual_rework_20261121.yml'
BASE='81e08973e7dd815367688ea557b6e8ee4d5ecb98'
PARENT_RUN=44916636684
BASELINES=LEARNERS[1:]+RULES
ARMS=tuple((m,a) for m in BASELINES for a in ('control','targeted'))+(('TMAPO','reference'),)
CASES=tuple((n,t,p) for n in POPS for t,p in ((1206121,30.),(1206129,150.)))
MEASURE=5.
WALL=3200
MARKER='STIN encrypted configuration: '
INHERITED=('focus_protocol.py','focus_fallback.py','energy_qos_worker.py','energy_qos_protocol.py',
           'energy_qos_telemetry.py','evaluation_adapters.py','review_checks.py','review_environment.py',
           'review_protocol.py','pilot_protocol.py','pilot_worker.py','scalability_worker.py',
           'scalability_protocol.py','optimization2_worker.py','optimization2_protocol.py','optimization2_core.py',
           'local_sched_protocol.py','local_sched_worker.py')
CRITERIA=dict(qos_max_pp=.5,strict_qos_max_pp=.5,completion_loss_max_pp=.5,class_completion_loss_max_pp=2.,
              mean_latency_regression_s=.001,mean_energy_regression_j=.001,min_mean_cost_gain=0.)


def plan(project=None,overlay=None):
    root=Path(project) if project else Path(__file__).resolve().parent
    def source(name):
        candidate=Path(overlay)/name if overlay else root/name
        return candidate if candidate.is_file() else root/name
    return dict(schema='individual-baseline-development-v1',parent=BASE,seed=SEED,cases=CASES,arms=ARMS,
        warmup=WARM,measure=MEASURE,wall_seconds=WALL,maximum_parallel=6,total_scenes=len(CASES)*len(ARMS),
        methods={
            'IPPO':'Compare upload estimates within 10ms of fastest on-time route; 0.005J upload band.',
            'PPO':'When all legal queues overflow, prefer the smallest service-time shortfall; 5ms band.',
            'SD-SAC-Adapted':'Protect last 20 percent satellite headroom if an on-time alternative remains.',
            'MARATD3-Adapted':'Use on-time ground option for comparable 5ms latency and >=5mJ energy gain.',
            'Greedy-NoPrivacy':'Capacity-filtered public quote plus previous shared offload guard.',
            'Greedy-GeoI':'Same quote and guard as otherwise identical NoPrivacy pair.',
            'Local-Only':'EDF versus choose EDF/SPT waiting order by predicted on-time count, then delay.',
            'TMAPO':'Existing frozen route100 plus local heavy fallback, reference only.'},
        baseline_control='The accepted previous guard; EDF for Local-Only.',selection=CRITERIA,
        keep_originals=True,train=False,paid=False,gpu=False,
        independent_screen='Only new development traces. No blind-test access, no forced trend.',
        inherited_review_hashes={p:hashlib.sha256(source(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in INHERITED})


def packet_body(packet,commit):
    return ('Individual baseline development comparison, unchanged workload and physics.\n\n'+MARKER+
            json.dumps(dict(commit=commit,sha=hashlib.sha256(packet).hexdigest(),
                            packet=b64encode(packet).decode('ascii')),separators=(',',':')))


def read_packet(release,commit):
    assert release['tag_name']==CAMPAIGN and release['target_commitish']==commit
    assert len(release['body'])<32768
    line,=[s[len(MARKER):] for s in release['body'].splitlines() if s.startswith(MARKER)]
    d=json.loads(line)
    raw=b64decode(d['packet'],validate=True)
    assert d['commit']==commit and 0<len(raw)<16384 and hashlib.sha256(raw).hexdigest()==d['sha']
    return raw

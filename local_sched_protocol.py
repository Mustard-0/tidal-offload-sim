"""Local-only scheduler comparison on the exact existing development windows."""

from base64 import b64decode,b64encode
import hashlib
import json
from pathlib import Path

from focus_protocol import FRESH,POPS,SEED,WARM
from review_protocol import digest


CAMPAIGN='local-only-queue-discipline-20261121'
BRANCH='experiment/'+CAMPAIGN
WORKFLOW='.github/workflows/local_only_queue_discipline_20261121.yml'
BASE='04f5382adf6a80e963bebe0795609b9b18883557'
TT_RUN=44784857436
VARIANTS=('fifo','edf','deadline_spt')
WALL=1800
MARKER='STIN encrypted configuration: '
INHERITED=('focus_protocol.py','review_protocol.py','review_environment.py','review_checks.py',
           'scalability_protocol.py','pilot_protocol.py','optimization2_protocol.py','optimization2_worker.py',
           'evaluation_adapters.py','energy_qos_telemetry.py')


def plan(project=None,overlay=None):
    project=Path(project) if project else Path(__file__).resolve().parent
    def local(p):
        candidate=Path(overlay)/p if overlay else project/p
        return candidate if candidate.is_file() else project/p
    return dict(schema='local-queue-three-disciplines-v1',base=BASE,cases=FRESH,variants=VARIANTS,
        seed_label=SEED,warmup=WARM,max_parallel=3,training=False,paid=False,max_scenes=42,
        wall_seconds=WALL,protocol='Same UE resources, lifetime, deadline, admission and power. '
                                   'Nonpreemptive reordering of already admitted local jobs only.',
        acceptance=dict(qos_increase_pp=.5,completion_loss_pp=.5,task_class_completion_loss_pp=2.),
        inherited_review_hashes={p:hashlib.sha256(local(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in INHERITED})


def packet_body(packet,commit):
    return ('One Local-Only scheduler comparison on the prior matched windows; no offload or retraining.\n\n'+
            MARKER+json.dumps(dict(sha=hashlib.sha256(packet).hexdigest(),commit=commit,
                                   packet=b64encode(packet).decode('ascii')),separators=(',',':')))


def read_packet(release,commit):
    assert release['tag_name']==CAMPAIGN and release['target_commitish']==commit
    assert len(release['body'])<32768
    line,=[s[len(MARKER):] for s in release['body'].splitlines() if s.startswith(MARKER)]
    i=json.loads(line)
    raw=b64decode(i['packet'],validate=True)
    assert i['commit']==commit and 0<len(raw)<16384 and hashlib.sha256(raw).hexdigest()==i['sha']
    return raw

"""Paired seven-baseline development comparison, one frozen training seed."""

from base64 import b64decode, b64encode
import hashlib
import json
from pathlib import Path

from focus_protocol import FRESH, POPS, SEED, WARM
from review_protocol import digest, LEARNERS
from scalability_protocol import RULES


CAMPAIGN = 'baseline-capacity-single-seed-20261121'
BRANCH = 'experiment/' + CAMPAIGN
WORKFLOW = '.github/workflows/baseline_capacity_single_seed_20261121.yml'
BASE = '04f5382adf6a80e963bebe0795609b9b18883557'
TT_RUN = 44784857436
OFFLOAD = LEARNERS[1:] + tuple(x for x in RULES if x != 'Local-Only')
METHODS = OFFLOAD + ('Local-Only',)
ARMS = tuple((m,v) for m in OFFLOAD for v in ('control','guard')) + (('Local-Only','control'),)
INHERITED = ('focus_protocol.py','focus_fallback.py','energy_qos_worker.py','energy_qos_protocol.py',
             'energy_qos_telemetry.py','evaluation_adapters.py','review_checks.py','review_environment.py',
             'review_protocol.py','pilot_protocol.py','pilot_worker.py','scalability_worker.py',
             'scalability_protocol.py','optimization2_worker.py','optimization2_protocol.py','optimization2_core.py')
WALL = 3300
MARKER = 'STIN encrypted configuration: '


def plan(project=None, overlay=None):
    project = Path(project) if project else Path(__file__).resolve().parent
    def local(name):
        preferred = Path(overlay)/name if overlay else project/name
        return preferred if preferred.is_file() else project/name
    return dict(schema='paired-baseline-capacity-single-seed-v1', cases=FRESH, warm_s=WARM,
        seed=SEED, arms=ARMS, base=BASE, matching_tt_run=TT_RUN,
        deadline_ratio_reporting_only=.95, max_parallel=6, train=False, paid=False, gpu=False,
        per_job_wall_s=WALL, stage_budget_hours=24, max_scenes=len(ARMS)*len(FRESH),
        change='Same heavy-task guard before location reporting, only when no eligible remote has room '
               'and a legal local action can still meet the original deadline. No route-logit changes.',
        scope='Same checkpoints, original deadlines, privacy, queue capacities, traffic, and reward. '
              'Local-Only is a control without an offload action. No final-test or multi-seed claim.',
        inherited_review_hashes={p:hashlib.sha256(local(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in INHERITED})


def packet_body(packet,commit):
    return ('One paired baseline development run using the same short windows as the TT-MAPPO single-seed review.\n\n'
        +MARKER+json.dumps(dict(sha=hashlib.sha256(packet).hexdigest(),commit=commit,
                                packet=b64encode(packet).decode('ascii')),separators=(',',':')))


def read_packet(release,commit):
    assert release['tag_name']==CAMPAIGN and release['target_commitish']==commit
    assert len(release['body'])<32768
    line,=[s[len(MARKER):] for s in release['body'].splitlines() if s.startswith(MARKER)]
    i=json.loads(line)
    raw=b64decode(i['packet'],validate=True)
    assert i['commit']==commit and 0<len(raw)<16384 and hashlib.sha256(raw).hexdigest()==i['sha']
    return raw

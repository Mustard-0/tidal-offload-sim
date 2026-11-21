"""Matched deployment and traffic-mixture exploration, never a final test."""

from base64 import b64decode, b64encode
import hashlib
import json
from pathlib import Path

import baseline_individual_protocol as parent
from baseline_individual_protocol import digest, POPS, RULES, WARM


CAMPAIGN = 'tt-mappo-admission-mix-20261122'
BRANCH = 'experiment/' + CAMPAIGN
WORKFLOW = '.github/workflows/tt_mappo_admission_mix_20261122.yml'
BASE = '4ab1751c176d3aecd117d33b0c0cb2e2662eac87'
PARENT_RUN = 44920692113
SEED = 871901
MIXES = {'original': (.35, .35, .30), 'heavy50': (.25, .50, .25)}
PROFILES = ('latency_sensitive', 'compute_heavy', 'balanced')
CASES = tuple((mix, n, 1208221, 90.) for n in POPS for mix in MIXES)
ARMS = (('TMAPO', 'reference'), ('TMAPO', 'admission')) + tuple(
    (m, 'targeted' if m == 'Local-Only' else 'control') for m in parent.BASELINES)
MEASURE = 5.
WALL = 3200
INHERITED = parent.INHERITED + ('baseline_individual_protocol.py',
    'baseline_individual_rules.py', 'baseline_individual_worker.py')
MARKER = 'STIN encrypted configuration: '


def settings_for_mix(cfg, mix):
    profiles = cfg['task_profiles']
    assert tuple(p['name'] for p in profiles) == PROFILES
    assert tuple(p['probability'] for p in profiles) == MIXES['original']
    out = dict(cfg, task_profiles=[
        dict(p, probability=w) for p, w in zip(profiles, MIXES[mix])])
    return out


def plan(project=None, overlay=None):
    root = Path(project) if project else Path(__file__).resolve().parent
    def path(p):
        candidate = Path(overlay) / p if overlay else root / p
        return candidate if candidate.is_file() else root / p
    return dict(schema='paired-admission-and-workload-mix-v1',parent=BASE,seed=SEED,arms=ARMS,
        cases=CASES,mixes=MIXES,profile_order=PROFILES,warmup=WARM,measure=MEASURE,wall_s=WALL,
        parallel=6,expected_scenes=len(ARMS)*len(CASES),train=False,gpu=False,paid=False,
        baseline='Previously accepted deployments only, unchanged in both workload regimes.',
        hypothesis='Add bounded local/offload logit residual from a public queue/service quote. '
                   'Separately increase compute-heavy share 35 to 50 percent for every method.',
        limits='One new development trace per workload, same trace/phase and arrival/profiles RNG. '
               'An observed workload-specific ranking is not an independent final result.',
        inherited_review_hashes={p:hashlib.sha256(path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in INHERITED})


def packet_body(packet, commit):
    return 'Paired development exploration.\n\n' + MARKER + json.dumps(dict(
        commit=commit,sha=hashlib.sha256(packet).hexdigest(),packet=b64encode(packet).decode('ascii')))


def read_packet(release, commit):
    assert release['tag_name']==CAMPAIGN and release['target_commitish']==commit
    assert len(release['body'])<32768
    line,=[s[len(MARKER):] for s in release['body'].splitlines() if s.startswith(MARKER)]
    d=json.loads(line)
    raw=b64decode(d['packet'],validate=True)
    assert d['commit']==commit and 0<len(raw)<16384 and hashlib.sha256(raw).hexdigest()==d['sha']
    return raw

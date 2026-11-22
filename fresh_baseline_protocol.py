"""Pre-register a new training budget and independent shared test inputs."""

import hashlib
from pathlib import Path

from native_policy_protocol import INHERITED as PREVIOUS
from paper_load_protocol import stage


BASE = '63b11a6fb4e32cade1f5c8b4a0b15cd514f5d7cb'
PARENT_RUN = 44988234102
CAMPAIGN = 'stin-fresh-seven-baselines-20261122'
BRANCH = 'experiment/'+CAMPAIGN
WORKFLOW = '.github/workflows/stin_fresh_seven_baselines_20261122.yml'
INHERITED = PREVIOUS + ('native_policy_protocol.py','native_policy_eval.py','native_policy_worker.py')
LEARNED = ('IPPO','PPO','SD-SAC-Adapted','MARATD3-Adapted')
RULES = ('Greedy-NoPrivacy','Greedy-GeoI','Local-Only')
SEEDS = (872011,872029)
CELLS = tuple((m,s) for m in LEARNED for s in SEEDS) + tuple((m,0) for m in RULES)
POPS = tuple(range(20,151,10))
TRACES = ((1226201,69.),(1226209,219.))
WARM, MEASURE = 3.,6.
TRAIN_USERS = 80
TRAIN_ADMIT = 3000
TRAIN_SLOTS = 3200
BLOCK = 100
WALL = 3600
RATIOS = (100,95)
_BASE = stage('mid')
digest, packet_body = _BASE.digest, _BASE.packet_body


def settings(method,n):
    cfg,_ = _BASE.settings(method, None if method in RULES else _BASE.SEED,n)
    return _BASE.settings_for_mix(cfg,'original')


def cell_id(method,seed):
    assert (method,seed) in CELLS
    return method if seed==0 else f'{method}_seed{seed}'


def read_packet(release,commit):
    assert release['tag_name']==CAMPAIGN
    return _BASE.read_packet(dict(release,tag_name=_BASE.CAMPAIGN),commit)


def plan(project=None,overlay=None):
    root=Path(project) if project else Path(__file__).resolve().parent
    def sha(path):
        local=(Path(overlay) if overlay else root)/path
        return hashlib.sha256((local if local.is_file() else root/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return dict(schema='fresh-seven-baselines-v1',parent=BASE,anchor=PARENT_RUN,cells=CELLS,
        train=dict(users=TRAIN_USERS,admit=TRAIN_ADMIT,slots=TRAIN_SLOTS,block=BLOCK,from_random=True,
                   phase=55.,trace_offset=4300000,seeds=SEEDS,team_reward=True),
        test=dict(users=POPS,traces=TRACES,warmup=WARM,measure=MEASURE,deterministic=True),
        same_physics=True, same_original_deadlines=True, gpu=False, paid=False,wall=WALL,
        policies={'IPPO':'local actor and local value, common team reward, individual PPO surrogate',
                  'PPO':'global-conditioned actor and global value, factored joint slot PPO surrogate',
                  'SD-SAC-Adapted':'masked event replay, double average Q, Q-clip, replay entropy penalty',
                  'MARATD3-Adapted':'GRU, multihead critics, minimum twin target, delayed updates and masked discrete control',
                  'Greedy':'same staged cost-minimizing controller for both reporters, no private location before release',
                  'Local-Only':'original device capacity, FIFO, no publication'},
        inherited_review_hashes={p:sha(p) for p in INHERITED})

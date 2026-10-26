"""Generate the new code contract, without importing Torch or training."""
import json
from pathlib import Path
from .protocol import ROOT, plan, digest, source_hashes


def main():
    previous = json.loads((ROOT/'cloud/contract.json').read_text())
    origins={
        'bd78917bd4b8783549deea908b6a0807fa0948f6': 'F:/stin_project/tmapo_short100_20261026/cloud/contract.json',
        '93a70a93868e8f85ebf5bcffcac76b358527c2d7': 'F:/stin_project/tmapo_short100_matrix_20261026/cloud/contract.json'}
    parents=dict(contracts={commit:json.loads(Path(path).read_text()) for commit,path in origins.items()},
                 scope='Already-public source/runtime contracts only; no experiment results or checkpoint content')
    (ROOT/'experiments/episode_return/parent_contracts.json').write_text(json.dumps(parents,indent=2),encoding='utf-8')
    value = dict(source_hashes=source_hashes(), plan_hash=digest(plan()),
        cloud_runtime=previous['cloud_runtime'],
        scope='N100 [34,33,33], 23 fixed cells, explicit completed2000 to3000 continuation, terminal LR retained')
    (ROOT/'cloud/contract.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(dict(plan_hash=value['plan_hash'],files=len(value['source_hashes']),
                         formal_training_started=False)))


if __name__ == '__main__':
    main()

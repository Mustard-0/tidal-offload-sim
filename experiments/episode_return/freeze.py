"""Generate the new code contract, without importing Torch or training."""
import json
from .protocol import ROOT, plan, digest, source_hashes


def main():
    previous = json.loads((ROOT/'cloud/contract.json').read_text())
    value = dict(source_hashes=source_hashes(), plan_hash=digest(plan()),
        cloud_runtime=previous['cloud_runtime'],
        scope='N100 [34,33,33], arrival100+drain, 2000 episodes, 23 predeclared LR/seed cells')
    (ROOT/'cloud/contract.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(dict(plan_hash=value['plan_hash'],files=len(value['source_hashes']),
                         formal_training_started=False)))


if __name__ == '__main__':
    main()

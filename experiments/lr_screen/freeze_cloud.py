"""Freeze a pure-code portability contract without changing scientific settings."""
import argparse
import json
from pathlib import Path
from .runtime import ROOT, source_hashes, json_write
from .protocol import plan, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-preflight',type=Path,required=True)
    a = p.parse_args()
    parent = json.loads(a.parent_preflight.read_text())
    current = source_hashes()
    changed = [name for name,value in parent['source_hashes'].items() if current.get(name) != value]
    if changed != ['model/environment.py'] or digest(plan()) != parent['plan_hash']:
        raise ValueError('Unexpected scientific source/plan change')
    value = dict(source_hashes=current,plan_hash=digest(plan()),parent_source_hashes=parent['source_hashes'],
        parent_commit='0e5906d73f4d8dce5500f1f01042fcebe3491032',
        reviewed_scientific_change='Only hoist identical NumPy observation reductions out of the user loop',
        unchanged='All algorithm, reward, privacy, task, resource, seed, LR, budget and evaluation definitions',
        cloud_runtime=dict(python_version='3.9.25',torch='2.5.1+cpu',numpy='2.0.2',scipy='1.13.1',matplotlib='3.9.4'),
        deployment='Public pure code; free ubuntu-22.04 only; encrypted results; at most 11 independent cells',
        parent_progress='TMAPO LR1 update28 / 3584 slots retained, not restarted',
        device_reason='Measured direct GPU execution slower; simulation and tiny inference dominate')
    # runtime_info does not expose Matplotlib yet; cloud checks it separately.
    value['cloud_runtime'].pop('matplotlib')
    json_write(ROOT/'cloud/contract.json',value)
    print(ROOT/'cloud/contract.json')


if __name__ == '__main__':
    main()

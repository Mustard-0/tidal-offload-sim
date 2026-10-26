import argparse
import json
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.highload.runner import test_model

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',required=True);p.add_argument('--lr',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    try:test_model(a.out,a.method,a.lr,a.seed,a.checkpoint)
    except BaseException as error:
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc())))
        raise

import argparse
import json
from pathlib import Path
import time
import traceback
from .runner import test_model

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',required=True);p.add_argument('--lr',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();folder=a.out/'evaluation';folder.mkdir(parents=True,exist_ok=True)
    (folder/'start.json').write_text(json.dumps(dict(started_epoch=time.time(),method=a.method,lr=a.lr,seed=a.seed)))
    try:
        test_model(folder,a.method,a.lr,a.seed,a.out if (a.out/'latest.pt').exists() else None)
        if (folder/'complete.json').exists() and (folder/'paused.json').exists():(folder/'paused.json').unlink()
    except BaseException as error:
        (folder/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc())))
        raise

"""Incremental private raw records, independent physical checks, bounded RAM."""
import csv
import gzip
import io
import math
import os
from pathlib import Path


def gzip_rows(path,rows):
    if not rows:
        return
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError('Never overwrite a raw prefix')
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as binary:
        with gzip.GzipFile(fileobj=binary,mode='wb',mtime=0,compresslevel=1) as z:
            with io.TextIOWrapper(z,encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,list(dict.fromkeys(k for r in rows for k in r)))
                writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def flush(env,out,index):
    for r in env.rows:
        energy=sum(r[k] for k in ('compute_energy_j','transmit_energy_j','report_energy_j'))
        assert math.isclose(energy,r['energy_j'],abs_tol=1e-9)
        assert r['qos']==int(not r['completed'] or r['end_s']-r['arrival_s']>r['deadline_s']+1e-9)
        assert r['active_wait_s']<=.02000001
    for b in env.batch_rows:
        users=list(map(int,b['user_ids'].split(';')))
        assert len(users)==len(set(users))==b['n_distinct']<=env.region_counts[b['region']]
    for name,rows in (('tasks',env.rows),('inputs',env.input_rows),('batches',env.batch_rows),('waits',env.wait_decision_rows)):
        gzip_rows(Path(out)/name/f'{index:06d}.csv.gz',rows)
        rows.clear()


def pooled(rows,key):
    denominator='completed' if key=='delay_s' else 'arrivals'
    g=[r for r in rows if r.get(key) is not None]
    n=sum(r[denominator] for r in g)
    return sum(r[key]*r[denominator] for r in g)/n if n else None

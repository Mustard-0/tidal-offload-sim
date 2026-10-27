"""Component-only serial/parallel equivalence and CPU throughput measurement."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CASES=('TMAPO','IPPO','SD-SAC-Adapted','MARATD3-Adapted')


def component(folder,method,profile=False):
    import cProfile
    import pstats
    import resource
    from collections import deque
    import numpy as np
    from experiments.highload.runner import train,torch
    from experiments.highload.protocol import digest
    profiler=cProfile.Profile() if profile else None
    started=time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        if profiler:profiler.enable()
        train(folder,method,1,87021,component=dict(updates=20,cycle_steps=1000,drain_slots=200,
            populations=[300],validation_updates=[],checkpoint_updates=10))
        if profiler:profiler.disable()
    wall=time.monotonic()-started
    saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    # Wall clocks differ across processes; compare all scientific numeric rows.
    rows=[{k:v for k,v in r.items() if k!='training_wall_s'} for r in saved['rows']]
    def tensors(v):
        if isinstance(v,torch.Tensor):return dict(shape=list(v.shape),dtype=str(v.dtype),sha=hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest())
        if isinstance(v,dict):return {str(k):tensors(x) for k,x in v.items()}
        if isinstance(v,np.ndarray):return dict(shape=list(v.shape),dtype=str(v.dtype),sha=hashlib.sha256(v.tobytes()).hexdigest())
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,(list,tuple,deque)):return [tensors(x) for x in v]
        if hasattr(v,'__dict__'):return dict(type=type(v).__name__,fields=tensors(vars(v)))
        return v
    result=dict(method=method,component_only=True,env_steps=2000,wall_seconds=wall,
        numeric_rows_hash=digest(rows),learner_hash=digest(tensors(saved['learner'])),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if profiler:
        stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(25)
        result['profile']=stream.getvalue()
    (folder/'benchmark.json').write_text(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--child',choices=CASES);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.child:component(a.out,a.child);return
    from ops.github_api import GitHub
    from ops.transport import seal,sha
    from experiments.highload.cloud_worker import request_from_tag
    req=request_from_tag();api=GitHub(os.environ['GH_TOKEN'])
    root=ROOT/'runs/packing_benchmark';root.mkdir(parents=True,exist_ok=False)
    timings={}
    for mode in ('serial','parallel'):
        started=time.monotonic();children=[]
        for m in CASES:
            cmd=[sys.executable,'-u','-m','ops.benchmark_packed','--child',m,'--out',str(root/mode/m)]
            if mode=='serial':subprocess.run(cmd,cwd=ROOT,check=True,timeout=600)
            else:children.append(subprocess.Popen(cmd,cwd=ROOT))
        for child in children:
            if child.wait(timeout=900):raise RuntimeError('Benchmark child failed')
        timings[mode]=time.monotonic()-started
    rows=[]
    for m in CASES:
        left=json.loads((root/'serial'/m/'benchmark.json').read_text());right=json.loads((root/'parallel'/m/'benchmark.json').read_text())
        if any(left[k]!=right[k] for k in ('numeric_rows_hash','learner_hash','env_steps')):raise ValueError('Process packing changed scientific results')
        rows.append(dict(method=m,serial=left,parallel=right))
    speedup=timings['serial']/timings['parallel']
    component(root/'profile','TMAPO',profile=True)
    result=dict(component_only=True,formal_training_started=False,processes=4,cpu_count=len(os.sched_getaffinity(0)),
        equivalence_passed=True,throughput_speedup=speedup,timings=timings,rows=rows,
        memory_limit_check='four independent process peaks below12GiB',
        profile=json.loads((root/'profile/benchmark.json').read_text())['profile'],
        passed=speedup>=1.15 and sum(r['parallel']['peak_rss_kib'] for r in rows)<12*1024**2)
    header=dict(campaign=req['campaign'],direction='packing_benchmark',source_commit=os.environ['GITHUB_SHA'],run_id=os.environ['GITHUB_RUN_ID'],attempt='1')
    packet=seal(json.dumps(result).encode(),bytes.fromhex(os.environ['LR_SCREEN_TRANSPORT_KEY']),header)
    name='packing_benchmark.'+os.environ['GITHUB_RUN_ID']+'.enc'
    api.upload(req['release_id'],name,packet)
    api.upload(req['release_id'],name+'.outcome.json',json.dumps(dict(header,archive_name=name,archive_sha256=sha(packet),archive_size=len(packet),passed=result['passed'],component_only=True)).encode())
    if not result['passed']:raise RuntimeError('Packing performance/resource gate not passed; do not activate')


if __name__=='__main__':main()

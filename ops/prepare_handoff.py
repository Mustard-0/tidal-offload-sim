"""Read-only host identity check and prefix audit; never imports Torch."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.cloud_owner import prefix_audit,hashes,HANDOFF
from ops.local_owner import save,specifications


def main():
    old=ROOT/'runs/highload_local_01'
    state=json.loads((old/'coordinator.json').read_text())
    ids={state['pid'],state['cells'][HANDOFF]['pid']}
    command='Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @('+','.join(map(str,ids))+') } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,timeout=60,check=True)
    if result.stdout.strip():raise ValueError('Old owner/worker PID still present; inspect before handoff')
    if (old/'owner.lock').exists():raise ValueError('Old coordinator lock remains')
    folder=old/'cells'/HANDOFF
    cp=prefix_audit(folder,specifications()[HANDOFF])
    if cp['update']!=197 or cp['env_steps']!=19700:raise ValueError('Unexpected handoff boundary')
    if json.loads((folder/'paused.json').read_text())['reason']!='explicit stop':raise ValueError('Unknown local stop')
    proof=dict(cell=HANDOFF,prefix=str(folder),local_processes_exited=True,checked_pids=sorted(ids),
        original_started_epoch=state['started_epoch'],local_segments=1,prefix_hashes=hashes(folder),
        checkpoint=cp,verified_epoch=time.time(),scope='Metadata/CSV/hash audit; no Torch/checkpoint unpickle')
    save(ROOT/'evidence/handoff_verified.json',proof)
    print(json.dumps(dict(status='clean_saved_handoff',update=cp['update'],env_steps=cp['env_steps'],old_pids_exited=sorted(ids))))


if __name__=='__main__':main()

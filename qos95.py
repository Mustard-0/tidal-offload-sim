"""Additional strict QoS diagnostic; never changes deadlines, rewards or actions."""
import math
import csv
import json
from pathlib import Path

FACTOR=.95
TOLERANCE_S=1e-9

def strict_violation(*,failed,original_qos,arrival_s,end_s,deadline_s):
    if not isinstance(failed,bool) or not isinstance(original_qos,bool):
        raise TypeError('Expected explicit boolean outcomes')
    if not all(math.isfinite(v) for v in (arrival_s,end_s,deadline_s)) or deadline_s<=0 or end_s<arrival_s-TOLERANCE_S:
        raise ValueError('Invalid task timing')
    return failed or original_qos or end_s-arrival_s>FACTOR*deadline_s+TOLERANCE_S

class StrictQoSMixin:
    def __init__(self,*args,**kwargs):
        self.strict95_count=0
        super().__init__(*args,**kwargs)

    def end(self,j,failed=False):
        measured=self.measured(j)
        strict=strict_violation(failed=bool(failed),original_qos=bool(j.violated),
            arrival_s=j.arrival,end_s=self.time,deadline_s=j.deadline)
        super().end(j,failed=failed)
        if measured:
            self.strict95_count+=int(strict)
            if self.collect:
                assert self.rows[-1]['task_id']==j.id
                self.rows[-1].update(qos95=int(strict),qos95_deadline_s=FACTOR*j.deadline)

    def metrics(self):
        result=super().metrics()
        if not 0<=self.measure_qos<=self.strict95_count<=self.measure_arrivals:
            raise AssertionError('Strict QoS must retain original failures and denominator')
        result.update(qos_original_rate=result['qos_rate'],qos95_rate=self.strict95_count/self.measure_arrivals,
            qos95_deadline_factor=FACTOR,qos95_changes_reward=False,qos95_changes_task_deadline=False)
        return result

def task_csv_metrics(path):
    """Apply the same diagnostic to any method's existing complete task log."""
    required={'arrival_s','end_s','deadline_s','completed','qos','task_id'}
    count=original=strict=0;seen=set()
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not required<=set(reader.fieldnames or []):
            raise ValueError('Need per-task outcomes/timing; aggregate CSV cannot recover stricter QoS')
        for r in reader:
            if r['task_id'] in seen:raise ValueError('Duplicate task: do not concatenate cumulative logs')
            seen.add(r['task_id'])
            if r['completed'] not in ('0','1') or r['qos'] not in ('0','1'):raise ValueError('Invalid binary outcome')
            old=r['qos']=='1';failed=r['completed']=='0'
            if failed and not old:raise ValueError('Original QoS must count failures')
            flag=strict_violation(failed=failed,original_qos=old,arrival_s=float(r['arrival_s']),
                end_s=float(r['end_s']),deadline_s=float(r['deadline_s']))
            count+=1;original+=int(old);strict+=int(flag)
    if count==0:raise ValueError('Empty task input')
    return dict(arrivals=count,qos_original_rate=original/count,qos95_rate=strict/count,
        qos95_deadline_factor=FACTOR,task_deadlines_unchanged=True,reward_unchanged=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task-csv',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Do not overwrite old evidence')
    a.out.write_text(json.dumps(task_csv_metrics(a.task_csv),indent=2),encoding='utf-8')

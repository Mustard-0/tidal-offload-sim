"""Same physics, exact slot clock, one prepared arrival state and bounded logs."""
import hashlib
import json
from study.environment import Environment as Base
from study.geoi import GeoIEnvironment as GeoIBase
from .protocol import METHODS


class PreparedMixin:
    def __init__(self,*args,**kwargs):
        self.prepared_slot=None
        self.full_input_hash=hashlib.sha256(b'').hexdigest()
        self.input_rows=[]
        super().__init__(*args,**kwargs)

    def generate(self):
        self.time=self.slot*self.cfg['dt_s']
        if self.prepared_slot==self.slot:
            return
        first=self.serial
        super().generate()
        h=hashlib.sha256(bytes.fromhex(self.full_input_hash))
        for task_id in range(first,self.serial):
            j=self.tasks[task_id]
            r=dict(task_id=j.id,user=j.user,region=j.region,arrival_slot=j.arrival_slot,
                arrival_s=j.arrival,task_bits=j.bits,task_cycles=j.cycles,deadline_s=j.deadline,
                task_profile=j.profile,arrival_tide_factor=j.arrival_tide_factor,
                truth_x_m=float(self.xy[j.user,0]),truth_y_m=float(self.xy[j.user,1]))
            h.update(json.dumps(r,sort_keys=True,separators=(',',':')).encode()+b'\n')
            if self.collect:
                self.input_rows.append(r)
        self.full_input_hash=h.hexdigest()
        self.prepared_slot=self.slot

    def prepare(self,arrivals):
        self.time=self.slot*self.cfg['dt_s']
        self.draining=not arrivals
        if arrivals:
            self.generate()
        return self.observations()[:2]

    def _observations(self):
        context,glob,masks,users,ready=super()._observations()
        if self.family in METHODS:
            context[:self.u,17:19]=0.
            context[:self.u,26]=0.
        return context,glob,masks,users,ready


class Environment(PreparedMixin,Base):
    pass


class GeoIEnvironment(PreparedMixin,GeoIBase):
    pass

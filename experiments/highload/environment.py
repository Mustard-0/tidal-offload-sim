"""Same physics, correct post-arrival bootstrap and causal-history ablation."""
from experiments.lr_screen.restricted_environment import CandidateRestrictedEnvironment


class Environment(CandidateRestrictedEnvironment):
    def __init__(self,*args,ablation=None,**kwargs):
        self.ablation=ablation
        self.prepared_slot=None
        super().__init__(*args,**kwargs)

    def generate(self):
        if self.prepared_slot!=self.slot:
            super().generate()
            self.prepared_slot=self.slot

    def prepare(self,arrivals):
        self.draining=not arrivals
        if arrivals:
            self.generate()
        return self.observations()[:2]

    def observations(self):
        obs,glob,masks,users,ready=super().observations()
        # Evaluation-window bookkeeping is not an always-on network signal.
        glob[46:48]=0.
        if self.ablation=='TMAPO-NoForecast':
            # Route rows differ from user rows; do not erase physical energy fields.
            obs[:self.u,4:6]=0.
            obs[self.u:,57:59]=0.
            glob[28:34]=0.
        return obs,glob,masks,users,ready

    def route_features(self,*args,**kwargs):
        obs,mask,cost=super().route_features(*args,**kwargs)
        if self.ablation=='TMAPO-NoForecast':
            obs[57:62]=0.
        return obs,mask,cost

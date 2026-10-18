"""Masked PPO and an explicitly architecture-level recurrent-attention TD3 adapter."""
from __future__ import annotations

from collections import deque
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from environment import OBS_DIM, N_ROUTE

CONTROL_DIM = 3 + 2 + N_ROUTE
STAGE_SLICES = {"user": slice(0, 3), "batch": slice(3, 5), "route": slice(5, CONTROL_DIM)}


def cost_prior(stage, obs, env_cfg, scale):
    """The same public service-cost guidance for every learned baseline."""
    if stage == "user":
        costs = obs[:, 17:19]
        return torch.cat((torch.zeros_like(costs[:, :1]), -scale*costs), dim=1)
    if stage == "route":
        cost = env_cfg["weight_time"]*obs[:, :N_ROUTE] + env_cfg["weight_energy"]*obs[:, N_ROUTE:2*N_ROUTE]
        cost += env_cfg["weight_violation"]*(obs[:, :N_ROUTE] > obs[:, 45:46]).to(obs.dtype)
        return -scale*cost
    return obs.new_tensor([0., 1.]).expand(len(obs), -1)


def client_rounds(rows_by_region, batch_size):
    """Bound each client's graph while retaining every local training sample."""
    batches = {}
    for region, rows in rows_by_region.items():
        indexes = np.random.permutation(len(rows))
        batches[region] = [[rows[i] for i in indexes[start:start+batch_size]]
                           for start in range(0, len(rows), batch_size)]
    for index in range(max((len(parts) for parts in batches.values()), default=0)):
        yield {region: parts[index] for region, parts in batches.items() if index < len(parts)}


def mlp(n, h, out):
    model = nn.Sequential(nn.Linear(n,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,out))
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    return model


class ReturnNormalizer:
    """Pooled running return moments; checkpointed with the value network."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.count, self.mean, self.m2 = 0, 0., 0.

    @property
    def std(self):
        return max(np.sqrt(self.m2/max(self.count, 1)), 1e-3) if self.count else 1.

    def update(self, values):
        if not self.enabled:
            return
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        n = len(x)
        if not n or not np.isfinite(x).all():
            raise FloatingPointError("Invalid value-normalization targets")
        delta = float(x.mean())-self.mean
        total = self.count+n
        self.m2 += float(((x-x.mean())**2).sum())+delta**2*self.count*n/total
        self.mean += delta*n/total
        self.count = total

    def normalize(self, values):
        return (values-self.mean)/self.std if self.enabled else values

    def denormalize(self, values):
        return values*self.std+self.mean if self.enabled else values

    def state(self):
        return dict(enabled=self.enabled, count=self.count, mean=self.mean, m2=self.m2)

    def load(self, state):
        self.__dict__.update(state)


class PPO:
    def __init__(self, method, cfg, env_cfg, device="cpu"):
        self.method, self.cfg, self.env_cfg, self.device = method,cfg,env_cfg,torch.device(device)
        h = cfg["hidden"]
        self.actors = nn.ModuleDict({"user":mlp(OBS_DIM,h,3),"batch":mlp(OBS_DIM,h,2),
                                     "route":mlp(OBS_DIM,h,N_ROUTE)}).to(device)
        for actor in self.actors.values():
            nn.init.zeros_(actor[-1].weight)
            nn.init.zeros_(actor[-1].bias)
        self.critic = mlp(OBS_DIM if method in ("IPPO", "DPFL-Adapted") else 2*OBS_DIM,h,1).to(device)
        self.value_norm = ReturnNormalizer(cfg.get("value_normalization", False))
        self.aopt = torch.optim.Adam(self.actors.parameters(),lr=cfg["actor_lr"])
        self.copt = torch.optim.Adam(self.critic.parameters(),lr=cfg["critic_lr"])
        self.deterministic = False
        self.buffer = []

    def tensor(self,x,dtype=torch.float32):
        return torch.as_tensor(x,dtype=dtype,device=self.device)

    def policy_logits(self, stage, obs):
        return self.actors[stage](obs) + cost_prior(stage, obs, self.env_cfg, self.cfg["actor_cost_prior_scale"])

    def critic_input(self,context,glob):
        if self.method in ("IPPO", "DPFL-Adapted"):
            return context
        return np.concatenate([context,np.broadcast_to(glob,context.shape)],axis=1)

    def begin(self,context,glob,n_user):
        self.vinput = self.critic_input(context,glob)
        if not self.deterministic:
            with torch.no_grad():
                self.values = self.raw_values(self.vinput)

    def raw_values(self, inputs):
        return self.value_norm.denormalize(self.critic(self.tensor(inputs)).squeeze(-1).detach().cpu().numpy())

    def normalize_targets(self, targets):
        self.value_norm.update(targets)
        return self.value_norm.normalize(targets)

    def set_progress(self, update, total):
        fraction = (update-1)/max(total-1, 1)
        scale = 1-(1-self.cfg.get("lr_final_fraction", .1))*fraction if self.cfg.get("linear_lr_decay") else 1.
        for optimizer, name in ((self.aopt,"actor_lr"),(self.copt,"critic_lr")):
            for group in optimizer.param_groups:
                group["lr"] = self.cfg[name]*scale

    def act(self,obs,mask,stage,agents):
        with torch.no_grad():
            logits = self.policy_logits(stage,self.tensor(obs)).masked_fill(~self.tensor(mask,torch.bool),-1e9)
            if self.deterministic:
                return logits.argmax(-1).cpu().numpy(), []
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).cpu().numpy()
        action = action.cpu().numpy()
        rec = [{"stage":stage,"obs":o.copy(),"mask":m.copy(),"agent":int(i),"action":int(a),"logp":float(p)}
               for o,m,i,a,p in zip(obs,mask,agents,action,logp) if np.count_nonzero(m)>1]
        return action,rec

    def remember(self,reward,records,done):
        self.buffer.append((self.vinput.copy(),self.values.copy(),reward,records,done))

    def update(self,context,glob):
        if not self.buffer:
            return {}
        c = self.cfg
        with torch.no_grad():
            next_value = self.raw_values(self.critic_input(context,glob))
        gae = np.zeros_like(next_value)
        vinputs, targets, actor_rows = [],[],[]
        for vin,value,reward,records,done in reversed(self.buffer):
            delta = reward+c["gamma"]*next_value*(not done)-value
            gae = delta+c["gamma"]*c["gae_lambda"]*(not done)*gae
            vinputs.append(vin)
            targets.append(gae+value)
            for rec in records:
                actor_rows.append(dict(rec,adv=float(gae[rec["agent"]])))
            next_value=value
        x = self.tensor(np.concatenate(vinputs))
        y = self.tensor(self.normalize_targets(np.concatenate(targets)))
        for stage in ("user", "batch", "route"):
            rows = [r for r in actor_rows if r["stage"] == stage]
            adv = np.asarray([r["adv"] for r in rows])
            if len(adv):
                for r, a in zip(rows, (adv-adv.mean())/(adv.std()+1e-8)):
                    r["adv"] = float(a)
        stats = {"kl":[],"clip_fraction":[],"entropy":[],"value_loss":[]}
        stopped_stages = set()
        for _ in range(c["epochs"]):
            for index in torch.randperm(len(x),device=self.device).split(c["minibatch"]):
                loss = .5*(self.critic(x[index]).squeeze(-1)-y[index]).square().mean()
                self.copt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),c["max_grad_norm"])
                self.copt.step(); stats["value_loss"].append(loss.item())
            for stage in ("user","batch","route"):
                rows = [r for r in actor_rows if r["stage"]==stage]
                if not rows or stage in stopped_stages:
                    continue
                ob=self.tensor(np.stack([r["obs"] for r in rows])); mask=self.tensor(np.stack([r["mask"] for r in rows]),torch.bool)
                actions=self.tensor([r["action"] for r in rows],torch.long)
                old=self.tensor([r["logp"] for r in rows]); advantages=self.tensor([r["adv"] for r in rows])
                for ix in torch.randperm(len(rows),device=self.device).split(c["minibatch"]):
                    dist=Categorical(logits=self.policy_logits(stage,ob[ix]).masked_fill(~mask[ix],-1e9))
                    logratio=dist.log_prob(actions[ix])-old[ix]; ratio=logratio.exp()
                    entropy=dist.entropy().mean()
                    kl = ((ratio-1)-logratio).mean().item()
                    if kl > 1.5*c.get("target_kl", float("inf")):
                        stopped_stages.add(stage)
                        break
                    loss=-torch.minimum(ratio*advantages[ix],ratio.clamp(1-c["ppo_clip"],1+c["ppo_clip"])*advantages[ix]).mean()-c["entropy"]*entropy
                    self.aopt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.actors.parameters(),c["max_grad_norm"])
                    self.aopt.step()
                    stats["kl"].append(kl)
                    stats["clip_fraction"].append(((ratio-1).abs()>c["ppo_clip"]).float().mean().item())
                    stats["entropy"].append(entropy.item())
        self.buffer.clear()
        return {**{k:float(np.mean(v)) if v else 0. for k,v in stats.items()},
                "return_mean":float(self.value_norm.mean),"return_std":float(self.value_norm.std),
                "kl_stopped_stages":float(len(stopped_stages))}

    def state(self,for_evaluation=False):
        state = {"actors":self.actors.state_dict(),"critic":self.critic.state_dict(),"value_norm":self.value_norm.state()}
        if not for_evaluation:
            state.update(aopt=self.aopt.state_dict(), copt=self.copt.state_dict())
        return state

    def load(self,state):
        self.actors.load_state_dict(state["actors"]); self.critic.load_state_dict(state["critic"])
        if "value_norm" in state:
            self.value_norm.load(state["value_norm"])
        if "aopt" in state:
            self.aopt.load_state_dict(state["aopt"]); self.copt.load_state_dict(state["copt"])

    def reset_history(self):
        pass


class RecurrentActor(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.gru=nn.GRU(OBS_DIM+CONTROL_DIM,h,batch_first=True)
        self.head=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,CONTROL_DIM),nn.Tanh())
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self,x):
        # B x agents x history x features; parameters are shared between agents.
        b,n,t,d=x.shape
        out,_=self.gru(x.reshape(b*n,t,d))
        return self.head(out[:,-1]).reshape(b,n,CONTROL_DIM)


class AttentionCritic(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.embed=nn.Linear(OBS_DIM+CONTROL_DIM,h)
        self.attn=nn.MultiheadAttention(h,4,batch_first=True)
        self.head=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))

    def forward(self,obs,actions):
        z=torch.relu(self.embed(torch.cat([obs,actions],-1)))
        z,_=self.attn(z,z,z,need_weights=False)
        return self.head(z).squeeze(-1)


class MATD3:
    """Continuous preferences mapped to feasible discrete controls, not an exact reproduction.

    Regional actor outputs routing biases around reported-position service costs.
    Bandwidth/CPU sharing is the common environment, NOT the paper's continuous
    resource-allocation optimization. Histories contain observations and actions.
    """
    def __init__(self,method,cfg,env_cfg,device="cpu"):
        self.method,self.cfg,self.env_cfg=method,cfg,env_cfg
        self.device=torch.device(device); h=cfg["hidden"]
        self.actor=RecurrentActor(h).to(device)
        self.q=nn.ModuleList([AttentionCritic(h),AttentionCritic(h)]).to(device)
        self.target_actor=copy.deepcopy(self.actor); self.target_q=copy.deepcopy(self.q)
        self.aopt=torch.optim.Adam(self.actor.parameters(),lr=cfg["actor_lr"])
        self.copt=torch.optim.Adam(self.q.parameters(),lr=cfg["critic_lr"])
        self.replay=deque(maxlen=cfg["td3_replay_slots"])
        self.history=None; self.prev=None; self.pending=None
        self.deterministic=False; self.steps=self.updates=0

    def tensor(self,x):
        return torch.as_tensor(x,dtype=torch.float32,device=self.device)

    def reset_history(self):
        self.history=None; self.prev=None

    def begin(self,context,glob,n_user):
        self.n_user=n_user
        if self.history is None:
            self.history=np.zeros((len(context),self.cfg["td3_history"],OBS_DIM+CONTROL_DIM),np.float32)
            self.prev=np.zeros((len(context),CONTROL_DIM),np.float32)
        self.history=np.roll(self.history,-1,axis=1)
        self.history[:,-1]=np.concatenate([context,self.prev],axis=1)
        self.current=self.history.copy()
        with torch.no_grad():
            action=self.actor(self.tensor(self.current[None]))[0].cpu().numpy()
        if not self.deterministic:
            if self.steps<self.cfg["td3_start_steps"]:
                action=np.random.uniform(-1,1,action.shape).astype(np.float32)
            else:
                action=np.clip(action+np.random.normal(0,self.cfg["td3_exploration"],action.shape),-1,1).astype(np.float32)
        self.action=action; self.prev=action.copy()

    def act(self,obs,mask,stage,agents):
        a=self.action[agents,STAGE_SLICES[stage]].copy()*self.cfg.get("td3_residual_scale", 4.)
        a+=cost_prior(stage, self.tensor(obs), self.env_cfg, self.cfg["actor_cost_prior_scale"]).cpu().numpy()
        return np.argmax(np.where(mask,a,-np.inf),axis=1),[]

    def remember(self,reward,records,done):
        if self.pending is not None:
            history,action,r,lastdone=self.pending
            self.replay.append((history,action,r,self.current.copy(),lastdone))
        self.pending=(self.current.copy(),self.action.copy(),reward,done)
        if done:
            self.replay.append((*self.pending[:3],np.zeros_like(self.current),True))
            self.pending=None
        self.steps+=1

    def update(self,context=None,glob=None):
        c=self.cfg
        if len(self.replay)<c["td3_batch"] or self.steps<c["td3_start_steps"]:
            return {}
        losses=[]
        # Match update-to-interaction ratio; no hidden extra environment samples.
        for _ in range(max(1,c["rollout_steps"]//c["td3_update_every"])):
            ix=np.random.choice(len(self.replay),c["td3_batch"],replace=False)
            h,a,r,nh,done=map(np.stack,zip(*(self.replay[i] for i in ix)))
            h,a,r,nh,done=map(self.tensor,(h,a,r,nh,done))
            ob=h[:,:,-1,:OBS_DIM]; nob=nh[:,:,-1,:OBS_DIM]
            with torch.no_grad():
                noise=(torch.randn_like(a)*c["td3_target_noise"]).clamp(-c["td3_noise_clip"],c["td3_noise_clip"])
                na=(self.target_actor(nh)+noise).clamp(-1,1)
                reward = r[:,None] if r.ndim == 1 else r
                target=reward+c["gamma"]*(1-done[:,None])*torch.minimum(self.target_q[0](nob,na),self.target_q[1](nob,na))
            loss=sum((q(ob,a)-target).square().mean() for q in self.q)
            self.copt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(),c["max_grad_norm"]); self.copt.step()
            losses.append(loss.item()); self.updates+=1
            if self.updates%c["td3_policy_delay"]==0:
                for p in self.q.parameters(): p.requires_grad_(False)
                aloss=-self.q[0](ob,self.actor(h)).mean()
                self.aopt.zero_grad(); aloss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),c["max_grad_norm"]); self.aopt.step()
                for p in self.q.parameters(): p.requires_grad_(True)
                with torch.no_grad():
                    for src,tgt in ((self.actor,self.target_actor),(self.q,self.target_q)):
                        for p,t in zip(src.parameters(),tgt.parameters()):
                            t.lerp_(p,c["td3_tau"])
        return {"td3_critic_loss":float(np.mean(losses)),"td3_updates":self.updates}

    def state(self,for_evaluation=False):
        result = {"actor":self.actor.state_dict(),"q":self.q.state_dict(),"target_actor":self.target_actor.state_dict(),
                "target_q":self.target_q.state_dict(),"aopt":self.aopt.state_dict(),"copt":self.copt.state_dict(),
                "steps":self.steps,"updates":self.updates,"history":None if for_evaluation else self.history,
                "prev":None if for_evaluation else self.prev,"pending":None if for_evaluation else self.pending,
                "replay":deque(maxlen=self.cfg["td3_replay_slots"]) if for_evaluation else self.replay}
        if for_evaluation:
            del result["aopt"], result["copt"]
        return result

    def load(self,s):
        for key in ("actor","q","target_actor","target_q","aopt","copt"):
            if key in s:
                getattr(self,key).load_state_dict(s[key])
        for key in ("steps","updates","history","prev","pending","replay"):
            setattr(self,key,s[key])
        self.actor.gru.flatten_parameters()
        self.target_actor.gru.flatten_parameters()

    set_progress = PPO.set_progress


class FederatedRecurrentActor(nn.Module):
    """Local DNN features followed by a short chronological GRU."""
    def __init__(self, hidden, outputs):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, outputs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sequence):
        features = self.local(sequence)
        recurrent, _ = self.gru(features)
        return self.head(recurrent[:, -1])


class DPFL(PPO):
    """DPFL-inspired federated recurrent PPO adapter, not the paper's code.

    The three regions are clients. Each computes a clipped gradient on its own
    observations and actions; independently noised gradients are aggregated by
    sample count. The location-report protocol is shared with every offloading
    baseline. Gradient noise has no claimed formal (epsilon, delta) guarantee.
    """
    def __init__(self, method, cfg, env_cfg, device="cpu"):
        super().__init__(method, cfg, env_cfg, device)
        h = cfg["hidden"]
        self.actors = nn.ModuleDict({"user": FederatedRecurrentActor(h, 3),
                                     "batch": FederatedRecurrentActor(h, 2),
                                     "route": FederatedRecurrentActor(h, N_ROUTE)}).to(device)
        self.aopt = torch.optim.Adam(self.actors.parameters(), lr=cfg["actor_lr"])
        self.history = None
        self.previous_context = None
        self.n_user = 0
        self.history_length = cfg["dpfl_history"]

    def reset_history(self):
        self.history = None
        self.previous_context = None

    def begin(self, context, glob, n_user):
        self.n_user = n_user
        if self.history is None or len(self.history) != len(context):
            self.history = np.zeros((len(context), self.history_length - 1, OBS_DIM), np.float32)
            self.previous_context = None
        if self.previous_context is not None and self.history.shape[1]:
            self.history = np.roll(self.history, -1, axis=1)
            self.history[:, -1] = self.previous_context
        self.previous_context = context.copy()
        super().begin(context, glob, n_user)

    def region(self, agent):
        per_region = self.env_cfg["users_per_region"]
        return agent // per_region if agent < self.n_user else agent - self.n_user

    def policy_logits(self, stage, sequence):
        obs = sequence[:, -1]
        logits = self.actors[stage](sequence)
        return logits + cost_prior(stage, obs, self.env_cfg, self.cfg["actor_cost_prior_scale"])

    def act(self, obs, mask, stage, agents):
        sequence = np.concatenate((self.history[agents], obs[:, None, :]), axis=1)
        with torch.no_grad():
            logits = self.policy_logits(stage, self.tensor(sequence)).masked_fill(~self.tensor(mask, torch.bool), -1e9)
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if self.deterministic else dist.sample()
            logp = dist.log_prob(action).cpu().numpy()
        choices = action.cpu().numpy()
        if self.deterministic:
            return choices, []
        records = [{"stage": stage, "sequence": seq.copy(), "mask": m.copy(),
                    "agent": int(agent), "region": self.region(int(agent)),
                    "action": int(a), "logp": float(p)}
                   for seq, m, agent, a, p in zip(sequence, mask, agents, choices, logp) if np.count_nonzero(m)>1]
        return choices, records

    def federated_step(self, model, optimizer, client_losses):
        params = list(model.parameters())
        weighted = [torch.zeros_like(p) for p in params]
        count = sum(size for _, _, size in client_losses)
        clip = self.cfg["dpfl_clip_norm"]
        noise = self.cfg["dpfl_noise_multiplier"]
        for _, loss, size in client_losses:
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            grads = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, grads)]
            norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
            factor = min(1., clip / (float(norm) + 1e-12))
            weight = size / count
            for aggregate, grad in zip(weighted, grads):
                aggregate.add_((grad * factor + torch.randn_like(grad) * (clip * noise)) * weight)
        optimizer.zero_grad()
        for p, grad in zip(params, weighted):
            p.grad = grad
        optimizer.step()

    def update(self, context, glob):
        if not self.buffer:
            return {}
        c = self.cfg
        with torch.no_grad():
            next_value = self.raw_values(self.critic_input(context, glob))
        gae = np.zeros_like(next_value)
        critic_rows, actor_rows = [], []
        for vin, value, reward, records, done in reversed(self.buffer):
            delta = reward + c["gamma"] * next_value * (not done) - value
            gae = delta + c["gamma"] * c["gae_lambda"] * (not done) * gae
            targets = gae + value
            for agent in range(len(value)):
                critic_rows.append((vin[agent], float(targets[agent]), self.region(agent)))
            for rec in records:
                actor_rows.append(dict(rec, adv=float(gae[rec["agent"]])))
            next_value = value
        normalized = self.normalize_targets(np.asarray([r[1] for r in critic_rows]))
        critic_rows = [(r[0], float(y), r[2]) for r,y in zip(critic_rows, normalized)]
        for stage in ("user", "batch", "route"):
            rows = [r for r in actor_rows if r["stage"] == stage]
            advantages = np.array([r["adv"] for r in rows])
            if len(advantages):
                normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                for rec, adv in zip(rows, normalized):
                    rec["adv"] = float(adv)
        value_losses, entropy_values, kl_values, clips = [], [], [], []
        stopped_stages = set()
        for _ in range(c["epochs"]):
            clients = {g: [r for r in critic_rows if r[2] == g] for g in range(3)}
            for batch in client_rounds(clients, c["minibatch"]):
                critic_clients = []
                for region, rows in batch.items():
                    x = self.tensor(np.stack([r[0] for r in rows]))
                    y = self.tensor([r[1] for r in rows])
                    loss = .5 * (self.critic(x).squeeze(-1) - y).square().mean()
                    critic_clients.append((region, loss, len(rows)))
                    value_losses.append(float(loss.detach()))
                self.federated_step(self.critic, self.copt, critic_clients)
            for stage in ("user", "batch", "route"):
                if stage in stopped_stages:
                    continue
                clients = {g: [r for r in actor_rows if r["stage"] == stage and r["region"] == g] for g in range(3)}
                for batch in client_rounds(clients, c["minibatch"]):
                    client_losses, round_kl = [], []
                    for region, rows in batch.items():
                        seq = self.tensor(np.stack([r["sequence"] for r in rows]))
                        mask = self.tensor(np.stack([r["mask"] for r in rows]), torch.bool)
                        actions = self.tensor([r["action"] for r in rows], torch.long)
                        old = self.tensor([r["logp"] for r in rows])
                        adv = self.tensor([r["adv"] for r in rows])
                        dist = Categorical(logits=self.policy_logits(stage, seq).masked_fill(~mask, -1e9))
                        logratio = dist.log_prob(actions) - old
                        ratio = logratio.exp()
                        entropy = dist.entropy().mean()
                        loss = -torch.minimum(ratio * adv, ratio.clamp(1-c["ppo_clip"], 1+c["ppo_clip"]) * adv).mean() - c["entropy"] * entropy
                        client_losses.append((region, loss, len(rows)))
                        entropy_values.append(float(entropy.detach()))
                        round_kl.append(float(((ratio-1)-logratio).mean().detach()))
                        clips.append(float(((ratio-1).abs()>c["ppo_clip"]).float().mean()))
                    kl_values.extend(round_kl)
                    if max(round_kl, default=0.) > 1.5*c.get("target_kl", float("inf")):
                        stopped_stages.add(stage)
                        break
                    self.federated_step(self.actors[stage], self.aopt, client_losses)
        self.buffer.clear()
        return {"value_loss": float(np.mean(value_losses)), "entropy": float(np.mean(entropy_values)) if entropy_values else 0.,
                "return_mean":float(self.value_norm.mean),"return_std":float(self.value_norm.std),
                "kl":float(np.mean(kl_values)) if kl_values else 0.,
                "clip_fraction":float(np.mean(clips)) if clips else 0.,
                "kl_stopped_stages":float(len(stopped_stages)),
                "fl_clients": 3., "gradient_noise_multiplier": float(c["dpfl_noise_multiplier"])}

    def state(self, for_evaluation=False):
        result = super().state(for_evaluation)
        result["history"] = None if for_evaluation else self.history
        result["previous_context"] = None if for_evaluation else self.previous_context
        result["n_user"] = self.n_user
        return result

    def load(self, state):
        super().load(state)
        self.history = state["history"]
        self.previous_context = state.get("previous_context")
        self.n_user = state["n_user"]


def make_policy(method,config,device="cpu"):
    if method=="Local-Only":
        from environment import LocalOnlyPolicy
        return LocalOnlyPolicy()
    if method=="Greedy":
        from environment import GreedyPolicy
        return GreedyPolicy(config["environment"])
    if method=="MARATD3-Adapted":
        return MATD3(method,config["training"],config["environment"],device)
    if method=="DPFL-Adapted":
        return DPFL(method, config["training"], config["environment"], device)
    if method not in {"TMAPO","IPPO","TMAPO-NoAmp","TMAPO-NoPred","TMAPO-NoWait"}:
        raise ValueError(method)
    return PPO(method,config["training"],config["environment"],device)

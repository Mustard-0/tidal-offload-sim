"""Independent baseline controllers for the fixed STIN observation/action contract."""

import copy
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


OBS = 64
ROUTE = 11
JOINT = 3 + ROUTE
GAMMA = .995


def mlp(inputs, outputs, width=48):
    return nn.Sequential(nn.Linear(inputs,width), nn.Tanh(), nn.Linear(width,width),
                         nn.Tanh(), nn.Linear(width,outputs))


def tensor(x):
    return torch.as_tensor(x,dtype=torch.float32)


def distribution(logits, mask):
    legal=torch.as_tensor(mask,dtype=torch.bool)
    return Categorical(logits=logits.masked_fill(~legal,-1e9))


def columns(records):
    return tuple(np.stack(part) for part in zip(*records))


def update_net(loss, optimizer, net):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def blend(target, net, tau=.01):
    for a,b in zip(target.parameters(),net.parameters()):
        a.lerp_(b,tau)


def encode(obs, mask, stage, glob=None):
    stage_id={'user':0,'route':1}[stage]
    mark=np.broadcast_to(np.eye(2,dtype=np.float32)[stage_id],(len(obs),2))
    padded=np.zeros((len(obs),ROUTE),dtype=bool)
    padded[:,:mask.shape[1]]=mask
    parts=(obs,mark) if glob is None else (obs,np.broadcast_to(glob,obs.shape),mark)
    return np.concatenate(parts,axis=1).astype(np.float32),padded


def advantages(rewards, values, dones, tail, gamma=GAMMA, lam=.95):
    result=np.empty_like(values)
    accum=np.zeros_like(tail)
    for i in range(len(values)-1,-1,-1):
        keep=1.-float(dones[i])
        accum=rewards[i]+gamma*keep*tail-values[i]+gamma*lam*keep*accum
        result[i]=accum
        tail=values[i]
    return result


class ClippedPPO:
    """Shared-weight IPPO with local values, or joint-action PPO with one global value."""
    def __init__(self, centralized):
        self.centralized=centralized
        self.actor=mlp(2*OBS+2 if centralized else OBS+2,ROUTE)
        self.critic=mlp(OBS,1)
        self.aopt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.copt=torch.optim.Adam(self.critic.parameters(),lr=6e-4)
        self.training=True
        self.steps=[]

    def begin(self, context, glob, n_user):
        self.glob=np.asarray(glob,dtype=np.float32)
        if self.training:
            self.view=self.glob[None].copy() if self.centralized else np.asarray(context,dtype=np.float32).copy()
            self.actions=[]

    @torch.no_grad()
    def act(self, obs, mask, stage, agents):
        x,m=encode(obs,mask,stage,self.glob if self.centralized else None)
        d=distribution(self.actor(tensor(x)),m)
        a=d.sample() if self.training else d.logits.argmax(-1)
        if self.training:
            self.actions.extend(zip(x.copy(),m.copy(),a.numpy(),d.log_prob(a).numpy(),
                                    np.zeros_like(agents) if self.centralized else agents))
        return a.numpy(),[]

    def finish(self,reward,done):
        self.steps.append((self.view,self.actions,reward,done))

    def learn(self, context, glob):
        views, decisions, rs, done=zip(*self.steps)
        self.steps=[]
        inputs=tensor(np.stack(views))
        with torch.no_grad():
            mv=self.critic(inputs).squeeze(-1)
            tail=self.critic(tensor(glob[None] if self.centralized else context)).squeeze(-1)
            adv=advantages(rs,mv.numpy(),done,tail.numpy())
            target=mv+tensor(adv)
        critic=update_net(.5*(self.critic(inputs).squeeze(-1)-target).square().mean(),self.copt,self.critic)
        facts=[(slot,*r) for slot,group in enumerate(decisions) for r in group if r[1].sum()>1]
        if not facts:
            return {'critic_loss':critic}
        slot,x,mask,a,old,who=columns(facts)
        slot=torch.as_tensor(slot,dtype=torch.long)
        a=torch.as_tensor(a,dtype=torch.long)
        x,old=tensor(x),tensor(old)
        if self.centralized:
            active=slot.unique()
            aa=tensor(adv[:,0])[active]
        else:
            aa=tensor(adv[slot.numpy(),who])
        aa=(aa-aa.mean())/(aa.std(unbiased=False)+1e-8)
        for _ in range(2):
            d=distribution(self.actor(x),mask)
            logs=d.log_prob(a)-old
            if self.centralized:
                logs=torch.zeros(len(views)).index_add(0,slot,logs)[active]
            ratio=logs.clamp(-20.,20.).exp()
            objective=torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa)
            policy=update_net(-objective.mean()-.01*d.entropy().mean(),self.aopt,self.actor)
        return {'actor_loss':policy,'critic_loss':critic}


def clipped_q_loss(q, reference, target, limit=.5):
    clipped=reference+(q-reference).clamp(-limit,limit)
    return torch.maximum((q-target).square(),(clipped-target).square()).mean()


class StableDiscreteSAC:
    """Stage/agent event replay, double-average Q, Q-clip and replay entropy penalty."""
    def __init__(self):
        dim=2*OBS+2
        self.actor=mlp(dim,ROUTE)
        self.critic=nn.ModuleList([mlp(dim,ROUTE) for _ in range(2)])
        self.targets=copy.deepcopy(self.critic)
        self.aopt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.copt=torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.replay=deque(maxlen=16384)
        self.pending={}
        self.training=True
        self.slot=0
        self.alpha=.05

    def begin(self, context, glob, n_user):
        self.glob=np.asarray(glob,dtype=np.float32)

    @torch.no_grad()
    def act(self, obs, mask, stage, agents):
        x,m=encode(obs,mask,stage,self.glob)
        dist=distribution(self.actor(tensor(x)),m)
        a=dist.sample() if self.training else dist.logits.argmax(-1)
        if self.training:
            for agent,xx,mm,action,h in zip(agents,x,m,a.numpy(),dist.entropy().numpy()):
                key=stage,int(agent)
                former=self.pending.get(key)
                if former is not None:
                    sx,sm,sa,sh,start,r=former
                    self.replay.append((sx,sm,sa,sh,r,xx.copy(),mm.copy(),GAMMA**(self.slot-start)))
                self.pending[key]=(xx.copy(),mm.copy(),int(action),float(h),self.slot,0.)
        return a.numpy(),[]

    def finish(self, reward, done):
        for key,(x,m,a,h,start,r) in self.pending.items():
            r+=GAMMA**(self.slot-start)*reward
            if done:
                self.replay.append((x,m,a,h,r,x,m,0.))
            else:
                self.pending[key]=(x,m,a,h,start,r)
        if done:
            self.pending.clear()
        self.slot+=1

    def learn(self, context, glob):
        if len(self.replay)<128:
            return {}
        for _ in range(6):
            x,m,a,h,r,nx,nm,discount=columns([self.replay[i] for i in
                np.random.choice(len(self.replay),128,replace=False)])
            x,nx=tensor(x),tensor(nx)
            a=torch.as_tensor(a,dtype=torch.long)[:,None]
            with torch.no_grad():
                next_dist=distribution(self.actor(nx),nm)
                q=torch.stack([net(nx) for net in self.targets]).mean(0)
                target=tensor(r)+tensor(discount)*(next_dist.probs*(q-self.alpha*next_dist.logits)).sum(-1)
                old_q=[net(x).gather(1,a).squeeze(1) for net in self.targets]
            loss=sum(clipped_q_loss(net(x).gather(1,a).squeeze(1),old,target)
                     for net,old in zip(self.critic,old_q))
            critic=update_net(loss,self.copt,self.critic)
            d=distribution(self.actor(x),m)
            with torch.no_grad():
                q=torch.stack([net(x) for net in self.critic]).mean(0)
            soft=(d.probs*(self.alpha*d.logits-q)).sum(-1).mean()
            penalty=.25*(d.entropy()-tensor(h)).square().mean()
            actor=update_net(soft+penalty,self.aopt,self.actor)
            blend(self.targets,self.critic)
        return {'actor_loss':actor,'critic_loss':critic,'entropy_penalty':float(penalty.detach())}


def straight_through(actions, legal):
    pieces=[]
    for start,end in ((0,3),(3,JOINT)):
        avail=legal[...,start:end]
        active=avail.any(-1,keepdim=True)
        scores=actions[...,start:end].masked_fill(~(avail|~active),-1e9)
        p=torch.softmax(scores,dim=-1)
        hard=torch.nn.functional.one_hot(p.argmax(-1),end-start).to(p)
        pieces.append((hard+(p-p.detach()))*active)
    return torch.cat(pieces,dim=-1)


class RecurrentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(OBS+JOINT,32,batch_first=True)

    def forward(self,h):
        batch,agents,steps,dim=h.shape
        y,_=self.rnn(h.reshape(batch*agents,steps,dim))
        return y[:,-1].reshape(batch,agents,32)


class RecurrentActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=RecurrentEncoder()
        self.out=nn.Linear(32,JOINT)

    def forward(self,h):
        return 4.*self.out(self.encoder(h)).tanh()


class AttentionalCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=RecurrentEncoder()
        self.join=nn.Linear(32+JOINT,32)
        self.attention=nn.MultiheadAttention(32,2,batch_first=True)
        self.value=mlp(64,1,width=32)

    def forward(self,h,a):
        x=self.join(torch.cat((self.encoder(h),a),dim=-1)).tanh()
        y,_=self.attention(x,x,x,need_weights=False)
        return self.value(torch.cat((x,y),dim=-1)).squeeze(-1).mean(-1)


class RecurrentAttentionTD3:
    """Discrete STIN adaptation: recurrent decentralized actor, centralized twin critics."""
    def __init__(self):
        self.actor=RecurrentActor()
        self.critic=nn.ModuleList([AttentionalCritic() for _ in range(2)])
        self.actor_target=copy.deepcopy(self.actor)
        self.targets=copy.deepcopy(self.critic)
        self.aopt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.copt=torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.replay=deque(maxlen=384)
        self.pending=None
        self.training=True
        self.history=None
        self.updates=0

    @torch.no_grad()
    def begin(self, context, glob, n_user):
        if self.history is None:
            self.history=np.zeros((len(context),4,OBS+JOINT),dtype=np.float32)
            self.prev=np.zeros((len(context),JOINT),dtype=np.float32)
        self.history=np.roll(self.history,-1,axis=1)
        self.history[:,-1]=np.concatenate((context,self.prev),axis=1)
        self.logits=self.actor(tensor(self.history[None]))[0].numpy()
        if self.training:
            self.logits+=np.random.normal(0.,.2,self.logits.shape)
        self.actions=np.zeros_like(self.prev)
        self.legal=np.zeros_like(self.prev,dtype=bool)
        self.counts=np.zeros((len(context),2),dtype=int)

    def act(self, obs, mask, stage, agents):
        k={'user':0,'route':1}[stage]
        start,end=((0,3),(3,JOINT))[k]
        selected=np.where(mask,self.logits[agents,start:end],-np.inf).argmax(-1)
        for agent,a,visible in zip(agents,selected,mask):
            old=self.counts[agent,k]
            row=self.actions[agent,start:end]
            row*=old
            row[a]+=1
            row/=old+1
            self.legal[agent,start:end]|=visible
            self.counts[agent,k]+=1
        self.prev=self.actions
        return selected,[]

    def finish(self,reward,done):
        if self.pending is not None:
            self.replay.append((*self.pending,self.history.copy(),self.legal.copy(),False))
        self.pending=(self.history.copy(),self.actions.copy(),self.legal.copy(),reward)
        if done:
            self.replay.append((*self.pending,self.history.copy(),self.legal.copy(),True))
            self.pending=None

    def learn(self,context,glob):
        if len(self.replay)<16:
            return {}
        result={}
        for _ in range(6):
            h,a,m,r,nh,nm,done=columns([self.replay[i] for i in
                np.random.choice(len(self.replay),16,replace=False)])
            h,a,nh=tensor(h),tensor(a),tensor(nh)
            with torch.no_grad():
                preference=self.actor_target(nh)
                perturbation=(.2*torch.randn_like(preference)).clamp(-.5,.5)
                action=straight_through((preference+perturbation).clamp(-4.,4.),torch.as_tensor(nm))
                target=tensor(r)+GAMMA*(1.-tensor(done))*torch.minimum(
                    self.targets[0](nh,action),self.targets[1](nh,action))
            loss=sum((model(h,a)-target).square().mean() for model in self.critic)
            result['critic_loss']=update_net(loss,self.copt,self.critic)
            self.updates+=1
            if self.updates%2==0:
                for p in self.critic[0].parameters():
                    p.requires_grad_(False)
                choice=straight_through(self.actor(h),torch.as_tensor(m))
                result['actor_loss']=update_net(-self.critic[0](h,choice).mean(),self.aopt,self.actor)
                for p in self.critic[0].parameters():
                    p.requires_grad_(True)
                blend(self.targets,self.critic)
                blend(self.actor_target,self.actor)
        return result


def controller(method):
    if method=='IPPO':
        return ClippedPPO(False)
    if method=='PPO':
        return ClippedPPO(True)
    if method=='SD-SAC-Adapted':
        return StableDiscreteSAC()
    if method=='MARATD3-Adapted':
        return RecurrentAttentionTD3()
    raise ValueError(method)

"""Explicit architecture adaptations, NOT author implementations or trained models.

See the full-paper mapping in the adjacent README. The supplied Atari code is
not imported. Both adapters implement the existing begin/act/remember API.
"""
import copy
from collections import deque
import math
from . import ROOT
from experiments.lr_screen.runtime import np, torch
from experiments.lr_screen.algorithms import prior
from environment import OBS_DIM, N_ROUTE
from learners import mlp
from torch import nn
from torch.distributions import Categorical


def defaults():
    return dict(hidden=32, actor_lr=3e-4, critic_lr=3e-4, gamma=.99,
        tau=.01, batch=64, replay_capacity=256, max_grad_norm=.5,
        actor_cost_prior_scale=2., temperature_lr=3e-4, initial_alpha=.1,
        entropy_fraction=.8, history=8, exploration=.05, target_noise=.1,
        noise_clip=.2, policy_delay=2, residual_scale=4.)


def tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype, device=device)


def grad_step(loss, optimizer, parameters, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite baseline loss')
    parameters = list(parameters)
    optimizer.zero_grad()
    loss.backward()
    norm = nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


@torch.no_grad()
def polyak(source, target, tau):
    for src, dst in zip(source.parameters(), target.parameters()):
        dst.lerp_(src, tau)


class MaskedDiscreteSAC:
    """Central decision-event SAC; slot time, not factor count, discounts returns.

    Within a slot, user/group/route choices form a sequential joint decision.
    Earlier factors receive zero reward; the final factor receives the scalar
    slot reward. No-action slots accumulate on the pending decision. A genuine
    terminal drains pending reward with zero bootstrap. Truncation is not done.
    """
    def __init__(self, cfg, env_cfg, device='cpu'):
        self.cfg, self.env_cfg, self.device = dict(cfg), env_cfg, torch.device(device)
        h, d = cfg['hidden'], 2*OBS_DIM+2
        self.actor = mlp(d, h, N_ROUTE).to(device)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.q = nn.ModuleList([mlp(d, h, N_ROUTE) for _ in range(2)]).to(device)
        self.target_q = copy.deepcopy(self.q)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.copt = torch.optim.Adam(self.q.parameters(), lr=cfg['critic_lr'])
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg['initial_alpha']), device=device))
        self.topt = torch.optim.Adam([self.log_alpha], lr=cfg['temperature_lr'])
        self.replay = deque(maxlen=cfg['replay_capacity'])
        self.pending, self.slot, self.updates = None, 0, 0
        self.deterministic = False
        self.collect_transitions = True

    def reset_history(self):
        if self.pending is not None:
            raise ValueError('Finish the terminal transition before resetting')
        self.slot = 0

    def begin(self, context, glob, n_user):
        self.glob = np.asarray(glob, np.float32).copy()

    def encode(self, obs, mask, stage):
        if stage not in ('user', 'route'):
            raise ValueError(stage)
        padded = np.zeros((len(obs), N_ROUTE), bool)
        padded[:, :mask.shape[1]] = mask
        if not padded.any(axis=1).all():
            raise ValueError('Empty feasible-action set')
        kind = np.tile([stage == 'user', stage == 'route'], (len(obs), 1))
        x = np.concatenate((obs, np.broadcast_to(self.glob, obs.shape), kind), axis=1).astype(np.float32)
        biases = np.zeros((len(obs), N_ROUTE), np.float32)
        biases[:, :mask.shape[1]] = prior(stage, tensor(obs, self.device),
            self.env_cfg, self.cfg['actor_cost_prior_scale']).detach().cpu().numpy()
        return x, padded, biases

    def distribution(self, x, mask, bias):
        logits = (self.actor(x)+bias).masked_fill(~mask, -1e9)
        logp = torch.log_softmax(logits, dim=-1)
        return logp.exp(), logp

    def connect(self, next_x, next_mask, next_bias):
        if self.pending is not None:
            x, mask, bias, a, reward, start = self.pending
            self.replay.append((x, mask, bias, a, reward, next_x.copy(),
                next_mask.copy(), next_bias.copy(), self.cfg['gamma']**(self.slot-start)))

    def act(self, obs, mask, stage, agents):
        x, feasible, bias = self.encode(np.asarray(obs), np.asarray(mask), stage)
        with torch.no_grad():
            probs, _ = self.distribution(tensor(x, self.device),
                tensor(feasible, self.device, torch.bool), tensor(bias, self.device))
            actions = (probs.argmax(-1) if self.deterministic else Categorical(probs=probs).sample()).cpu().numpy()
        if self.collect_transitions and not self.deterministic:
            for state, allowed, offset, action in zip(x, feasible, bias, actions):
                self.connect(state, allowed, offset)
                self.pending = (state.copy(), allowed.copy(), offset.copy(), int(action), 0., self.slot)
        return actions, []

    def remember(self, reward, records, done):
        values = np.asarray(reward)
        if values.ndim and not np.allclose(values, values.flat[0]):
            raise ValueError('Requires the same scalar team reward as other methods')
        if self.pending is not None:
            x, mask, bias, action, accrued, start = self.pending
            accrued += self.cfg['gamma']**(self.slot-start)*float(values.flat[0])
            self.pending = (x, mask, bias, action, accrued, start)
            if done:
                self.replay.append((x, mask, bias, action, accrued, x.copy(), mask.copy(), bias.copy(), 0.))
                self.pending = None
        elif float(values.flat[0]) != 0.:
            raise ValueError('Nonzero reward without an earlier controllable action')
        self.slot += 1

    def update(self, context=None, glob=None):
        c = self.cfg
        if len(self.replay) < c['batch']:
            return {}
        ix = np.random.choice(len(self.replay), c['batch'], replace=False)
        arrays = list(map(np.stack, zip(*(self.replay[i] for i in ix))))
        x, m, bias, a, reward, nx, nm, nbias, discount = [
            tensor(v, self.device, torch.bool if i in (1, 6) else torch.long if i == 3 else torch.float32)
            for i, v in enumerate(arrays)]
        with torch.no_grad():
            probs, logp = self.distribution(nx, nm, nbias)
            next_q = torch.minimum(*[net(nx) for net in self.target_q])
            value = (probs*(next_q-self.log_alpha.exp()*logp)).sum(-1)
            target = reward+discount*value
        qloss = sum((net(x).gather(1, a[:, None]).squeeze(1)-target).square().mean() for net in self.q)
        grad_step(qloss, self.copt, self.q.parameters(), c['max_grad_norm'])
        probs, logp = self.distribution(x, m, bias)
        with torch.no_grad():
            minimum = torch.minimum(*[net(x) for net in self.q])
        actor_loss = (probs*(self.log_alpha.exp().detach()*logp-minimum)).sum(-1).mean()
        grad_step(actor_loss, self.aopt, self.actor.parameters(), c['max_grad_norm'])
        entropy = -(probs.detach()*logp.detach()).sum(-1)
        target_entropy = c['entropy_fraction']*m.sum(-1).float().log()
        tloss = (self.log_alpha*(entropy-target_entropy)).mean()
        grad_step(tloss, self.topt, [self.log_alpha], c['max_grad_norm'])
        polyak(self.q, self.target_q, c['tau'])
        self.updates += 1
        return dict(critic_loss=float(qloss), actor_loss=float(actor_loss),
            entropy=float(entropy.mean()), alpha=float(self.log_alpha.exp()), updates=self.updates)

    def state(self):
        return copy.deepcopy(dict(actor=self.actor.state_dict(), q=self.q.state_dict(),
            target_q=self.target_q.state_dict(), log_alpha=self.log_alpha.detach().cpu(),
            aopt=self.aopt.state_dict(), copt=self.copt.state_dict(), topt=self.topt.state_dict(),
            replay=self.replay, pending=self.pending, slot=self.slot, updates=self.updates,
            glob=getattr(self, 'glob', None)))

    def load(self, state):
        for name in ('actor', 'q', 'target_q', 'aopt', 'copt', 'topt'):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'].to(self.device))
        for name in ('replay', 'pending', 'slot', 'updates', 'glob'):
            setattr(self, name, copy.deepcopy(state[name]))


# Separate user-WLO, class-group gate and local/BS/LEO route preferences.
INTENT = 6+N_ROUTE


class RecurrentIntentActor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(OBS_DIM+INTENT, hidden, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(.01), nn.Linear(hidden, INTENT), nn.Tanh())
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, history):
        b, n, t, d = history.shape
        z, _ = self.lstm(history.reshape(b*n, t, d))
        return self.head(z[:, -1]).reshape(b, n, INTENT)


class RecurrentAttentionCritic(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        if hidden % 2:
            raise ValueError('Two attention heads require even hidden size')
        self.lstm = nn.LSTM(OBS_DIM+INTENT, hidden, num_layers=2, batch_first=True)
        self.embed = nn.Linear(hidden+INTENT, hidden)
        self.attention = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.LeakyReLU(.01), nn.Linear(hidden, 1))

    def forward(self, history, action):
        b, n, t, d = history.shape
        if n < 2:
            raise ValueError('Multi-agent attention requires at least two entities')
        z, _ = self.lstm(history.reshape(b*n, t, d))
        own = torch.nn.functional.leaky_relu(
            self.embed(torch.cat((z[:, -1].reshape(b, n, -1), action), -1)), .01)
        other, _ = self.attention(own, own, own,
            attn_mask=torch.eye(n, device=own.device, dtype=torch.bool), need_weights=False)
        return self.head(torch.cat((own, other), -1)).squeeze(-1)


class RecurrentAttentionTD3:
    """Coarse continuous-intent adaptation with common fixed resource sharing.

    User and region routing entities share an actor; critics see their public
    joint histories and continuous intents. Actual route feasibility and the
    shared reported-position physical-cost prior are applied when decoding.
    This does not claim continuous CPU/bandwidth allocation from the paper.
    """
    def __init__(self, cfg, env_cfg, device='cpu'):
        self.cfg, self.env_cfg, self.device = dict(cfg), env_cfg, torch.device(device)
        self.actor = RecurrentIntentActor(cfg['hidden']).to(device)
        self.q = nn.ModuleList([RecurrentAttentionCritic(cfg['hidden']) for _ in range(2)]).to(device)
        self.target_actor, self.target_q = copy.deepcopy(self.actor), copy.deepcopy(self.q)
        self.aopt = recurrent_optimizer(self.actor, cfg['actor_lr'])
        self.copt = recurrent_optimizer(self.q, cfg['critic_lr'])
        self.replay = CompactReplay(cfg['replay_capacity'],cfg['history'])
        self.history = self.prev = self.pending = None
        self.updates = self.steps = 0
        self.deterministic = False

    def reset_history(self):
        if self.pending is not None:
            raise ValueError('Flush terminal replay before resetting')
        self.history = self.prev = None

    def begin(self, context, glob, n_user):
        if self.history is None:
            self.history = np.zeros((len(context), self.cfg['history'], OBS_DIM+INTENT), np.float32)
            self.prev = np.zeros((len(context), INTENT), np.float32)
        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = np.concatenate((context, self.prev), axis=1)
        self.current = self.history.copy()
        if self.pending is not None:
            old, action, reward = self.pending
            self.replay.append((old, action, reward, self.current.copy(), False))
            self.pending = None
        with torch.no_grad():
            self.action = self.actor(tensor(self.current[None], self.device))[0].cpu().numpy()
        if not self.deterministic:
            self.action = np.clip(self.action+np.random.normal(0, self.cfg['exploration'], self.action.shape), -1, 1).astype(np.float32)
        self.prev = self.action.copy()

    def act(self, obs, mask, stage, agents):
        if stage not in ('user', 'route') or not np.asarray(mask).any(axis=1).all():
            raise ValueError('Unknown stage or empty action mask')
        if stage == 'route':
            preferences = self.action[agents, 6:]
        else:
            group = np.asarray(obs)[:, 63] < 0
            preferences = np.where(group[:, None], self.action[agents, 3:6], self.action[agents, :3])
        offset = prior(stage, tensor(obs, self.device), self.env_cfg,
            self.cfg['actor_cost_prior_scale']).detach().cpu().numpy()
        return np.argmax(np.where(mask, preferences*self.cfg['residual_scale']+offset, -np.inf), axis=1), []

    def remember(self, reward, records, done):
        values = np.asarray(reward)
        if values.ndim and not np.allclose(values, values.flat[0]):
            raise ValueError('Requires scalar team reward')
        reward = float(values.flat[0])
        if done:
            self.replay.append((self.current.copy(), self.action.copy(), reward, np.zeros_like(self.current), True))
        else:
            self.pending = (self.current.copy(), self.action.copy(), reward)
        self.steps += 1

    def update(self, context=None, glob=None):
        c = self.cfg
        if len(self.replay) < c['batch']:
            return {}
        eligible=self.replay.indices_for_entities(self.current.shape[0])
        if len(eligible)<c['batch']:return {}
        ix = np.random.choice(eligible, c['batch'], replace=False)
        h, a, reward, nh, done = [tensor(v, self.device) for v in
            map(np.stack, zip(*(self.replay[i] for i in ix)))]
        with torch.no_grad():
            noise = (torch.randn_like(a)*c['target_noise']).clamp(-c['noise_clip'], c['noise_clip'])
            next_action = (self.target_actor(nh)+noise).clamp(-1, 1)
            target = reward[:, None]+c['gamma']*(1-done[:, None])*torch.minimum(
                *[net(nh, next_action) for net in self.target_q])
        loss = sum((net(h, a)-target).square().mean() for net in self.q)
        grad_step(loss, self.copt, self.q.parameters(), c['max_grad_norm'])
        self.updates += 1
        result = dict(critic_loss=float(loss), updates=self.updates)
        if self.updates % c['policy_delay'] == 0:
            for p in self.q.parameters():
                p.requires_grad_(False)
            try:
                actor_loss = -self.q[0](h, self.actor(h)).mean()
                grad_step(actor_loss, self.aopt, self.actor.parameters(), c['max_grad_norm'])
            finally:
                for p in self.q.parameters():
                    p.requires_grad_(True)
            polyak(self.actor, self.target_actor, c['tau'])
            polyak(self.q, self.target_q, c['tau'])
            result['actor_loss'] = float(actor_loss)
        return result

    def state(self):
        result = {name: getattr(self, name).state_dict() for name in
            ('actor', 'q', 'target_actor', 'target_q', 'aopt', 'copt')}
        result.update({name: getattr(self, name, None) for name in
            ('replay', 'pending', 'history', 'prev', 'current', 'action', 'steps', 'updates')})
        return copy.deepcopy(result)

    def load(self, state):
        for name in ('actor', 'q', 'target_actor', 'target_q', 'aopt', 'copt'):
            getattr(self, name).load_state_dict(state[name])
        for name in ('replay', 'pending', 'history', 'prev', 'current', 'action', 'steps', 'updates'):
            setattr(self, name, copy.deepcopy(state[name]))


def recurrent_optimizer(module, lr):
    recurrent, heads = [], []
    for name,p in module.named_parameters():
        (recurrent if '.lstm.' in '.'+name else heads).append(p)
    return torch.optim.Adam([{'params':recurrent,'lr':lr*.1,'ratio':.1},
                             {'params':heads,'lr':lr,'ratio':1.}])


class CompactReplay:
    """Store chronological frames once per transition, reconstruct causal history.

    Two frames/transition replace two entire H-step windows. Samples whose past
    was evicted are excluded; histories never cross a true episode boundary.
    """
    def __init__(self,capacity,history):
        self.rows=deque(maxlen=capacity)
        self.history=history
        self.serial=0
        self.episode=0

    def append(self,transition):
        h,a,r,nh,done=transition
        self.rows.append((self.serial,self.episode,h[:,-1].copy(),a.copy(),r,
                          nh[:,-1].copy(),bool(done)))
        self.serial+=1
        if done:self.episode+=1

    def valid(self):
        rows=self.rows
        return [i for i,r in enumerate(rows) if i>=self.history-1 or
                (rows[0][0]==0 or (i>0 and any(rows[j][1]!=r[1] for j in range(i))))]

    def __len__(self):
        return len(self.valid())

    def indices_for_entities(self,n):
        return [i for i,pos in enumerate(self.valid()) if self.rows[pos][2].shape[0]==n]

    def __getitem__(self,index):
        rows=self.rows
        i=self.valid()[index]
        serial,episode,frame,a,r,nframe,done=rows[i]
        result=np.zeros((frame.shape[0],self.history,frame.shape[1]),np.float32)
        for offset in range(self.history):
            j=i-offset
            if j<0 or rows[j][1]!=episode:break
            result[:,-1-offset]=rows[j][2]
        next_history=np.zeros_like(result) if done else np.concatenate((result[:,1:],nframe[:,None]),axis=1)
        return result,a,r,next_history,done

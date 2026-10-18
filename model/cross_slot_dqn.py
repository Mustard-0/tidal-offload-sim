"""MATORA-DQN architecture adaptation, NOT the original joint allocator.

Distributed local Q decisions, replay, target network and epsilon-greedy
exploration follow the DQN offloading component described by Jia et al. (2025)
and the same group's CN116405962B. Resource allocation, queues, objective,
privacy and feasible nodes remain the common simulator, not their convex
allocator. WLO/group/route are discrete adapted heads. This is not renamed PPO.
"""
from collections import deque
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from environment import OBS_DIM, N_ROUTE
from learners import mlp


class CrossSlotDQN:
    schema = "matora-dqn-offloading-adaptation-v1"

    def __init__(self, method, cfg, env_cfg, device="cpu"):
        self.method, self.cfg, self.env_cfg = method, cfg, env_cfg
        self.device = torch.device(device)
        self.q = nn.ModuleDict({"user": mlp(OBS_DIM, cfg["hidden"], 3),
                                "route": mlp(OBS_DIM, cfg["hidden"], N_ROUTE)}).to(device)
        rng = np.random.default_rng(cfg["actor_init_seed"])
        with torch.no_grad():
            for net in self.q.values():
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        gain = .01 if layer is net[-1] else 1.
                        bound = gain*np.sqrt(6./(layer.in_features+layer.out_features))
                        layer.weight.copy_(torch.as_tensor(rng.uniform(-bound, bound, tuple(layer.weight.shape)),
                                                          dtype=torch.float32, device=device))
                        layer.bias.zero_()
        self.target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["critic_lr"])
        self.replay = deque(maxlen=16384)
        self.pending = {}
        self.steps, self.updates, self.progress = 0, 0, 0.
        self.deterministic = False

    def tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def set_training_clock(self, *args):
        pass

    def set_progress(self, update, total):
        self.progress = (update-1)/max(total-1, 1)
        for group in self.opt.param_groups:
            group["lr"] = self.cfg["critic_lr"]*(1-.9*self.progress)

    def reset_history(self):
        if self.pending and not self.deterministic:
            raise ValueError("Cannot discard unclosed DQN event returns")
        self.pending = {}

    def begin(self, context, glob, n_user):
        pass

    def _close(self, key, observation, mask, done):
        prior = self.pending.pop(key, None)
        if prior is not None:
            prior.update(next_obs=observation.copy(), next_mask=mask.copy(), done=done)
            self.replay.append(prior)

    def act(self, obs, masks, stage, agents):
        if not np.all(np.any(masks, axis=1)):
            raise ValueError("DQN cannot choose from an empty feasible-action set")
        training = not self.deterministic and not getattr(self, "sample_evaluation", False)
        with torch.no_grad():
            q = self.q[stage](self.tensor(obs)).masked_fill(~self.tensor(masks, torch.bool), -1e9)
            if getattr(self, "sample_evaluation", False):
                actions = torch.distributions.Categorical(logits=q).sample().cpu().numpy()
            else:
                actions = q.argmax(-1).cpu().numpy()
        epsilon = max(.05, 1-.95*min(1., self.progress/.2))
        for i, (ob, mask, agent) in enumerate(zip(obs, masks, agents)):
            if training and np.random.random() < epsilon:
                actions[i] = np.random.choice(np.flatnonzero(mask))
            if training:
                key = (int(agent), stage, round(float(ob[10]), 6) if stage == "user" and ob[63] < 0 else 0.)
                self._close(key, ob, mask, False)
                self.pending[key] = dict(obs=ob.copy(), mask=mask.copy(), action=int(actions[i]),
                    stage=stage, agent=int(agent), reward=0., discount=1.)
        return actions, []

    def remember(self, reward, records, done):
        for event in self.pending.values():
            event["reward"] += event["discount"]*float(reward[event["agent"]])
            event["discount"] *= self.cfg["gamma"]
        self.steps += 1
        if done:
            for key, event in list(self.pending.items()):
                self._close(key, event["obs"], event["mask"], True)

    def update(self, context=None, glob=None):
        losses = []
        if self.steps >= self.cfg.get("actor_warmup_steps", 2048) and len(self.replay) >= 128:
            for _ in range(max(1, self.cfg["rollout_steps"]//8)):
                chosen = np.random.choice(len(self.replay), 128, replace=False)
                rows = [self.replay[int(i)] for i in chosen]
                self.opt.zero_grad()
                loss = self.tensor(0.)
                for stage in ("user", "route"):
                    sub = [r for r in rows if r["stage"] == stage]
                    if not sub:
                        continue
                    obs = self.tensor(np.stack([r["obs"] for r in sub]))
                    prediction = self.q[stage](obs).gather(1, self.tensor(
                        [r["action"] for r in sub], torch.long)[:, None]).squeeze(1)
                    with torch.no_grad():
                        nxt = self.target[stage](self.tensor(np.stack([r["next_obs"] for r in sub])))
                        nxt = nxt.masked_fill(~self.tensor(np.stack([r["next_mask"] for r in sub]), torch.bool), -1e9).max(1).values
                        target = self.tensor([r["reward"] for r in sub])+self.tensor(
                            [r["discount"]*(not r["done"]) for r in sub])*nxt
                    loss = loss+len(sub)/len(rows)*F.smooth_l1_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DQN target/loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg["max_grad_norm"])
                self.opt.step()
                self.updates += 1
                if self.updates % 200 == 0:
                    self.target.load_state_dict(self.q.state_dict())
                losses.append(loss.item())
        return dict(dqn_loss=float(np.mean(losses)) if losses else 0.,
                    dqn_updates=float(self.updates), dqn_replay=float(len(self.replay)))

    def state(self, for_evaluation=False):
        result = dict(schema=self.schema, q=self.q.state_dict(), target=self.target.state_dict(),
                      steps=self.steps, updates=self.updates, progress=self.progress)
        if not for_evaluation:
            result.update(opt=self.opt.state_dict(), replay=copy.deepcopy(self.replay),
                          pending=copy.deepcopy(self.pending))
        return result

    def load(self, state):
        if state.get("schema") != self.schema:
            raise ValueError("Not a DQN adaptation checkpoint")
        self.q.load_state_dict(state["q"])
        self.target.load_state_dict(state["target"])
        self.steps, self.updates, self.progress = state["steps"], state["updates"], state["progress"]
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
            self.replay = copy.deepcopy(state["replay"])
            self.pending = copy.deepcopy(state["pending"])

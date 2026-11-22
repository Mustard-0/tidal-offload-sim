"""Use the actual task observation for each discrete TD3 decision and update."""

import numpy as np
import torch
from torch.nn import functional as F

from fresh_baseline_learners import JOINT, OBS, blend, straight_through, tensor, update_net
from public_baseline_updates import PublicTD3, TD3_DELAY, TD3_TAU, td3_score_noise, td3_target


STAGES = {"user": (0, 3), "route": (3, JOINT)}


def decision_history(history, agents, obs):
    states = history[agents].copy()
    states[:, -1, :OBS] = obs
    return states


def projected_actions(actor, histories, events, *, smooth=False):
    """Rebuild each event before averaging decisions for the slot-level critic."""
    batches, agents, observations, masks = [], [], [], []
    for i, slot in enumerate(events):
        for a, o, m in slot:
            batches.extend([i] * len(a))
            agents.extend(a)
            observations.append(o)
            masks.append(m)
    b, n = histories.shape[:2]
    if not agents:
        return histories.new_zeros((b, n, JOINT))
    batch_index = torch.as_tensor(batches, dtype=torch.long)
    agent_index = torch.as_tensor(agents, dtype=torch.long)
    h = histories[batch_index, agent_index].clone()
    h[:, -1, :OBS] = tensor(np.concatenate(observations))
    scores = actor(h[:, None, :, :])[:, 0, :]
    if smooth:
        scores = td3_score_noise(scores, torch.randn_like(scores))
    choices = straight_through(scores, torch.as_tensor(np.concatenate(masks), dtype=torch.bool))
    offset = batch_index * n + agent_index
    sums = choices.new_zeros((b * n, JOINT)).index_add_(0, offset, choices)
    counts = torch.bincount(offset, minlength=b * n).clamp_min(1)
    return (sums / counts[:, None]).reshape(b, n, JOINT)


class TaskAwareTD3(PublicTD3):
    def begin(self, context, glob, n_user):
        if self.history is None:
            self.history = np.zeros((len(context), 4, OBS + JOINT), dtype=np.float32)
            self.prev = np.zeros((len(context), JOINT), dtype=np.float32)
        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = np.concatenate((context, self.prev), axis=1)
        self.actions = np.zeros_like(self.prev)
        self.prev = np.zeros_like(self.actions)
        self.counts = np.zeros(len(context), dtype=np.int64)
        self.events = []

    @torch.no_grad()
    def act(self, obs, mask, stage, agents):
        start, end = STAGES[stage]
        h = decision_history(self.history, agents, obs)
        scores = self.actor(tensor(h[:, None]))[:, 0, start:end].numpy()
        if self.training:
            scores += np.random.normal(0., .2, scores.shape)
        selected = np.where(mask, scores, -np.inf).argmax(-1)
        if self.training:
            padded = np.zeros((len(mask), JOINT), dtype=bool)
            padded[:, start:end] = mask
            self.events.append((agents.copy(), obs.copy(), padded))
        for idx, a in zip(agents, selected):
            self.actions[idx, start+a] += 1
            self.counts[idx] += 1
            self.prev[idx] = self.actions[idx] / self.counts[idx]
        return selected, []

    def finish(self, reward, done):
        h, events = self.history.copy(), tuple(self.events)
        if self.pending is not None:
            self.replay.append((*self.pending, h, events, False))
        self.pending = (h, self.prev.copy(), events, reward)
        if done:
            self.replay.append((*self.pending, h, (), True))
            self.pending = None

    def learn(self, context, glob):
        if len(self.replay) < 16:
            return {}
        result = {}
        for _ in range(6):
            ids = np.random.choice(len(self.replay), 16, replace=False)
            h, a, events, r, nh, next_events, done = zip(*(self.replay[i] for i in ids))
            h, a, nh = (tensor(np.stack(column)) for column in (h, a, nh))
            with torch.no_grad():
                next_action = projected_actions(self.actor_target, nh, next_events, smooth=True)
                q1, q2 = (net(nh, next_action) for net in self.targets)
                target = td3_target(tensor(r), tensor(done), q1, q2)
            critic = sum(F.mse_loss(net(h, a), target) for net in self.critic)
            result["critic_loss"] = update_net(critic, self.copt, self.critic)
            self.updates += 1
            if self.updates % TD3_DELAY == 0:
                if any(events):
                    self.critic[0].requires_grad_(False)
                    try:
                        actions = projected_actions(self.actor, h, events)
                        loss = -self.critic[0](h, actions).mean()
                        result["actor_loss"] = update_net(loss, self.aopt, self.actor)
                    finally:
                        self.critic[0].requires_grad_(True)
                blend(self.targets, self.critic, TD3_TAU)
                blend(self.actor_target, self.actor, TD3_TAU)
        return result

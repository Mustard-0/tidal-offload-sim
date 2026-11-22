"""Public learner update rules adapted to masked STIN actions.

The PPO policy/value objectives follow CleanRL ppo.py at fe8d8a0 (MIT).
TD3 target and update scheduling follow sfujim/TD3 at 34770cc (MIT).
See the accompanying CleanRL and TD3 license texts.
The SDSAC equations are independently implemented from the TMLR paper and
checked against the corresponding primitives in coldsummerday/SD-SAC at 006809d.
"""

import torch
from torch.nn import functional as F

from fresh_baseline_learners import (
    ClippedPPO, GAMMA, RecurrentAttentionTD3, StableDiscreteSAC, advantages,
    blend, columns, distribution, np, straight_through, tensor, update_net,
)


TD3_TAU = .005
TD3_DELAY = 2


def ppo_policy_loss(log_ratio, adv, clip=.2):
    ratio = log_ratio.clamp(-20., 20.).exp()
    return torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip)).mean()


def ppo_value_loss(value, rollout_value, target, clip=.2):
    clipped = rollout_value + (value - rollout_value).clamp(-clip, clip)
    return .5 * torch.maximum((value - target).square(), (clipped - target).square()).mean()


class PublicPPO(ClippedPPO):
    def __init__(self):
        super().__init__(centralized=True)

    def learn(self, context, glob):
        views, decisions, rewards, dones = zip(*self.steps)
        self.steps.clear()
        inputs = tensor(np.stack(views))
        with torch.no_grad():
            rollout_v = self.critic(inputs).squeeze(-1)
            tail = self.critic(tensor(glob[None])).squeeze(-1)
            adv = tensor(advantages(rewards, rollout_v.numpy(), dones, tail.numpy()))
            returns = rollout_v + adv
        actions = [(slot, *record[:4]) for slot, group in enumerate(decisions)
                   for record in group if record[1].sum() > 1]
        if actions:
            slot, features, legal, choice, old_logp = columns(actions)
            slot = torch.as_tensor(slot, dtype=torch.long)
            active = slot.unique()
            a = adv[active, 0]
            a = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)
            features, old_logp = tensor(features), tensor(old_logp)
            choice = torch.as_tensor(choice, dtype=torch.long)
        result = {}
        for _ in range(2 if actions else 1):
            value = self.critic(inputs).squeeze(-1)
            result["critic_loss"] = update_net(ppo_value_loss(value, rollout_v, returns), self.copt, self.critic)
            if actions:
                d = distribution(self.actor(features), legal)
                delta = torch.zeros(len(views)).index_add_(0, slot, d.log_prob(choice) - old_logp)[active]
                loss = ppo_policy_loss(delta, a) - .01 * d.entropy().mean()
                result["actor_loss"] = update_net(loss, self.aopt, self.actor)
        return result


def sdsac_value(dist, q1, q2, alpha):
    return (dist.probs * ((q1 + q2) * .5)).sum(dim=-1) + alpha * dist.entropy()


def sdsac_q_loss(q, ref, target, limit=.5):
    clipped = ref + (q - ref).clamp(-limit, limit)
    return torch.maximum(F.mse_loss(q, target), F.mse_loss(clipped, target))


class PublicSDSAC(StableDiscreteSAC):
    def learn(self, context, glob):
        if len(self.replay) < 128:
            return {}
        for _ in range(6):
            selected = np.random.choice(len(self.replay), 128, replace=False)
            x, mask, action, entropy, reward, nx, nmask, discount = columns([self.replay[i] for i in selected])
            x, nx = tensor(x), tensor(nx)
            action = torch.as_tensor(action, dtype=torch.long)[:, None]
            with torch.no_grad():
                d = distribution(self.actor(nx), nmask)
                q1, q2 = (model(nx) for model in self.targets)
                target = tensor(reward) + tensor(discount) * sdsac_value(d, q1, q2, self.alpha)
                refs = [model(x).gather(1, action).flatten() for model in self.targets]
            critic = sum(sdsac_q_loss(net(x).gather(1, action).flatten(), old, target)
                         for net, old in zip(self.critic, refs))
            critic_loss = update_net(critic, self.copt, self.critic)
            d = distribution(self.actor(x), mask)
            with torch.no_grad():
                q1, q2 = (net(x) for net in self.critic)
            penalty = .25 * F.mse_loss(d.entropy(), tensor(entropy))
            actor = -sdsac_value(d, q1, q2, self.alpha).mean() + penalty
            actor_loss = update_net(actor, self.aopt, self.actor)
            blend(self.targets, self.critic)
        return dict(actor_loss=actor_loss, critic_loss=critic_loss, entropy_penalty=float(penalty.detach()))


def td3_score_noise(scores, standard_normal):
    noise = (standard_normal * .2).clamp(-.5, .5)
    return (scores + noise).clamp(-4., 4.)


def td3_target(reward, done, twin_1, twin_2):
    return reward + (1. - done) * GAMMA * torch.minimum(twin_1, twin_2)


class PublicTD3(RecurrentAttentionTD3):
    def learn(self, context, glob):
        if len(self.replay) < 16:
            return {}
        result = {}
        for _ in range(6):
            selected = np.random.choice(len(self.replay), 16, replace=False)
            hist, action, mask, r, next_hist, next_mask, done = columns([self.replay[i] for i in selected])
            hist, action, next_hist = tensor(hist), tensor(action), tensor(next_hist)
            with torch.no_grad():
                scores = self.actor_target(next_hist)
                smoothed = td3_score_noise(scores, torch.randn_like(scores))
                next_action = straight_through(smoothed, torch.as_tensor(next_mask))
                q1, q2 = (net(next_hist, next_action) for net in self.targets)
                target = td3_target(tensor(r), tensor(done), q1, q2)
            critic_loss = sum(F.mse_loss(net(hist, action), target) for net in self.critic)
            result["critic_loss"] = update_net(critic_loss, self.copt, self.critic)
            self.updates += 1
            if self.updates % TD3_DELAY == 0:
                self.critic[0].requires_grad_(False)
                try:
                    choice = straight_through(self.actor(hist), torch.as_tensor(mask))
                    result["actor_loss"] = update_net(-self.critic[0](hist, choice).mean(), self.aopt, self.actor)
                finally:
                    self.critic[0].requires_grad_(True)
                blend(self.targets, self.critic, TD3_TAU)
                blend(self.actor_target, self.actor, TD3_TAU)
        return result


def make_controller(method):
    return {"PPO": PublicPPO, "SD-SAC-Adapted": PublicSDSAC, "MARATD3-Adapted": PublicTD3}[method]()

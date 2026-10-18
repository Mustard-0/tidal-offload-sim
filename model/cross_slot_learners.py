"""Cross-slot adapters; legacy learners and old experiment outputs stay intact.

The TD3 adapter uses recurrent decision histories, twin attention critics,
delayed updates and target smoothing. Discrete actions use an explicitly
registered surrogate, not the original paper's continuous resource allocation.
"""
from collections import deque
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from environment import OBS_DIM, N_ROUTE
from learners import PPO, MATD3, STAGE_SLICES, CONTROL_DIM, cost_prior


def cross_slot_prior(stage, obs, cfg, scale):
    if stage != "user":
        return cost_prior(stage, obs, cfg, scale)
    # Negative obs[63] identifies a class gate; the shared W/L/O actor is
    # retained, but LOCAL is masked and obs[17] now estimates WAIT cost.
    group = obs[:, 63] < 0
    wait = torch.where(group, obs[:, 17], obs[:, 26])
    return -scale * torch.stack((wait, obs[:, 17], obs[:, 18]), dim=-1)


def training_clock(remaining, arrivals_remaining, total):
    if total <= 0 or not 0 <= arrivals_remaining <= remaining <= total:
        raise ValueError("Invalid finite training horizon")
    return np.asarray([remaining/total, arrivals_remaining/total], dtype=np.float32)


class CrossSlotPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.training_clock = np.zeros(2, np.float32)
        # The paper's TMPPO label retains TMAPO as its internal factory key.
        if self.cfg.get("shared_team_value", False) and self.method != "TMAPO":
            raise ValueError("Shared centralized value input is a TMPPO diagnostic only")
        frozen = self.cfg.get("frozen_actor_stages", [])
        if frozen not in ([], ["user"], ["route"]):
            raise ValueError("Unregistered actor-stage ablation")

    def set_training_clock(self, remaining, arrivals_remaining, total):
        self.training_clock = training_clock(remaining, arrivals_remaining, total)

    def critic_input(self, context, glob):
        if self.cfg.get("shared_team_value", False):
            context = np.broadcast_to(np.mean(context, axis=0), context.shape)
        inputs = super().critic_input(context, glob)
        if self.cfg.get("critic_clock_features", 0):
            inputs = np.concatenate([inputs, np.broadcast_to(self.training_clock,
                                                              (len(inputs), 2))], axis=1)
        return inputs

    def remember(self, reward, records, done):
        super().remember(reward, records, done)
        self.steps += 1

    def state(self, for_evaluation=False):
        result = super().state(for_evaluation)
        result.update(training_steps=self.steps, training_clock=self.training_clock.copy(),
                      shared_team_value=self.cfg.get("shared_team_value", False),
                      frozen_actor_stages=list(self.cfg.get("frozen_actor_stages", [])))
        return result

    def load(self, state):
        if state.get("shared_team_value", False) != self.cfg.get("shared_team_value", False):
            raise ValueError("Checkpoint centralized value-input contract differs")
        if state.get("frozen_actor_stages", []) != self.cfg.get("frozen_actor_stages", []):
            raise ValueError("Checkpoint actor-stage ablation contract differs")
        super().load(state)
        self.steps = state.get("training_steps", 0)
        self.training_clock = np.asarray(state.get("training_clock", [0., 0.]), np.float32).copy()

    def policy_logits(self, stage, obs):
        return self.actors[stage](obs) + cross_slot_prior(
            stage, obs, self.env_cfg, self.cfg["actor_cost_prior_scale"])

    def normalize_targets(self, targets):
        # PopArt output compensation preserves raw values when moments change.
        old_mean, old_std = self.value_norm.mean, self.value_norm.std
        result = super().normalize_targets(targets)
        if self.value_norm.enabled:
            layer = self.critic[-1]
            with torch.no_grad():
                layer.weight.mul_(old_std / self.value_norm.std)
                layer.bias.copy_((old_std * layer.bias + old_mean -
                                  self.value_norm.mean) / self.value_norm.std)
        return result

    def update(self, context, glob):
        counts = {name: 0 for name in ("user", "group", "route")}
        actions = {name: np.zeros(11 if name == "route" else 3, dtype=int)
                   for name in counts}
        advantages = {name: [] for name in counts}
        if self.cfg.get("record_advantage_diagnostics", False):
            next_value = self.raw_values(self.critic_input(context, glob))
            gae = np.zeros_like(next_value)
            for _, value, reward, records, done in reversed(self.buffer):
                delta = reward + self.cfg["gamma"] * next_value * (not done) - value
                gae = delta + self.cfg["gamma"] * self.cfg["gae_lambda"] * (not done) * gae
                for row in records:
                    role = ("group" if row["stage"] == "user" and row["obs"][63] < 0
                            else row["stage"])
                    if role in advantages:
                        advantages[role].append(float(gae[row["agent"]]))
                next_value = value
        for _, _, _, records, _ in self.buffer:
            for row in records:
                name = ("group" if row["stage"] == "user" and row["obs"][63] < 0
                        else row["stage"])
                if name in counts:
                    counts[name] += 1
                    actions[name][row["action"]] += 1
        result = super().update(context, glob)
        result["actor_frozen"] = float(self.steps < self.cfg.get("actor_warmup_steps", 0))
        for stage in ("user", "route"):
            result[stage + "_optimizer_frozen"] = float(stage in self.cfg.get("frozen_actor_stages", []))
        for name in counts:
            result[name + "_trainable_decisions"] = counts[name]
            for action, count in enumerate(actions[name]):
                result["%s_action_%d_fraction" % (name, action)] = count / max(counts[name], 1)
            if self.cfg.get("record_advantage_diagnostics", False):
                result[name + "_raw_adv_mean"] = float(np.mean(advantages[name])) if advantages[name] else 0.
                result[name + "_raw_adv_std"] = float(np.std(advantages[name])) if advantages[name] else 0.
        return result


class CrossSlotTD3(MATD3):
    """Event-time discrete TD3 adaptation, with an explicit discrete surrogate.

    Each (agent, stage, public class) stream has its own recurrent history.
    Transitions close at its next actual decision, accumulating discounted
    shared/local rewards over elapsed global slots. A route event therefore
    cannot accidentally replay a class gate's observation or unused action.
    The critic consumes the executed one-hot action, not an unexecuted vector.
    """
    schema = "cross-slot-discrete-td3-v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.histories = {}
        self.open_events = {}
        self.previous_actions = {}
        self.slot_records = []
        self.clock = 0
        self.training_clock = np.zeros(2, np.float32)

    def set_training_clock(self, remaining, arrivals_remaining, total):
        self.training_clock = training_clock(remaining, arrivals_remaining, total)

    def reset_history(self):
        self.histories = {}
        self.open_events = {}
        self.previous_actions = {}
        self.slot_records = []

    def begin(self, context, glob, n_user):
        self.n_user = n_user
        self.global_obs = np.asarray(glob, dtype=np.float32).copy()
        if self.cfg.get("critic_clock_features", 0):
            self.global_obs = np.r_[self.global_obs, self.training_clock]
        self.slot_records = []

    def _key(self, agent, stage, obs):
        return (int(agent), stage, round(float(obs[10]), 6)
                if stage == "user" and obs[63] < 0 else 0.)

    def _close(self, key, history, glob, mask, done=False):
        prior = self.open_events.pop(key, None)
        if prior is not None:
            prior.update(next_history=history.copy(), next_global=glob.copy(),
                         next_mask=mask.copy(), done=bool(done))
            self.replay.append(prior)

    def act(self, obs, mask, stage, agents):
        histories, keys = [], []
        for ob, allow, agent in zip(obs, mask, agents):
            key = self._key(agent, stage, ob)
            h = self.histories.get(key, np.zeros(
                (self.cfg["td3_history"], OBS_DIM + CONTROL_DIM), np.float32)).copy()
            action = self.previous_actions.get(key, np.zeros(CONTROL_DIM, np.float32))
            h = np.roll(h, -1, axis=0)
            h[-1] = np.r_[ob, action]
            if not self.deterministic and not getattr(self, "sample_evaluation", False):
                self._close(key, h, self.global_obs, allow)
            self.histories[key] = h
            histories.append(h)
            keys.append(key)
        h = self.tensor(np.stack(histories))[:, None]
        with torch.no_grad():
            preference = self.actor(h)[:, 0, STAGE_SLICES[stage]]
            if not self.deterministic and not getattr(self, "sample_evaluation", False):
                if self.steps < self.cfg["td3_start_steps"]:
                    preference = torch.empty_like(preference).uniform_(-1, 1)
                else:
                    preference = (preference + torch.randn_like(preference) *
                                  self.cfg["td3_exploration"]).clamp(-1, 1)
            logits = preference * self.cfg.get("td3_residual_scale", 4.)
            logits += cross_slot_prior(stage, self.tensor(obs), self.env_cfg,
                                       self.cfg["actor_cost_prior_scale"])
            logits = logits.masked_fill(~self.tensor(mask).bool(), -1e9)
            if getattr(self, "sample_evaluation", False):
                actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            else:
                actions = logits.argmax(-1).cpu().numpy()
        for key, history, ob, allow, a, agent in zip(keys, histories, obs, mask, actions, agents):
            executed = np.zeros(CONTROL_DIM, np.float32)
            executed[STAGE_SLICES[stage].start + int(a)] = 1.
            self.previous_actions[key] = executed
            if not self.deterministic and not getattr(self, "sample_evaluation", False):
                self.open_events[key] = dict(
                    history=history.copy(), global_obs=self.global_obs.copy(),
                    mask=allow.copy(), stage=stage, action=executed,
                    reward=0., discount=1., elapsed=0, agent=int(agent))
        return actions, []

    def remember(self, reward, records, done):
        for event in self.open_events.values():
            event["reward"] += event["discount"] * float(reward[event["agent"]])
            event["discount"] *= self.cfg["gamma"]
            event["elapsed"] += 1
        self.steps += 1
        self.clock += 1
        if done:
            for key, event in list(self.open_events.items()):
                self._close(key, event["history"], event["global_obs"], event["mask"], True)
            self.reset_history()

    def _critic(self, q, history, glob, action, masks, stages):
        ob = history[:, -1, :OBS_DIM]
        # Global state and explicit feasible-action/stage token provide central
        # information; they never enter the decentralized recurrent actor.
        marker = torch.zeros_like(ob)
        for i, (mask, stage) in enumerate(zip(masks, stages)):
            marker[i, STAGE_SLICES[stage]] = mask
            marker[i, 32] = float(stage == "route")
        if self.cfg.get("critic_clock_features", 0):
            clock = glob[:, -2:]
            ob = torch.cat((ob, clock), dim=1)
            marker = torch.cat((marker, clock), dim=1)
        tokens = torch.stack((ob, glob, marker), dim=1)
        acts = torch.stack((action, torch.zeros_like(action), torch.zeros_like(action)), dim=1)
        return q(tokens, acts)[:, 0]

    def _actions(self, actor, history, masks, stages, smooth=False, straight_through=False):
        preferences = actor(history[:, None])[:, 0]
        out = []
        for i, (mask, stage) in enumerate(zip(masks, stages)):
            sl = STAGE_SLICES[stage]
            p = preferences[i, sl]
            if smooth:
                p = (p + (torch.randn_like(p) * self.cfg["td3_target_noise"]).clamp(
                    -self.cfg["td3_noise_clip"], self.cfg["td3_noise_clip"])).clamp(-1, 1)
            logits = p * self.cfg.get("td3_residual_scale", 4.)
            logits = logits + cross_slot_prior(stage, history[i:i+1, -1, :OBS_DIM],
                                               self.env_cfg, self.cfg["actor_cost_prior_scale"])[0]
            soft = logits.masked_fill(~mask.bool(), -1e9).softmax(-1)
            hard = F.one_hot(soft.argmax(), len(soft)).to(soft.dtype)
            a = hard + soft - soft.detach() if straight_through else hard
            out.append(F.pad(a, (sl.start, CONTROL_DIM - sl.stop)))
        return torch.stack(out)

    def categorical_actor_loss(self, history, glob, masks, stages):
        """Score only executed-action vertices, avoiding off-simplex Q gradients.

        This is a discrete categorical surrogate, not the original continuous
        TD3 policy gradient. Q values are constants in this actor objective.
        """
        preference = self.actor(history[:, None])[:, 0]
        losses = []
        for i, (mask, stage) in enumerate(zip(masks, stages)):
            sl = STAGE_SLICES[stage]
            allowed = torch.nonzero(mask.bool(), as_tuple=False).flatten()
            logits = preference[i, sl] * self.cfg.get("td3_residual_scale", 4.)
            logits = logits + cross_slot_prior(
                stage, history[i:i+1, -1, :OBS_DIM], self.env_cfg,
                self.cfg["actor_cost_prior_scale"])[0]
            probabilities = logits[allowed].softmax(-1)
            actions = F.pad(F.one_hot(allowed, len(mask)).float(),
                            (sl.start, CONTROL_DIM - sl.stop))
            count = len(allowed)
            with torch.no_grad():
                values = self._critic(
                    self.q[0], history[i:i+1].expand(count, -1, -1),
                    glob[i:i+1].expand(count, -1), actions,
                    [mask] * count, [stage] * count)
                values = values - values.mean()
            losses.append(-(probabilities * values).sum())
        return torch.stack(losses).mean()

    def update(self, context=None, glob=None):
        c = self.cfg
        if len(self.replay) < c["td3_batch"] or self.steps < c["td3_start_steps"]:
            return dict(td3_replay=len(self.replay), td3_updates=self.updates)
        losses, actor_losses = [], []
        for _ in range(max(1, c["rollout_steps"] // c["td3_update_every"])):
            rows = [self.replay[i] for i in np.random.choice(len(self.replay), c["td3_batch"], False)]
            get = lambda k: self.tensor(np.stack([r[k] for r in rows]))
            h, nh, g, ng, a = [get(k) for k in ("history", "next_history", "global_obs", "next_global", "action")]
            masks = [self.tensor(r["mask"]) for r in rows]
            nmasks = [self.tensor(r["next_mask"]) for r in rows]
            stages = [r["stage"] for r in rows]
            with torch.no_grad():
                na = self._actions(self.target_actor, nh, nmasks, stages, smooth=True)
                nq = torch.minimum(*(self._critic(q, nh, ng, na, nmasks, stages) for q in self.target_q))
                target = get("reward") + get("discount") * (1 - get("done")) * nq
            predictions = [self._critic(q, h, g, a, masks, stages) for q in self.q]
            if c.get("td3_huber", False):
                loss = sum(F.smooth_l1_loss(prediction, target) for prediction in predictions)
            else:
                loss = sum((prediction - target).square().mean() for prediction in predictions)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TD3 critic loss")
            self.copt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), c["max_grad_norm"])
            self.copt.step()
            losses.append(float(loss.detach()))
            self.updates += 1
            if (self.updates % c["td3_policy_delay"] == 0 and
                    self.steps >= c.get("actor_warmup_steps", 0)):
                for p in self.q.parameters():
                    p.requires_grad_(False)
                if c.get("td3_discrete_objective", "straight_through") == "categorical_expectation":
                    aloss = self.categorical_actor_loss(h, g, masks, stages)
                else:
                    pa = self._actions(self.actor, h, masks, stages, straight_through=True)
                    aloss = -self._critic(self.q[0], h, g, pa, masks, stages).mean()
                if not torch.isfinite(aloss):
                    raise FloatingPointError("Nonfinite TD3 actor objective")
                self.aopt.zero_grad()
                aloss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), c["max_grad_norm"])
                self.aopt.step()
                actor_losses.append(float(aloss.detach()))
                for p in self.q.parameters():
                    p.requires_grad_(True)
            if self.updates % c["td3_policy_delay"] == 0:
                with torch.no_grad():
                    for source, target_model in ((self.actor, self.target_actor), (self.q, self.target_q)):
                        for source_p, target_p in zip(source.parameters(), target_model.parameters()):
                            target_p.lerp_(source_p, c["td3_tau"])
        return dict(td3_critic_loss=float(np.mean(losses)), td3_updates=self.updates,
                    td3_actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.,
                    td3_replay=len(self.replay),
                    actor_frozen=float(self.steps < c.get("actor_warmup_steps", 0)))

    def state(self, for_evaluation=False):
        result = super().state(for_evaluation)
        result.update(schema=self.schema, clock=self.clock,
                      training_clock=self.training_clock.copy(),
                      discrete_objective=self.cfg.get("td3_discrete_objective", "straight_through"),
                      histories={} if for_evaluation else copy.deepcopy(self.histories),
                      open_events={} if for_evaluation else copy.deepcopy(self.open_events),
                      previous_actions={} if for_evaluation else copy.deepcopy(self.previous_actions))
        return result

    def load(self, state):
        if state.get("schema") != self.schema:
            raise ValueError("Legacy TD3 checkpoints are not decision-aligned")
        if state.get("discrete_objective", "straight_through") != self.cfg.get(
                "td3_discrete_objective", "straight_through"):
            raise ValueError("Checkpoint uses a different discrete actor objective")
        super().load(state)
        self.clock = state["clock"]
        self.training_clock = np.asarray(state.get("training_clock", [0., 0.]), np.float32).copy()
        self.histories = copy.deepcopy(state["histories"])
        self.open_events = copy.deepcopy(state["open_events"])
        self.previous_actions = copy.deepcopy(state["previous_actions"])

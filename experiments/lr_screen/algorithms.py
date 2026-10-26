"""Real decentralized CTDE, independent PPO, and centralized joint-action PPO.

Only already-public observations enter actors/critics. A common environment
provides identical admissible actions and physics. PPO is a single controller,
not the old release-only pilot or a renamed IPPO.
"""
import copy
from .runtime import np, torch
from torch import nn
from torch.distributions import Categorical
from environment import OBS_DIM, N_ROUTE
from learners import PPO as LegacyPPO, mlp, ReturnNormalizer, cost_prior


def prior(stage, obs, env, scale):
    if stage != "user":
        return cost_prior(stage, obs, env, scale)
    group = obs[:, 63] < 0
    wait = torch.where(group, obs[:, 17], obs[:, 26])
    return -scale * torch.stack((wait, obs[:, 17], obs[:, 18]), dim=-1)


def compensate(layer, normalizer, old_mean, old_std):
    if normalizer.enabled:
        with torch.no_grad():
            layer.weight.mul_(old_std / normalizer.std)
            layer.bias.copy_((old_std * layer.bias + old_mean - normalizer.mean) / normalizer.std)


class DecentralizedPPO(LegacyPPO):
    def __init__(self, method, cfg, env_cfg, device="cpu"):
        if method not in ("TMAPO", "IPPO"):
            raise ValueError("Use CentralPPO for plain PPO")
        super().__init__(method, cfg, env_cfg, device)
        del self.actors["batch"]  # Cross-slot class gate shares the W/L/O head.
        self.aopt = torch.optim.Adam(self.actors.parameters(), lr=cfg["actor_lr"])

    def policy_logits(self, stage, obs):
        return self.actors[stage](obs) + prior(stage, obs, self.env_cfg, self.cfg["actor_cost_prior_scale"])

    def normalize_targets(self, targets):
        mean, std = self.value_norm.mean, self.value_norm.std
        result = super().normalize_targets(targets)
        compensate(self.critic[-1], self.value_norm, mean, std)
        return result

    def update(self, context, glob):
        records = [r for _, _, _, rows, _ in self.buffer for r in rows]
        result = super().update(context, glob)
        return dict(result, **action_diagnostics(records))


def action_diagnostics(records):
    result = {}
    for kind in ("user", "group", "route"):
        rows = [r for r in records if ("group" if r["stage"] == "user" and r["obs"][63] < 0
                                      else r["stage"]) == kind]
        result[kind + "_trainable_decisions"] = len(rows)
        for a in range(N_ROUTE if kind == "route" else 3):
            result[f"{kind}_action_{a}_fraction"] = sum(r["action"] == a for r in rows) / max(1, len(rows))
    return result


class CentralPPO:
    """Factorized autoregressive action within each global slot.

    The joint likelihood is the PRODUCT of all feasible, stochastic decision
    factors in a slot. One clipped PPO surrogate uses their SUM of log ratios,
    one scalar team return and V(public-global). Fixed masks/recorded conditional
    observations make sequential resource decisions reproducible. No sum of
    independent per-agent surrogates is mislabeled as ordinary centralized PPO.
    Entropy regularization uses mean per-decision entropy, shared coefficient .01.
    """
    def __init__(self, cfg, env_cfg, device="cpu"):
        self.cfg, self.env_cfg, self.device = cfg, env_cfg, torch.device(device)
        h = cfg["hidden"]
        self.actors = nn.ModuleDict({"user": mlp(2*OBS_DIM, h, 3),
                                     "route": mlp(2*OBS_DIM, h, N_ROUTE)}).to(device)
        for actor in self.actors.values():
            nn.init.zeros_(actor[-1].weight)
            nn.init.zeros_(actor[-1].bias)
        self.critic = mlp(OBS_DIM, h, 1).to(device)
        self.value_norm = ReturnNormalizer(cfg["value_normalization"])
        self.aopt = torch.optim.Adam(self.actors.parameters(), lr=cfg["actor_lr"])
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.deterministic = False
        self.buffer = []

    def tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def raw_value(self, glob):
        with torch.no_grad():
            x = self.critic(self.tensor(np.asarray(glob)[None, :])).item()
        return self.value_norm.denormalize(x)

    def begin(self, context, glob, n_user):
        self.glob = np.asarray(glob, dtype=np.float32).copy()
        if not self.deterministic:
            self.value = self.raw_value(glob)

    def logits(self, stage, obs, globals):
        return self.actors[stage](torch.cat((obs, globals), dim=-1)) + prior(
            stage, obs, self.env_cfg, self.cfg["actor_cost_prior_scale"])

    def act(self, obs, mask, stage, agents):
        if not np.asarray(mask).any(axis=1).all():
            raise AssertionError("Empty action mask")
        with torch.no_grad():
            dist = Categorical(logits=self.logits(stage, self.tensor(obs),
                self.tensor(np.broadcast_to(self.glob, obs.shape).copy())).masked_fill(~self.tensor(mask, torch.bool), -1e9))
            a = dist.probs.argmax(-1) if self.deterministic else dist.sample()
            lp = dist.log_prob(a).cpu().numpy()
        a = a.cpu().numpy()
        rows = [] if self.deterministic else [dict(stage=stage, obs=o.copy(), mask=m.copy(),
                action=int(action), logp=float(logp), glob=self.glob.copy())
                for o, m, action, logp in zip(obs, mask, a, lp) if np.count_nonzero(m) > 1]
        return a, rows

    def remember(self, rewards, records, done):
        if not np.allclose(rewards, rewards[0], rtol=0, atol=1e-7):
            raise AssertionError("Central PPO requires the common team reward")
        self.buffer.append((self.glob.copy(), self.value, float(rewards[0]), copy.deepcopy(records), bool(done)))

    def set_progress(self, update, total):
        fraction = (update-1)/max(total-1, 1)
        scale = 1-(1-self.cfg["lr_final_fraction"])*fraction if self.cfg["linear_lr_decay"] else 1.
        for opt, key in ((self.aopt, "actor_lr"), (self.copt, "critic_lr")):
            for group in opt.param_groups:
                group["lr"] = self.cfg[key]*scale

    def update(self, context, glob):
        c = self.cfg
        next_value, gae = self.raw_value(glob), 0.
        rows, targets, advantages = [], [], []
        for g, v, reward, rec, done in reversed(self.buffer):
            gae = reward+c["gamma"]*next_value*(not done)-v + c["gamma"]*c["gae_lambda"]*(not done)*gae
            rows.append((g, rec))
            targets.append(gae+v)
            advantages.append(gae)
            next_value = v
        x = self.tensor(np.stack([g for g, _ in rows]))
        old_mean, old_std = self.value_norm.mean, self.value_norm.std
        self.value_norm.update(targets)
        y = self.tensor(self.value_norm.normalize(np.asarray(targets)))
        compensate(self.critic[-1], self.value_norm, old_mean, old_std)
        active = [i for i, (_, rec) in enumerate(rows) if rec]
        adv = np.asarray(advantages)
        if active:
            adv = (adv-adv[active].mean()) / (adv[active].std()+1e-8)
        adv = self.tensor(adv)
        # Preassemble conditional factors once; index_add builds joint slot logs.
        stages = {}
        for stage in ("user", "route"):
            factors = [(i, r) for i, (_, rec) in enumerate(rows) for r in rec if r["stage"] == stage]
            if factors:
                stages[stage] = (self.tensor([i for i, _ in factors], torch.long),
                    self.tensor(np.stack([r["obs"] for _, r in factors])),
                    self.tensor(np.stack([r["glob"] for _, r in factors])),
                    self.tensor(np.stack([r["mask"] for _, r in factors]), torch.bool),
                    self.tensor([r["action"] for _, r in factors], torch.long),
                    self.tensor([r["logp"] for _, r in factors]))
        stats = dict(kl=[], entropy=[], value_loss=[], clip_fraction=[])
        stopped = False
        for _ in range(c["epochs"]):
            for ix in torch.randperm(len(rows), device=self.device).split(c["minibatch"]):
                loss = .5*(self.critic(x[ix]).squeeze(-1)-y[ix]).square().mean()
                self.copt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c["max_grad_norm"])
                self.copt.step()
                stats["value_loss"].append(loss.item())
            if not active or stopped:
                continue
            # This protocol's rollout=128 <= minibatch=256: whole-slot batches.
            for selected in torch.randperm(len(active), device=self.device).split(c["minibatch"]):
                ix = self.tensor(active, torch.long)[selected]
                logratio = torch.zeros(len(rows), device=self.device)
                entropies = []
                for stage, (slot, obs, gs, mask, action, old) in stages.items():
                    chosen = torch.isin(slot, ix)
                    if not chosen.any():
                        continue
                    dist = Categorical(logits=self.logits(stage, obs[chosen], gs[chosen]).masked_fill(~mask[chosen], -1e9))
                    logratio = logratio.index_add(0, slot[chosen], dist.log_prob(action[chosen])-old[chosen])
                    entropies.append(dist.entropy())
                lr = logratio[ix]
                ratio = lr.exp()
                kl = ((ratio-1)-lr).mean()
                if not torch.isfinite(ratio).all():
                    raise FloatingPointError("Central joint likelihood overflow")
                stats["kl"].append(kl.item())
                if kl.item() > 1.5*c["target_kl"]:
                    stopped = True
                    break
                entropy = torch.cat(entropies).mean()
                loss = -torch.minimum(ratio*adv[ix], ratio.clamp(1-c["ppo_clip"], 1+c["ppo_clip"])*adv[ix]).mean()-c["entropy"]*entropy
                self.aopt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actors.parameters(), c["max_grad_norm"])
                self.aopt.step()
                stats["entropy"].append(entropy.item())
                stats["clip_fraction"].append(((ratio-1).abs()>c["ppo_clip"]).float().mean().item())
        records = [r for _, rec in rows for r in rec]
        self.buffer.clear()
        return dict({k:float(np.mean(v)) if v else 0. for k, v in stats.items()},
                    kl_stopped_stages=float(stopped), return_mean=self.value_norm.mean,
                    return_std=self.value_norm.std, **action_diagnostics(records))

    def state(self, for_evaluation=False):
        s = dict(actors=self.actors.state_dict(), critic=self.critic.state_dict(), value_norm=self.value_norm.state())
        if not for_evaluation:
            s.update(aopt=self.aopt.state_dict(), copt=self.copt.state_dict())
        return s

    def load(self, state):
        self.actors.load_state_dict(state["actors"])
        self.critic.load_state_dict(state["critic"])
        self.value_norm.load(state["value_norm"])
        if "aopt" in state:
            self.aopt.load_state_dict(state["aopt"])
            self.copt.load_state_dict(state["copt"])

    def reset_history(self):
        pass


def learner_for(method, cfg, env, device="cpu"):
    return CentralPPO(cfg, env, device) if method == "PPO" else DecentralizedPPO(method, cfg, env, device)

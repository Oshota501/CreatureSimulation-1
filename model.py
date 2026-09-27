"""Actor-Critic ネットワーク（尻尾・左ヒレ・右ヒレの3つの離散行動を同時に出力）"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fish_env import ACTION_DIMS, FishConfig


def _layer(i, o, gain=math.sqrt(2)):
    layer = nn.Linear(i, o)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(i, h, o, out_gain):
    return nn.Sequential(_layer(i, h), nn.Tanh(), _layer(h, h), nn.Tanh(), _layer(h, o, out_gain))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=256, action_dims=ACTION_DIMS):
        super().__init__()
        self.action_dims = tuple(action_dims)
        self.actor = _mlp(obs_dim, hidden, sum(self.action_dims), 0.01)
        self.critic = _mlp(obs_dim, hidden, 1, 1.0)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def _log_probs(self, obs):
        logits = self.actor(obs)
        return [F.log_softmax(l, -1) for l in logits.split(self.action_dims, -1)]

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        actions, logp = [], 0
        for lp in self._log_probs(obs):
            if deterministic:
                a = lp.argmax(-1)
            else:  # Gumbel-max でサンプリング（どのデバイスでも速い）
                g = -torch.log(-torch.log(torch.rand_like(lp).clamp_(1e-9, 1 - 1e-9)))
                a = (lp + g).argmax(-1)
            actions.append(a)
            logp = logp + lp.gather(-1, a[:, None]).squeeze(-1)
        return torch.stack(actions, -1), logp, self.value(obs)

    def evaluate(self, obs, actions):
        logp, entropy = 0, 0
        for i, lp in enumerate(self._log_probs(obs)):
            logp = logp + lp.gather(-1, actions[:, i : i + 1]).squeeze(-1)
            entropy = entropy - (lp.exp() * lp).sum(-1)
        return logp, entropy, self.value(obs)


def save_checkpoint(path, model, env_cfg: FishConfig, hidden, global_step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = dict(
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        obs_dim=model.actor[0].in_features,
        hidden=hidden,
        env_cfg=vars(env_cfg).copy(),
        global_step=global_step,
    )
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)  # 学習中に play.py が読んでも壊れたファイルを掴まないように


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    cfg = FishConfig(**data["env_cfg"])
    model = ActorCritic(data["obs_dim"], data["hidden"]).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, cfg, data

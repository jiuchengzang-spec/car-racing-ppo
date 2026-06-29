"""A minimal PPO actor-critic in pure PyTorch — no stable-baselines3.

Just the pieces train.py and enjoy.py both need: a small two-headed MLP over the
env's low-dimensional observation (rangefinders + speeds + heading + steer) and
helpers to save/load it. The training loop itself (rollouts, GAE, the clipped
update, curriculum) lives in train.py; keeping the *network* here lets enjoy.py
rebuild and load a checkpoint without dragging in the trainer.

Dependency-light by design: numpy + torch only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer(in_dim: int, out_dim: int, std: float = float(np.sqrt(2.0)), bias: float = 0.0) -> nn.Linear:
    """A Linear with orthogonal init — the PPO default that keeps early logits sane."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Separate policy/value MLPs (tanh, 2 hidden layers) with a state-independent
    log-std for the continuous action — the standard PPO-for-continuous setup.

    The policy head outputs the Gaussian *mean* per action dim; ``log_std`` is a
    free parameter (not a function of the state), which trains more stably for
    locomotion/driving than a state-dependent std. Actions are unbounded here and
    clipped to the env's [-1, 1] box by the caller when stepping.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.actor_mean = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, act_dim, std=0.01),  # small last layer -> calm initial policy
        )
        # Start a little exploratory (std ~0.6) rather than 1.0, so early rollouts
        # don't thrash the controls before the value function means anything.
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or score a given) action; return (action, log_prob, entropy, value)."""
        mean = self.actor_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Numpy-in / numpy-out single- or batch-action helper for playback."""
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32))
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
        if deterministic:
            action = self.actor_mean(obs)
        else:
            action, _, _, _ = self.get_action_and_value(obs)
        action = action.clamp(-1.0, 1.0).cpu().numpy()
        return action[0] if single else action

    @torch.no_grad()
    def activations(self, obs_np: np.ndarray) -> dict[str, Any]:
        """Run a forward pass and return every intermediate activation (for viz).

        Captures the post-Tanh outputs of each hidden layer in both heads, plus the
        action mean / std and the value estimate, all as numpy. Single obs only.
        """
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0)

        def run(seq: nn.Sequential) -> tuple[list[np.ndarray], np.ndarray]:
            h = obs
            hidden = []
            for layer in seq:
                h = layer(h)
                if isinstance(layer, nn.Tanh):
                    hidden.append(h.squeeze(0).cpu().numpy())
            return hidden, h.squeeze(0).cpu().numpy()  # hidden tanh outs, final linear out

        actor_hidden, mean = run(self.actor_mean)
        critic_hidden, value = run(self.critic)
        return {
            "obs": np.asarray(obs_np, dtype=np.float32),
            "actor_hidden": actor_hidden,
            "critic_hidden": critic_hidden,
            "mean": mean,
            "std": torch.exp(self.log_std).cpu().numpy(),
            "value": float(value[0]),
        }


def save_policy(path: str, model: ActorCritic, meta: dict[str, Any] | None = None) -> None:
    """Persist the policy as a small ``.pt`` (state dict + the dims to rebuild it)."""
    payload = {
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "act_dim": model.act_dim,
        "hidden": model.hidden,
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_policy(path: str, device: str = "cpu") -> tuple[ActorCritic, dict[str, Any]]:
    """Rebuild an :class:`ActorCritic` from a checkpoint saved by :func:`save_policy`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})

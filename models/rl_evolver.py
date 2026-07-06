"""RL policy for confusion-matrix-driven cache evolution."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


RL_ACTIONS = ("skip", "merge", "split", "evict", "refresh_old", "boost_confused")


def build_confusion_matrix(
    labels: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Row-normalized confusion matrix [num_classes, num_classes]."""
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    preds_np = preds.cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for true_l, pred_l in zip(labels_np, preds_np):
        if 0 <= int(true_l) < num_classes and 0 <= int(pred_l) < num_classes:
            cm[int(true_l), int(pred_l)] += 1.0
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return cm / row_sum


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    return np.diag(cm)


class EvolutionPolicy(nn.Module):
    """Small MLP: confusion-matrix state -> evolution action logits."""

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)


class RLEvolutionAgent:
    """Online REINFORCE agent — one policy-gradient step per task."""

    def __init__(
        self,
        max_classes: int = 40,
        num_actions: int = len(RL_ACTIONS),
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
        reward_old_weight: float = 3.0,
        reward_new_weight: float = 1.0,
        reward_cache_penalty: float = 0.05,
    ):
        self.max_classes = max_classes
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.reward_old_weight = reward_old_weight
        self.reward_new_weight = reward_new_weight
        self.reward_cache_penalty = reward_cache_penalty
        self.state_dim = max_classes * max_classes + 5
        self.policy = EvolutionPolicy(self.state_dim, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._last_log_prob: Optional[torch.Tensor] = None
        self._last_entropy: Optional[torch.Tensor] = None
        self.total_reward = 0.0
        self.action_history: list[str] = []

    def build_state(
        self,
        cm: np.ndarray,
        num_classes: int,
        cache_size: int,
        task_id: int,
        known_classes: int,
        cache_budget: int,
    ) -> torch.Tensor:
        padded = np.zeros((self.max_classes, self.max_classes), dtype=np.float32)
        n = min(num_classes, self.max_classes)
        padded[:n, :n] = cm[:n, :n].astype(np.float32)

        if known_classes > 0 and n > 0:
            old_recall = float(np.diag(cm[:known_classes, :known_classes]).mean())
        else:
            old_recall = 1.0

        extras = np.array(
            [
                cache_size / max(cache_budget, 1),
                task_id / 10.0,
                known_classes / max(self.max_classes, 1),
                old_recall,
                num_classes / max(self.max_classes, 1),
            ],
            dtype=np.float32,
        )
        state = np.concatenate([padded.flatten(), extras])
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def select_action(self, state: torch.Tensor, greedy: bool = False) -> int:
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        self._last_log_prob = dist.log_prob(action)
        self._last_entropy = dist.entropy()
        return int(action.item())

    def compute_reward(
        self,
        before: dict,
        after: dict,
        cache_delta: int,
    ) -> float:
        old_gain = after.get("old_acc", 0.0) - before.get("old_acc", 0.0)
        new_gain = after.get("new_acc", 0.0) - before.get("new_acc", 0.0)
        total_gain = after.get("total_acc", 0.0) - before.get("total_acc", 0.0)
        reward = (
            self.reward_old_weight * old_gain
            + self.reward_new_weight * new_gain
            + 0.5 * total_gain
            - self.reward_cache_penalty * max(cache_delta, 0)
        )
        self.total_reward += reward
        return reward

    def learn(self, reward: float, entropy_coef: float = 0.01) -> float:
        if self._last_log_prob is None:
            return 0.0
        loss = -(self._last_log_prob * reward)
        if self._last_entropy is not None:
            loss = loss - entropy_coef * self._last_entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._last_log_prob = None
        self._last_entropy = None
        return float(loss.item())

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_reward": self.total_reward,
            "action_history": list(self.action_history),
        }

    def load_state_dict(self, state: dict) -> None:
        if "policy" in state:
            self.policy.load_state_dict(state["policy"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.total_reward = float(state.get("total_reward", 0.0))
        self.action_history = list(state.get("action_history", []))

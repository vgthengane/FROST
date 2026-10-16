"""CorrMem: correlation memory (closed-form ridge over random features).

Maintains additive sufficient statistics ``(G, Q)`` in a frozen random-feature
lift of width ``V`` (``ridge_rp_dim`` / ``corrmem.lift_v``), then solves
``W = (G + λI)^{-1} Q`` once per session. Past sessions are summarised by
``(G, Q)`` — no raw exemplar store is required for CorrMem itself.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from utils.toolkit import target2onehot


class CorrMem:
    """Analytical CorrMem head (ridge in random feature space).

    ``G`` is class-agnostic second-moment accumulation; ``Q`` is class-aware
    feature–label accumulation (one column per class observed so far).
    """

    def __init__(
        self,
        feature_dim: int,
        rp_dim: int,
        nb_classes: int,
        device: torch.device | str,
        ridge: float | None = None,
        seed: int = 0,
        ridge_val_fraction: float = 0.2,
        ridge_lambda_min: float = 1e-8,
        ridge_lambda_max: float | None = None,
    ):
        self.feature_dim = feature_dim
        self.rp_dim = rp_dim
        self.nb_classes = nb_classes
        self.device = torch.device(device)
        self.ridge = ridge
        self.selection_seed = int(seed)
        self.ridge_val_fraction = float(ridge_val_fraction)
        self.ridge_lambda_min = float(ridge_lambda_min)
        self.ridge_lambda_max = ridge_lambda_max

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self.W_rand = torch.randn(feature_dim, rp_dim, generator=gen)
        self.Q = torch.zeros(rp_dim, nb_classes)
        self.G = torch.zeros(rp_dim, rp_dim)
        self._weights: torch.Tensor | None = None
        self.replay_updates = 0

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        h = features @ self.W_rand.to(features.device)
        return F.relu(h)

    def _ridge_grid(self, n_samples: int) -> np.ndarray:
        """Build a lambda grid capped for low-shot stability."""
        min_log = int(np.floor(np.log10(max(self.ridge_lambda_min, 1e-12))))
        if self.ridge_lambda_max is not None:
            max_log = int(np.floor(np.log10(max(self.ridge_lambda_max, 1.0))))
        else:
            # Fewer points -> lower max lambda to avoid over-regularized weights.
            auto_max = 2.0 + 1.5 * np.log10(max(n_samples, 10))
            max_log = int(np.floor(min(8.0, auto_max)))
        if max_log < min_log:
            max_log = min_log
        return 10.0 ** np.arange(min_log, max_log + 1)

    def _select_ridge(self, features_h: torch.Tensor, labels_oh: torch.Tensor) -> float:
        """Pick ridge lambda with a shuffled hold-out grid search."""
        n = features_h.shape[0]
        if n < 4:
            return 1.0

        perm = np.random.default_rng(self.selection_seed).permutation(n)
        f = features_h[perm]
        y = labels_oh[perm]

        n_val = max(1, min(n - 2, int(round(n * self.ridge_val_fraction))))
        f_val, f_train = f[:n_val], f[n_val:]
        y_val, y_train = y[:n_val], y[n_val:]
        if f_train.shape[0] == 0:
            f_train, y_train = f_val, y_val
            f_val, y_val = f[:1], y[:1]

        ridges = self._ridge_grid(n)
        g_train = f_train.T @ f_train
        q_train = f_train.T @ y_train
        best_ridge, best_loss = float(ridges[0]), float("inf")
        eye = torch.eye(g_train.shape[0])
        for lam in ridges:
            w = torch.linalg.solve(g_train + float(lam) * eye, q_train).T
            pred = f_val @ w.T
            loss = F.mse_loss(pred, y_val).item()
            if loss < best_loss:
                best_loss = loss
                best_ridge = float(lam)
        logging.info(
            "CorrMem lambda grid n=%d max=%.2e selected=%.2e",
            n,
            float(ridges[-1]),
            best_ridge,
        )
        return best_ridge

    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weight: float = 1.0,
        fit_ridge: bool = False,
    ) -> None:
        """Accumulate statistics from a batch of embeddings."""
        if features.numel() == 0:
            return
        features = features.detach().cpu().float()
        labels = labels.detach().cpu().long()
        h = self._project(features)
        y = target2onehot(labels, self.nb_classes).float()

        if weight != 1.0:
            h = h * float(np.sqrt(weight))
            y = y * float(weight)

        self.Q += h.T @ y
        self.G += h.T @ h
        self._weights = None

        if fit_ridge and self.ridge is None:
            self.ridge = self._select_ridge(h, y)
            logging.info("CorrMem selected lambda=%.2e", self.ridge)

    def replay_classes(
        self,
        class_embeddings: dict[int, torch.Tensor],
        confusion_matrix: np.ndarray,
        recall_threshold: float,
        replay_weight: float = 1.0,
        max_classes: int = 5,
        class_range: tuple[int, int] | None = None,
    ) -> int:
        """Re-accumulate ridge stats for low-recall classes in ``confusion_matrix``."""
        if replay_weight <= 0:
            return 0
        lo, hi = class_range if class_range is not None else (0, confusion_matrix.shape[0])
        candidates = []
        for cls in class_embeddings:
            if cls < lo or cls >= hi or cls >= confusion_matrix.shape[0]:
                continue
            row_sum = float(confusion_matrix[cls].sum())
            if row_sum <= 0:
                continue
            recall = float(confusion_matrix[cls, cls] / row_sum)
            if recall < recall_threshold:
                candidates.append((recall, cls))
        candidates.sort(key=lambda x: x[0])

        n_replayed = 0
        for _, cls in candidates[:max_classes]:
            emb = class_embeddings[cls]
            if emb is None or emb.numel() == 0:
                continue
            labels = torch.full((emb.shape[0],), cls, dtype=torch.long)
            self.update(emb, labels, weight=replay_weight)
            n_replayed += 1
        self.replay_updates += n_replayed
        return n_replayed

    def _solve_weights(self, num_classes: int) -> torch.Tensor:
        lam = self.ridge if self.ridge is not None else 1.0
        g = self.G + lam * torch.eye(self.G.shape[0])
        w = torch.linalg.solve(g, self.Q).T
        return w[:num_classes]

    def classify(self, features: torch.Tensor, num_classes: int) -> torch.Tensor:
        if self.Q.sum() == 0:
            return torch.zeros(
                features.shape[0], num_classes, device=features.device
            )
        if self._weights is None or self._weights.shape[0] < num_classes:
            self._weights = self._solve_weights(num_classes).to(features.device)
        h = self._project(features.to(features.device))
        return h @ self._weights.T

    def state_dict(self) -> dict:
        return {
            "W_rand": self.W_rand.cpu(),
            "Q": self.Q.cpu(),
            "G": self.G.cpu(),
            "ridge": self.ridge,
            "replay_updates": self.replay_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.W_rand = state["W_rand"]
        self.Q = state["Q"]
        self.G = state["G"]
        self.ridge = state.get("ridge")
        self.replay_updates = int(state.get("replay_updates", 0))
        self._weights = None

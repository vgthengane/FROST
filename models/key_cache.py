"""Self-evolving key-value memory cache over frozen embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EvolutionStats:
    """Per-task evolution log for analysis and ablations."""

    task: int = 0
    cache_size: int = 0
    inserts: int = 0
    merges: int = 0
    splits: int = 0
    evictions: int = 0
    refreshes: int = 0
    action: str = "none"
    val_acc: float = 0.0
    old_acc: float = 0.0
    confusion_peak: float = 0.0
    per_class_min: int = 0
    per_class_max: int = 0
    per_class_mean: float = 0.0
    n_classes_cached: int = 0


@dataclass
class BanditState:
    """UCB1 statistics for evolution operators."""

    counts: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in _ACTIONS})
    rewards: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in _ACTIONS})


_ACTIONS = ("skip", "merge", "split", "evict", "refresh_old", "boost_confused")


class KeyCache:
    """Key-value cache with optional rule-based or bandit-driven evolution."""

    def __init__(
        self,
        feature_dim: int,
        device: torch.device | str,
        max_prototypes_per_class: int = 3,
        merge_threshold: float = 0.95,
        split_confusion_threshold: float = 0.15,
        cache_budget: int = 120,
        evolution_policy: str = "rules",
        bandit_ucb_c: float = 1.0,
        rl_agent=None,
        recall_refresh_threshold: float = 0.85,
        prototype_pooling: str = "max",
        pooling_temperature: float = 0.1,
        insert_strategy: str = "mean",
        split_strategy: str = "confusion_pair",
        enable_gated_growth: bool = False,
        gated_growth_recall_threshold: float = 0.9,
        gated_growth_max_classes: int = 2,
        gated_growth_include_novel: bool = False,
        novel_gated_growth_threshold: float = 0.85,
        novel_gated_growth_max_classes: int = 5,
        # Diagonal-Gaussian cache scoring (optional; see classify + insert stats).
        gauss_var_floor: float = 1e-4,
        gauss_var_shrink: float = 0.5,
        gauss_use_logdet: bool = False,
    ):
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.max_prototypes_per_class = max_prototypes_per_class
        self.merge_threshold = merge_threshold
        self.split_confusion_threshold = split_confusion_threshold
        self.cache_budget = cache_budget
        self.evolution_policy = evolution_policy
        self.bandit_ucb_c = bandit_ucb_c
        self.rl_agent = rl_agent
        self.recall_refresh_threshold = recall_refresh_threshold
        self.prototype_pooling = prototype_pooling
        self.pooling_temperature = pooling_temperature
        self.insert_strategy = insert_strategy
        self.split_strategy = split_strategy
        self.enable_gated_growth = enable_gated_growth
        self.gated_growth_recall_threshold = gated_growth_recall_threshold
        self.gated_growth_max_classes = gated_growth_max_classes
        self.gated_growth_include_novel = gated_growth_include_novel
        self.novel_gated_growth_threshold = novel_gated_growth_threshold
        self.novel_gated_growth_max_classes = novel_gated_growth_max_classes
        self.gauss_var_floor = float(gauss_var_floor)
        self.gauss_var_shrink = float(gauss_var_shrink)
        self.gauss_use_logdet = bool(gauss_use_logdet)

        self.keys: torch.Tensor = torch.empty(0, feature_dim, device=self.device)
        self.labels: torch.Tensor = torch.empty(0, dtype=torch.long, device=self.device)
        self.margins: torch.Tensor = torch.empty(0, device=self.device)
        # Per-class diagonal Gaussian stats from *all* shot embeddings at insert
        # (not only stored prototypes). Used when prototype_pooling == "diag_gauss".
        self._gauss_count: Dict[int, int] = {}
        self._gauss_mean: Dict[int, torch.Tensor] = {}
        self._gauss_var: Dict[int, torch.Tensor] = {}
        self.bandit = BanditState()
        self.history: List[EvolutionStats] = []

    @property
    def size(self) -> int:
        return int(self.keys.shape[0])

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def classify(self, features: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Return per-class cache logits [B, num_classes].

        Default: cosine similarity pooled over stored keys (max / mean / logsumexp).
        ``diag_gauss``: shrunk diagonal Gaussian log-score from shot stats recorded
        at insert (μ, σ² per dim). Designed for few-shot novel classes where raw
        max-over-keys is noisy.
        """
        if self.prototype_pooling == "diag_gauss":
            return self._classify_diag_gauss(features, num_classes)

        if self.size == 0:
            return torch.zeros(features.shape[0], num_classes, device=features.device)

        feat = self._normalize(features)
        keys = self._normalize(self.keys)
        sim = feat @ keys.T

        logits = torch.full(
            (features.shape[0], num_classes),
            -1e4,
            device=features.device,
            dtype=features.dtype,
        )
        for cls in range(num_classes):
            mask = self.labels == cls
            if not mask.any():
                continue
            cls_sim = sim[:, mask]
            if self.prototype_pooling == "logsumexp":
                tau = max(self.pooling_temperature, 1e-4)
                logits[:, cls] = tau * torch.logsumexp(cls_sim / tau, dim=1)
            elif self.prototype_pooling == "mean":
                logits[:, cls] = cls_sim.mean(dim=1)
            else:
                logits[:, cls] = cls_sim.max(dim=1).values
        return logits

    def _gauss_prior_var(self) -> torch.Tensor:
        """Mean diagonal variance across classes (shrinkage target)."""
        floor = self.gauss_var_floor
        if not self._gauss_var:
            return torch.full(
                (self.feature_dim,), floor, device=self.device, dtype=torch.float32
            )
        stacked = torch.stack(list(self._gauss_var.values()), dim=0)
        return stacked.mean(dim=0).clamp_min(floor)

    def _classify_diag_gauss(
        self, features: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """Shrunk diag-Gaussian scores; falls back to cosine-max if no stats."""
        logits = torch.full(
            (features.shape[0], num_classes),
            -1e4,
            device=features.device,
            dtype=features.dtype,
        )
        if not self._gauss_mean:
            # No distribution stats yet — degrade to cosine max over keys.
            if self.size == 0:
                return torch.zeros(
                    features.shape[0], num_classes, device=features.device
                )
            feat = self._normalize(features)
            keys = self._normalize(self.keys)
            sim = feat @ keys.T
            for cls in range(num_classes):
                mask = self.labels == cls
                if mask.any():
                    logits[:, cls] = sim[:, mask].max(dim=1).values
            return logits

        feat = features.to(self.device)
        prior = self._gauss_prior_var().to(device=feat.device, dtype=feat.dtype)
        shrink = min(max(self.gauss_var_shrink, 0.0), 1.0)
        floor = self.gauss_var_floor
        tau = max(self.pooling_temperature, 1e-4)

        for cls in range(num_classes):
            if cls not in self._gauss_mean:
                continue
            mu = self._gauss_mean[cls].to(device=feat.device, dtype=feat.dtype)
            var = self._gauss_var[cls].to(device=feat.device, dtype=feat.dtype)
            var = (1.0 - shrink) * var + shrink * prior
            var = var.clamp_min(floor)
            # Mean over dims keeps logit scale closer to cosine fusion.
            maha = ((feat - mu) ** 2 / var).mean(dim=-1)
            score = -maha / tau
            if self.gauss_use_logdet:
                score = score - 0.5 * torch.log(var).mean()
            logits[:, cls] = score
        return logits

    def _update_class_gauss_stats(self, cls: int, cls_emb: torch.Tensor) -> None:
        """Fit diagonal Gaussian from all available class embeddings (e.g. 5-shot)."""
        emb = cls_emb.detach().to(self.device)
        if emb.ndim != 2 or emb.shape[0] == 0:
            return
        n = int(emb.shape[0])
        mean = emb.mean(dim=0)
        if n >= 2:
            var = emb.var(dim=0, unbiased=False)
        else:
            var = torch.full(
                (self.feature_dim,),
                self.gauss_var_floor,
                device=self.device,
                dtype=emb.dtype,
            )
        var = var.clamp_min(self.gauss_var_floor)
        self._gauss_count[cls] = n
        self._gauss_mean[cls] = mean
        self._gauss_var[cls] = var

    def _clear_class_gauss_stats(self, class_ids: set[int]) -> None:
        for cls in class_ids:
            self._gauss_count.pop(cls, None)
            self._gauss_mean.pop(cls, None)
            self._gauss_var.pop(cls, None)

    def insert_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_ids: np.ndarray,
    ) -> int:
        """Insert one mean prototype per class (skip if at per-class cap)."""
        inserted = 0
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        for cls in class_ids:
            cls = int(cls)
            n_existing = int((self.labels == cls).sum().item()) if self.size else 0
            if n_existing >= self.max_prototypes_per_class:
                continue
            mask = labels == cls
            if not mask.any():
                continue
            cls_emb = embeddings[mask]
            # Always refresh class distribution from the full shot set (5-shot etc.).
            self._update_class_gauss_stats(cls, cls_emb)
            n_new = self.max_prototypes_per_class - n_existing
            if self.insert_strategy == "per_sample" and cls_emb.shape[0] > 0:
                protos = cls_emb[:n_new]
                self._append_keys(
                    protos,
                    torch.full((protos.shape[0],), cls, dtype=torch.long, device=self.device),
                )
                inserted += int(protos.shape[0])
            elif self.insert_strategy == "cluster" and cls_emb.shape[0] > 1 and n_new > 1:
                protos = self._farthest_prototypes(cls_emb, n_new)
                self._append_keys(
                    protos, torch.full((protos.shape[0],), cls, dtype=torch.long, device=self.device)
                )
                inserted += int(protos.shape[0])
            elif self.insert_strategy == "mean_best2" and cls_emb.shape[0] > 0:
                protos = self._mean_best2_prototypes(cls_emb, n_new)
                self._append_keys(
                    protos,
                    torch.full((protos.shape[0],), cls, dtype=torch.long, device=self.device),
                )
                inserted += int(protos.shape[0])
            elif self.insert_strategy == "mean_close_far" and cls_emb.shape[0] > 0:
                protos = self._mean_close_far_prototypes(cls_emb, n_new)
                self._append_keys(
                    protos,
                    torch.full((protos.shape[0],), cls, dtype=torch.long, device=self.device),
                )
                inserted += int(protos.shape[0])
            elif self.insert_strategy == "mean_far2" and cls_emb.shape[0] > 0:
                protos = self._mean_far2_prototypes(cls_emb, n_new)
                self._append_keys(
                    protos,
                    torch.full((protos.shape[0],), cls, dtype=torch.long, device=self.device),
                )
                inserted += int(protos.shape[0])
            elif self.insert_strategy == "mean_nearmean" and cls_emb.shape[0] > 0:
                protos = self._mean_nearmean_prototypes(cls_emb, n_new)
                self._append_keys(
                    protos,
                    torch.full((protos.shape[0],), cls, dtype=torch.long, device=self.device),
                )
                inserted += int(protos.shape[0])
            else:
                proto = cls_emb.mean(dim=0)
                self._append_keys(proto.unsqueeze(0), torch.tensor([cls], device=self.device))
                inserted += 1
        return inserted

    def replace_class_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_ids: np.ndarray,
    ) -> int:
        """Drop and re-insert prototypes for the given classes."""
        if self.size == 0:
            return self.insert_prototypes(embeddings, labels, class_ids)

        class_ids = {int(c) for c in class_ids}
        self._clear_class_gauss_stats(class_ids)
        keep = torch.tensor(
            [int(lbl.item()) not in class_ids for lbl in self.labels],
            dtype=torch.bool,
            device=self.device,
        )
        if keep.any():
            self.keys = self.keys[keep]
            self.labels = self.labels[keep]
            self.margins = self.margins[keep]
        else:
            self.keys = torch.empty(0, self.feature_dim, device=self.device)
            self.labels = torch.empty(0, dtype=torch.long, device=self.device)
            self.margins = torch.empty(0, device=self.device)
        return self.insert_prototypes(embeddings, labels, np.array(sorted(class_ids)))

    def _mean_best2_prototypes(
        self, embeddings: torch.Tensor, n_new: int
    ) -> torch.Tensor:
        """Class mean plus up to two shots most similar to the mean (representative)."""
        emb = embeddings.to(self.device)
        n_slots = max(1, min(int(n_new), 3))
        mean_proto = emb.mean(dim=0, keepdim=True)
        if n_slots == 1 or emb.shape[0] == 1:
            return mean_proto
        sim = F.cosine_similarity(
            self._normalize(emb), self._normalize(mean_proto), dim=1
        )
        k_best = min(n_slots - 1, 2, emb.shape[0])
        best_idx = torch.topk(sim, k=k_best, largest=True).indices
        return torch.cat([mean_proto, emb[best_idx]], dim=0)

    def _mean_close_far_prototypes(
        self, embeddings: torch.Tensor, n_new: int
    ) -> torch.Tensor:
        """Class mean plus one shot closest and one farthest from the mean."""
        emb = embeddings.to(self.device)
        n_slots = max(1, min(int(n_new), 3))
        mean_proto = emb.mean(dim=0, keepdim=True)
        if n_slots == 1 or emb.shape[0] == 1:
            return mean_proto
        sim = F.cosine_similarity(
            self._normalize(emb), self._normalize(mean_proto), dim=1
        )
        close_idx = int(torch.argmax(sim).item())
        protos = [mean_proto, emb[close_idx].unsqueeze(0)]
        if n_slots < 3:
            return torch.cat(protos, dim=0)
        remaining = [i for i in range(emb.shape[0]) if i != close_idx]
        if not remaining:
            return torch.cat(protos, dim=0)
        rem_t = torch.tensor(remaining, device=self.device)
        far_local = int(torch.argmin(sim[rem_t]).item())
        far_idx = remaining[far_local]
        protos.append(emb[far_idx].unsqueeze(0))
        return torch.cat(protos, dim=0)

    def _mean_far2_prototypes(
        self, embeddings: torch.Tensor, n_new: int
    ) -> torch.Tensor:
        """Class mean plus up to two shots farthest from the mean."""
        emb = embeddings.to(self.device)
        n_slots = max(1, min(int(n_new), 3))
        mean_proto = emb.mean(dim=0, keepdim=True)
        if n_slots == 1 or emb.shape[0] == 1:
            return mean_proto
        sim = F.cosine_similarity(
            self._normalize(emb), self._normalize(mean_proto), dim=1
        )
        n_far = min(n_slots - 1, 2, emb.shape[0])
        far_idx = torch.topk(sim, k=n_far, largest=False).indices
        return torch.cat([mean_proto, emb[far_idx]], dim=0)

    def _mean_nearmean_prototypes(
        self, embeddings: torch.Tensor, n_new: int
    ) -> torch.Tensor:
        """Class mean plus one shot closest to the mean (no far / evo slot)."""
        emb = embeddings.to(self.device)
        n_slots = max(1, min(int(n_new), 2))
        mean_proto = emb.mean(dim=0, keepdim=True)
        if n_slots == 1 or emb.shape[0] == 1:
            return mean_proto
        sim = F.cosine_similarity(
            self._normalize(emb), self._normalize(mean_proto), dim=1
        )
        close_idx = int(torch.argmax(sim).item())
        return torch.cat([mean_proto, emb[close_idx].unsqueeze(0)], dim=0)

    def _farthest_prototypes(self, embeddings: torch.Tensor, k: int) -> torch.Tensor:
        """Pick diverse class prototypes using farthest-point sampling in embedding space."""
        emb = embeddings.to(self.device)
        n = emb.shape[0]
        k = max(1, min(k, n))
        if k == 1:
            return emb.mean(dim=0, keepdim=True)

        centers = [int(torch.randint(0, n, (1,), device=self.device).item())]
        dist = torch.cdist(emb, emb[centers], p=2).squeeze(1)
        for _ in range(1, k):
            idx = int(torch.argmax(dist).item())
            centers.append(idx)
            dist = torch.minimum(dist, torch.cdist(emb, emb[[idx]], p=2).squeeze(1))
        return emb[torch.tensor(centers, device=self.device)]

    def _append_keys(self, keys: torch.Tensor, labels: torch.Tensor) -> None:
        if self.size == 0:
            self.keys = keys.to(self.device)
            self.labels = labels.to(self.device)
            self.margins = torch.zeros(keys.shape[0], device=self.device)
        else:
            self.keys = torch.cat([self.keys, keys.to(self.device)], dim=0)
            self.labels = torch.cat([self.labels, labels.to(self.device)], dim=0)
            self.margins = torch.cat(
                [self.margins, torch.zeros(keys.shape[0], device=self.device)], dim=0
            )

    def sync_fc_weights(self, fc) -> None:
        """Write primary (first) prototype per class into cosine classifier weights."""
        if self.size == 0 or fc is None:
            return
        with torch.no_grad():
            for cls in self.labels.unique():
                cls = int(cls.item())
                mask = self.labels == cls
                if mask.any() and cls < fc.out_features:
                    fc.weight.data[cls] = self.keys[mask][0]

    def _per_class_counts(self) -> dict[int, int]:
        """Map class id -> number of cache keys (prototypes) for that class."""
        if self.size == 0:
            return {}
        max_label = int(self.labels[: self.size].max().item())
        counts = torch.bincount(self.labels[: self.size], minlength=max_label + 1)
        return {
            cls: int(counts[cls].item())
            for cls in range(counts.shape[0])
            if int(counts[cls].item()) > 0
        }

    def _per_class_count_stats(self) -> tuple[int, int, float, int]:
        """Return (min, max, mean, n_classes) prototype counts over cached labels."""
        counts = self._per_class_counts()
        if not counts:
            return 0, 0, 0.0, 0
        vals = list(counts.values())
        return min(vals), max(vals), sum(vals) / len(vals), len(counts)

    def _apply_per_class_stats(self, stats: EvolutionStats) -> dict[int, int]:
        """Fill summary fields on ``stats`` and return per-class key counts."""
        stats.cache_size = self.size
        counts = self._per_class_counts()
        if not counts:
            stats.per_class_min = 0
            stats.per_class_max = 0
            stats.per_class_mean = 0.0
            stats.n_classes_cached = 0
            return counts
        vals = list(counts.values())
        stats.per_class_min = min(vals)
        stats.per_class_max = max(vals)
        stats.per_class_mean = sum(vals) / len(vals)
        stats.n_classes_cached = len(counts)
        return counts

    def _log_per_class_key_counts(self, task_id: int, stats: EvolutionStats) -> None:
        """Log per-class key counts: how many prototypes each class has in cache."""
        counts = self._per_class_counts()
        if not counts:
            logging.info(
                "KeyCache per-class keys task=%d n_classes=0 min=0 max=0 mean=0.00 total=0",
                task_id,
            )
            return

        hist: dict[int, int] = {}
        for n_keys in counts.values():
            hist[n_keys] = hist.get(n_keys, 0) + 1
        hist_str = ",".join(f"{k}x{v}" for k, v in sorted(hist.items()))
        cls_str = ",".join(f"{cls}:{n}" for cls, n in sorted(counts.items()))

        logging.info(
            "KeyCache per-class keys task=%d n_classes=%d min=%d max=%d mean=%.2f "
            "total=%d histogram=%s",
            task_id,
            stats.n_classes_cached,
            stats.per_class_min,
            stats.per_class_max,
            stats.per_class_mean,
            stats.cache_size,
            hist_str,
        )
        logging.info("KeyCache per-class keys task=%d cls_keys=%s", task_id, cls_str)

    def evolve(
        self,
        task_id: int,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        num_classes: int,
        known_classes: int = 0,
        confusion_matrix: np.ndarray | None = None,
        old_class_embeddings: dict | None = None,
    ) -> EvolutionStats:
        """Run evolution policy after a task."""
        stats = EvolutionStats(task=task_id, cache_size=self.size)
        if self.evolution_policy == "static":
            stats.action = "static"
            self._apply_per_class_stats(stats)
            self.history.append(stats)
            self._log_per_class_key_counts(task_id, stats)
            return stats

        if confusion_matrix is None:
            confusion_matrix = self._build_cm(labels, preds, num_classes)

        off_diag = confusion_matrix.copy()
        np.fill_diagonal(off_diag, 0.0)
        stats.confusion_peak = float(off_diag.max()) if off_diag.size else 0.0

        before_metrics = self._split_metrics(embeddings, labels, num_classes, known_classes)
        stats.val_acc = before_metrics["total_acc"]
        stats.old_acc = before_metrics["old_acc"]
        cache_before = self.size

        if self.evolution_policy == "rl" and self.rl_agent is not None:
            stats = self._evolve_rl(
                stats,
                task_id,
                embeddings,
                labels,
                preds,
                num_classes,
                known_classes,
                confusion_matrix,
                old_class_embeddings or {},
                before_metrics,
                cache_before,
            )
        elif self.evolution_policy == "aggressive":
            stats.refreshes = self._refresh_old_prototypes(
                old_class_embeddings or {}, confusion_matrix, known_classes
            )
            gated_splits = self._gated_grow_confused(
                embeddings, labels, preds, confusion_matrix, 0, known_classes
            )
            if self.gated_growth_include_novel and known_classes < num_classes:
                gated_splits += self._gated_grow_confused(
                    embeddings,
                    labels,
                    preds,
                    confusion_matrix,
                    known_classes,
                    num_classes,
                    recall_threshold=self.novel_gated_growth_threshold,
                    max_classes=self.novel_gated_growth_max_classes,
                )
            split_splits = self._split_confused(
                embeddings, labels, preds, top_k=5
            )
            stats.splits = gated_splits + split_splits
            stats.merges = self._merge_similar()
            stats.evictions = self._evict_if_needed()
            stats.action = "aggressive"
            after_metrics = self._split_metrics(embeddings, labels, num_classes, known_classes)
            stats.val_acc = after_metrics["total_acc"]
            stats.old_acc = after_metrics["old_acc"]
        elif self.evolution_policy == "bandit":
            action = self._select_bandit_action()
            reward_before = before_metrics["total_acc"]
            self._apply_action(
                action, embeddings, labels, preds, num_classes,
                confusion_matrix, old_class_embeddings or {},
            )
            after_metrics = self._split_metrics(embeddings, labels, num_classes, known_classes)
            self._update_bandit(action, after_metrics["total_acc"] - reward_before)
            stats.action = action
            stats.val_acc = after_metrics["total_acc"]
            stats.old_acc = after_metrics["old_acc"]
        else:
            gated_splits = self._gated_grow_confused(
                embeddings, labels, preds, confusion_matrix, 0, known_classes
            )
            if self.gated_growth_include_novel and known_classes < num_classes:
                gated_splits += self._gated_grow_confused(
                    embeddings,
                    labels,
                    preds,
                    confusion_matrix,
                    known_classes,
                    num_classes,
                    recall_threshold=self.novel_gated_growth_threshold,
                    max_classes=self.novel_gated_growth_max_classes,
                )
            stats.merges = self._merge_similar()
            split_splits = self._split_confused(embeddings, labels, preds)
            stats.splits = gated_splits + split_splits
            stats.evictions = self._evict_if_needed()
            stats.refreshes = self._refresh_old_prototypes(
                old_class_embeddings or {}, confusion_matrix, known_classes
            )
            stats.action = "rules"
            after_metrics = self._split_metrics(embeddings, labels, num_classes, known_classes)
            stats.val_acc = after_metrics["total_acc"]
            stats.old_acc = after_metrics["old_acc"]
        self.history.append(stats)
        self._apply_per_class_stats(stats)
        logging.info(
            "KeyCache evolve task=%d policy=%s size=%d merges=%d splits=%d "
            "evictions=%d refreshes=%d action=%s old_acc=%.2f total_acc=%.2f peak_conf=%.3f",
            task_id,
            self.evolution_policy,
            stats.cache_size,
            stats.merges,
            stats.splits,
            stats.evictions,
            stats.refreshes,
            stats.action,
            stats.old_acc,
            stats.val_acc,
            stats.confusion_peak,
        )
        self._log_per_class_key_counts(task_id, stats)
        return stats

    def _evolve_rl(
        self,
        stats: EvolutionStats,
        task_id: int,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        num_classes: int,
        known_classes: int,
        confusion_matrix: np.ndarray,
        old_class_embeddings: dict,
        before_metrics: dict,
        cache_before: int,
    ) -> EvolutionStats:
        agent = self.rl_agent
        state = agent.build_state(
            confusion_matrix,
            num_classes,
            self.size,
            task_id,
            known_classes,
            self.cache_budget,
        )
        action_idx = agent.select_action(state)
        action = _ACTIONS[action_idx]
        agent.action_history.append(action)

        self._apply_action(
            action, embeddings, labels, preds, num_classes,
            confusion_matrix, old_class_embeddings,
        )
        after_metrics = self._split_metrics(embeddings, labels, num_classes, known_classes)
        cache_delta = self.size - cache_before
        reward = agent.compute_reward(before_metrics, after_metrics, cache_delta)
        loss = agent.learn(reward)

        stats.action = action
        stats.val_acc = after_metrics["total_acc"]
        stats.old_acc = after_metrics["old_acc"]
        stats.merges = int(action == "merge")
        stats.splits = int(action in ("split", "boost_confused"))
        stats.evictions = int(action == "evict")
        stats.refreshes = int(action in ("refresh_old", "boost_confused"))
        logging.info(
            "RL evolve task=%d action=%s reward=%.3f loss=%.4f total_r=%.3f",
            task_id, action, reward, loss, agent.total_reward,
        )
        return stats

    @staticmethod
    def _build_cm(labels, preds, num_classes) -> np.ndarray:
        labels_t = labels if torch.is_tensor(labels) else torch.tensor(labels)
        preds_t = preds if torch.is_tensor(preds) else torch.tensor(preds)
        cm = np.zeros((num_classes, num_classes), dtype=np.float64)
        for true_l, pred_l in zip(labels_t.tolist(), preds_t.tolist()):
            if 0 <= int(true_l) < num_classes and 0 <= int(pred_l) < num_classes:
                cm[int(true_l), int(pred_l)] += 1.0
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return cm / row_sum

    def _split_metrics(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        known_classes: int,
    ) -> dict:
        if embeddings.numel() == 0:
            return {"total_acc": 0.0, "old_acc": 0.0, "new_acc": 0.0}
        logits = self.classify(embeddings.to(self.device), num_classes)
        preds = logits.argmax(dim=1).cpu()
        labels_cpu = labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels)
        total_acc = float((preds == labels_cpu).float().mean().item() * 100.0)

        old_acc, new_acc = total_acc, total_acc
        if known_classes > 0:
            old_mask = labels_cpu < known_classes
            if old_mask.any():
                old_acc = float((preds[old_mask] == labels_cpu[old_mask]).float().mean().item() * 100.0)
            new_mask = labels_cpu >= known_classes
            if new_mask.any():
                new_acc = float((preds[new_mask] == labels_cpu[new_mask]).float().mean().item() * 100.0)
        return {"total_acc": total_acc, "old_acc": old_acc, "new_acc": new_acc}

    def _refresh_old_prototypes(
        self,
        old_class_embeddings: dict,
        confusion_matrix: np.ndarray,
        known_classes: int,
    ) -> int:
        """Update primary keys for old classes with low recall."""
        if known_classes <= 0 or not old_class_embeddings:
            return 0
        refreshed = 0
        for cls in range(min(known_classes, confusion_matrix.shape[0])):
            recall = float(confusion_matrix[cls, cls])
            if recall >= self.recall_refresh_threshold:
                continue
            embeddings = old_class_embeddings.get(cls)
            if embeddings is None or embeddings.numel() == 0:
                continue
            proto = embeddings.to(self.device).mean(dim=0)
            mask = self.labels == cls
            if mask.any():
                idx = int(torch.where(mask)[0][0].item())
                self.keys[idx] = proto
                refreshed += 1
        return refreshed

    def _boost_confused(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        confusion_matrix: np.ndarray,
        old_class_embeddings: dict,
        known_classes: int,
    ) -> tuple[int, int]:
        """Split top confused pairs and refresh low-recall old classes."""
        splits = self._split_confused(embeddings, labels, preds, top_k=3)
        refreshes = self._refresh_old_prototypes(
            old_class_embeddings, confusion_matrix, known_classes
        )
        return splits, refreshes

    def _merge_similar(self) -> int:
        if self.size < 2:
            return 0
        merged = 0
        keys_norm = self._normalize(self.keys)
        keep_mask = torch.ones(self.size, dtype=torch.bool, device=self.device)

        for i in range(self.size):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, self.size):
                if not keep_mask[j]:
                    continue
                if self.labels[i] != self.labels[j]:
                    continue
                sim = float((keys_norm[i] * keys_norm[j]).sum().item())
                if sim >= self.merge_threshold:
                    self.keys[i] = (self.keys[i] + self.keys[j]) / 2.0
                    keep_mask[j] = False
                    merged += 1

        if merged:
            self._compact(keep_mask)
        return merged

    def _split_confused(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        top_k: int | None = None,
    ) -> int:
        splits = 0
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        preds = preds.to(self.device)

        confused_pairs: Dict[Tuple[int, int], int] = {}
        for true_l, pred_l in zip(labels.tolist(), preds.tolist()):
            if true_l != pred_l:
                confused_pairs[(true_l, pred_l)] = confused_pairs.get((true_l, pred_l), 0) + 1

        total = max(len(labels), 1)
        ranked = sorted(confused_pairs.items(), key=lambda x: -x[1])
        if top_k is not None:
            ranked = ranked[:top_k]

        for (true_c, pred_c), count in ranked:
            rate = count / total
            if top_k is None and rate < self.split_confusion_threshold:
                continue
            n_existing = int((self.labels == true_c).sum().item())
            if n_existing >= self.max_prototypes_per_class:
                continue
            if self.split_strategy == "hard_negative":
                true_idx = torch.where(labels == true_c)[0]
                if true_idx.numel() == 0:
                    continue
                true_emb = embeddings[true_idx]
                pred_mask = self.labels == pred_c
                if pred_mask.any():
                    pred_proto = self.keys[pred_mask].mean(dim=0, keepdim=True)
                    sim = F.cosine_similarity(self._normalize(true_emb), self._normalize(pred_proto), dim=1)
                    k_pick = max(1, int(0.5 * sim.numel()))
                    hard = torch.topk(sim, k=k_pick, largest=True).indices
                    mask = torch.zeros_like(labels, dtype=torch.bool)
                    mask[true_idx[hard]] = True
                else:
                    mask = (labels == true_c) & (preds == pred_c)
            else:
                mask = (labels == true_c) & (preds == pred_c)
            if not mask.any():
                mask = labels == true_c
            if not mask.any():
                continue
            proto = embeddings[mask].mean(dim=0)
            self._append_keys(proto.unsqueeze(0), torch.tensor([true_c], device=self.device))
            splits += 1
        return splits

    def _gated_grow_confused(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        confusion_matrix: np.ndarray,
        class_lo: int,
        class_hi: int,
        recall_threshold: float | None = None,
        max_classes: int | None = None,
    ) -> int:
        """Add at most one extra prototype for hardest low-recall classes."""
        if not self.enable_gated_growth or class_hi <= class_lo:
            return 0

        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        preds = preds.to(self.device)
        thr = (
            self.gated_growth_recall_threshold
            if recall_threshold is None
            else recall_threshold
        )
        max_cls = (
            self.gated_growth_max_classes if max_classes is None else max_classes
        )

        candidates: list[tuple[float, int]] = []
        for cls in range(class_lo, min(class_hi, confusion_matrix.shape[0])):
            recall = float(confusion_matrix[cls, cls])
            if recall < thr:
                candidates.append((recall, cls))
        if not candidates:
            return 0

        candidates.sort(key=lambda x: x[0])
        added = 0
        for _, cls in candidates[:max_cls]:
            n_existing = int((self.labels == cls).sum().item())
            if n_existing >= self.max_prototypes_per_class:
                continue
            mask = (labels == cls) & (preds != cls)
            if not mask.any():
                mask = labels == cls
            if not mask.any():
                continue
            proto = embeddings[mask].mean(dim=0)
            self._append_keys(
                proto.unsqueeze(0),
                torch.tensor([cls], dtype=torch.long, device=self.device),
            )
            added += 1
        return added

    def _evict_if_needed(self) -> int:
        """Evict lowest-margin keys until within budget.

        Never removes the last key for a class so classify() always has
        a finite logit for every represented label.
        """
        evicted = 0
        while self.size > self.cache_budget:
            counts = torch.bincount(self.labels[: self.size], minlength=int(self.labels.max().item()) + 1)
            sole_mask = counts[self.labels[: self.size]] <= 1
            margins = self.margins[: self.size].clone()
            margins[sole_mask] = float("inf")
            if not torch.isfinite(margins).any():
                break
            idx = int(margins.argmin().item())
            keep_mask = torch.ones(self.size, dtype=torch.bool, device=self.device)
            keep_mask[idx] = False
            self._compact(keep_mask)
            evicted += 1
        return evicted

    def _compact(self, keep_mask: torch.Tensor) -> None:
        self.keys = self.keys[keep_mask]
        self.labels = self.labels[keep_mask]
        self.margins = self.margins[keep_mask]

    def update_margins(
        self, embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int
    ) -> None:
        """Track per-key margin for eviction policy."""
        if self.size == 0:
            return
        embeddings = embeddings.to(self.device)
        logits = self.classify(embeddings, num_classes)
        top2 = logits.topk(2, dim=1).values
        sample_margin = top2[:, 0] - top2[:, 1]

        key_margin = torch.zeros(self.size, device=self.device)
        key_count = torch.zeros(self.size, device=self.device)
        feat = self._normalize(embeddings)
        keys = self._normalize(self.keys)
        sim = feat @ keys.T
        nearest = sim.argmax(dim=1)

        for i, key_idx in enumerate(nearest.tolist()):
            key_margin[key_idx] += sample_margin[i]
            key_count[key_idx] += 1

        valid = key_count > 0
        self.margins[valid] = key_margin[valid] / key_count[valid]

    def _select_bandit_action(self) -> str:
        total = sum(self.bandit.counts.values()) + 1
        best_action = "skip"
        best_score = -1e9
        for action in _ACTIONS:
            n = self.bandit.counts[action]
            mean_r = self.bandit.rewards[action] / max(n, 1)
            ucb = mean_r + self.bandit_ucb_c * np.sqrt(np.log(total) / max(n, 1))
            if ucb > best_score:
                best_score = ucb
                best_action = action
        return best_action

    def _update_bandit(self, action: str, reward: float) -> None:
        self.bandit.counts[action] += 1
        self.bandit.rewards[action] += reward

    def _apply_action(
        self,
        action: str,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        num_classes: int,
        confusion_matrix: np.ndarray | None = None,
        old_class_embeddings: dict | None = None,
    ) -> None:
        old_class_embeddings = old_class_embeddings or {}
        known_classes = int(max(old_class_embeddings.keys()) + 1) if old_class_embeddings else 0
        if confusion_matrix is None:
            confusion_matrix = self._build_cm(labels, preds, num_classes)

        if action == "merge":
            self._merge_similar()
        elif action == "split":
            self._split_confused(embeddings, labels, preds)
        elif action == "evict":
            self._evict_if_needed()
        elif action == "refresh_old":
            self._refresh_old_prototypes(old_class_embeddings, confusion_matrix, known_classes)
        elif action == "boost_confused":
            self._boost_confused(
                embeddings, labels, preds, confusion_matrix, old_class_embeddings, known_classes
            )
        self.update_margins(embeddings, labels, num_classes)

    def state_dict(self) -> dict:
        out = {
            "keys": self.keys.cpu(),
            "labels": self.labels.cpu(),
            "margins": self.margins.cpu(),
            "bandit_counts": dict(self.bandit.counts),
            "bandit_rewards": dict(self.bandit.rewards),
            "history": [s.__dict__ for s in self.history],
            "gauss_count": dict(self._gauss_count),
            "gauss_mean": {k: v.cpu() for k, v in self._gauss_mean.items()},
            "gauss_var": {k: v.cpu() for k, v in self._gauss_var.items()},
        }
        if self.rl_agent is not None:
            out["rl_agent"] = self.rl_agent.state_dict()
        return out

    def load_state_dict(self, state: dict) -> None:
        self.keys = state["keys"].to(self.device)
        self.labels = state["labels"].to(self.device)
        self.margins = state["margins"].to(self.device)
        if "bandit_counts" in state:
            self.bandit.counts.update(state["bandit_counts"])
        if "bandit_rewards" in state:
            self.bandit.rewards.update(state["bandit_rewards"])
        if "history" in state:
            self.history = [EvolutionStats(**h) for h in state["history"]]
        self._gauss_count = {
            int(k): int(v) for k, v in state.get("gauss_count", {}).items()
        }
        self._gauss_mean = {
            int(k): v.to(self.device) for k, v in state.get("gauss_mean", {}).items()
        }
        self._gauss_var = {
            int(k): v.to(self.device) for k, v in state.get("gauss_var", {}).items()
        }
        if self.rl_agent is not None and "rl_agent" in state:
            self.rl_agent.load_state_dict(state["rl_agent"])

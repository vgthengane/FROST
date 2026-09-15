"""SelfEvo: frozen Uni3D + self-evolving key-cache memory."""

from __future__ import annotations

import copy
import importlib.util
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.toolkit import tensor2numpy

from datasets.fscil_protocol import prepare_fscil_task_datasets
from models.base import BaseLearner
from utils.inc_net import SimpleVitNet
from utils.toolkit import safe_topk_indices

_models_dir = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _models_dir / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_key_cache_mod = _load_module("key_cache", "key_cache.py")
_rl_mod = _load_module("rl_evolver", "rl_evolver.py")
_ridge_mod = _load_module("ridge_head", "ridge_head.py")
_fscil_cfg_mod = _load_module("fscil_ablation_config", "fscil_ablation_config.py")
_text_mod = _load_module("text_prototype_head", "text_prototype_head.py")
KeyCache = _key_cache_mod.KeyCache
RLEvolutionAgent = _rl_mod.RLEvolutionAgent
RidgeHead = _ridge_mod.RidgeHead
prepare_fscil_args = _fscil_cfg_mod.prepare_fscil_args
TextPrototypeHead = _text_mod.TextPrototypeHead
ordered_class_names = _text_mod.ordered_class_names
build_confusion_matrix = _rl_mod.build_confusion_matrix

num_workers = 8


def _unpack_batch(batch, device):
    """Unpack ModelNet40 batches (supports optional type / multi-view)."""
    if not isinstance(batch, (list, tuple)) or len(batch) < 3:
        raise ValueError(f"Unexpected batch format: {type(batch)}")
    has_type = not torch.is_tensor(batch[-1])
    label = batch[-2] if has_type else batch[-1]
    data = batch[1]
    return data.to(device), label.to(device)


class Learner(BaseLearner):
    """Frozen backbone + key-cache with optional ridge head and evolution."""

    def __init__(self, args):
        args, self._modules = prepare_fscil_args(dict(args))
        self.fscil_shot_protocol = bool(args.get("fscil_shot_protocol", False))
        if self.fscil_shot_protocol and bool(args.get("fscil_adapter_base_only", False)):
            logging.info(
                "fscil_adapter_base_only: task0_warmup=%s increment_warmup=%s",
                self._modules.adapter.task0_warmup,
                self._modules.adapter.increment_warmup,
            )
        if args.get("max_train_per_class") is not None and self.fscil_shot_protocol:
            logging.info(
                "fscil_shot_protocol: ignoring global max_train_per_class=%s "
                "(task 0 uses full data; later tasks use increment_train_per_class)",
                args.pop("max_train_per_class"),
            )
        self.task0_adapter_warmup = self._modules.adapter.task0_warmup
        if self.task0_adapter_warmup:
            if "_adapter" not in str(args.get("backbone_type", "")):
                args["backbone_type"] = "uni3d_base_patch14_448_adapter"
            args.setdefault("ffn_num", 64)
            args.setdefault("tuned_epoch", 5)
            args.setdefault("init_lr", 0.01)
            args.setdefault("weight_decay", 0.0005)
            args.setdefault("min_lr", 0.0)
            args.setdefault("optimizer", "sgd")
        super().__init__(args)
        self._task0_best_adapter_state: dict | None = None
        self._network = SimpleVitNet(self.args, True)
        if self.task0_adapter_warmup:
            self._set_adapter_warmup_trainable(train_fc=True)
        else:
            self._freeze_backbone()
        self.rl_agent = None
        self.ridge_head = None

        policy = self._modules.evolution.policy
        if policy == "rl":
            self.rl_agent = RLEvolutionAgent(
                max_classes=int(args.get("num_category", 40)),
                lr=float(args.get("rl_lr", 1e-3)),
                device=self._device,
                reward_old_weight=float(args.get("rl_reward_old_weight", 3.0)),
                reward_new_weight=float(args.get("rl_reward_new_weight", 1.0)),
                reward_cache_penalty=float(args.get("rl_cache_penalty", 0.05)),
            )

        self.classifier_mode = self._modules.classifier.mode
        self._uses_ridge = self._modules.classifier.uses_ridge
        self._uses_text = self.classifier_mode in (
            "text",
            "ridge_text",
            "ridge_cache_text",
        )
        self.text_fusion_gamma = float(args.get("text_fusion_gamma", 0.10))
        cr = self._modules.conflict_replay
        self.conflict_replay = cr.enabled
        self.conflict_replay_on_test = cr.use_test_confusion_matrix
        self.replay_weight = cr.old_weight
        self.novel_replay_weight = cr.novel_weight
        self.replay_recall_threshold = cr.old_recall_threshold
        self.increment_adapter_warmup = self._modules.adapter.increment_warmup
        self.ridge_cache_alpha = self._modules.classifier.ridge_cache_alpha
        self._ncm_prototypes: torch.Tensor | None = None
        self._apply_low_shot_blend(args)

        if self._uses_ridge:
            seed_val = args.get("seed", 0)
            if isinstance(seed_val, list):
                seed_val = seed_val[0]
            self.ridge_head = RidgeHead(
                feature_dim=self.feature_dim,
                rp_dim=int(args.get("ridge_rp_dim", args.get("M", 1000))),
                nb_classes=int(args.get("nb_classes", args.get("num_category", 40))),
                device=self._device,
                ridge=args.get("ridge_lambda"),
                seed=int(seed_val),
                ridge_val_fraction=float(args.get("ridge_val_fraction", 0.2)),
                ridge_lambda_min=float(args.get("ridge_lambda_min", 1e-8)),
                ridge_lambda_max=args.get("ridge_lambda_max"),
            )

        self.text_head: TextPrototypeHead | None = None
        if self._uses_text:
            self.text_head = TextPrototypeHead(
                device=self._device,
                embed_dim=int(args.get("embed_dim", self.feature_dim)),
                prompt_template=str(
                    args.get("text_prompt_template", "a point cloud model of a {}")
                ),
                clip_model=str(args.get("text_clip_model", "EVA02-E-14-plus")),
                clip_pretrained=str(args.get("text_clip_pretrained", "eva02")),
                clip_checkpoint=args.get("text_clip_checkpoint"),
                logit_scale=args.get("text_logit_scale"),
                prototype_path=args.get("text_prototype_path"),
            )
            for param in self.text_head.parameters():
                param.requires_grad = False

        self.cache = KeyCache(
            feature_dim=self.feature_dim,
            device=self._device,
            max_prototypes_per_class=int(args.get("max_prototypes_per_class", 3)),
            merge_threshold=float(args.get("merge_threshold", 0.95)),
            split_confusion_threshold=float(args.get("split_confusion_threshold", 0.15)),
            cache_budget=int(args.get("cache_budget", 120)),
            evolution_policy=policy,
            bandit_ucb_c=float(args.get("bandit_ucb_c", 1.0)),
            rl_agent=self.rl_agent,
            recall_refresh_threshold=float(args.get("recall_refresh_threshold", 0.85)),
            prototype_pooling=str(args.get("prototype_pooling", "max")),
            pooling_temperature=float(args.get("pooling_temperature", 0.1)),
            insert_strategy=str(args.get("insert_strategy", "mean")),
            split_strategy=str(args.get("split_strategy", "confusion_pair")),
            enable_gated_growth=bool(args.get("enable_gated_growth", False)),
            gated_growth_recall_threshold=float(
                args.get("gated_growth_recall_threshold", 0.9)
            ),
            gated_growth_max_classes=int(args.get("gated_growth_max_classes", 2)),
            gated_growth_include_novel=bool(args.get("fscil_gated_growth_novel", False)),
            novel_gated_growth_threshold=float(
                args.get("fscil_novel_gated_growth_threshold", 0.85)
            ),
            novel_gated_growth_max_classes=int(
                args.get("fscil_novel_gated_growth_max_classes", 5)
            ),
        )

    def _apply_low_shot_blend(self, args: dict) -> None:
        """Reduce ridge weight when per-class training caps are very small."""
        if not bool(args.get("low_shot_cache_blend", False)):
            return
        cap = args.get("max_train_per_class")
        if cap is None:
            return
        cap = int(cap)
        if cap >= 20:
            return
        base_alpha = float(args.get("ridge_cache_alpha", 0.85))
        floor_alpha = float(args.get("low_shot_ridge_alpha", 0.60))
        # cap=1 -> floor; cap=20 -> base_alpha
        t = min(1.0, max(0.0, (cap - 1) / 19.0))
        self.ridge_cache_alpha = floor_alpha + t * (base_alpha - floor_alpha)
        logging.info(
            "Low-shot ridge/cache blend: max_train_per_class=%d alpha=%.3f",
            cap,
            self.ridge_cache_alpha,
        )

    def _freeze_backbone(self) -> None:
        for param in self._network.backbone.parameters():
            param.requires_grad = False
        if self._network.fc is not None:
            for param in self._network.fc.parameters():
                param.requires_grad = False

    def _set_adapter_warmup_trainable(self, train_fc: bool) -> None:
        """Enable adapter (+ optional fc) params for task-0 warmup."""
        for param in self._network.backbone.parameters():
            param.requires_grad = False
        for param in self._network.backbone.adaptmlp.parameters():
            param.requires_grad = True
        if self._network.fc is not None:
            for param in self._network.fc.parameters():
                param.requires_grad = train_fc

    def _warmup_lr(self) -> float:
        """LR for adapter warmup; task 0 may use ``task0_init_lr``."""
        if self._cur_task == 0 and self.args.get("task0_init_lr") is not None:
            return float(self.args["task0_init_lr"])
        return float(self.args.get("init_lr", 0.01))

    def _task0_warmup_ckpt_dir(self) -> Path | None:
        if self._cur_task != 0:
            return None
        log_dir = self.args.get("log_dir")
        if not log_dir:
            return None
        method = str(self.args.get("method_name", self.args.get("prefix", "selfevo")))
        seed = self.args.get("seed", 0)
        if isinstance(seed, list):
            seed = seed[0]
        return Path(log_dir) / method / f"task0_warmup_seed={seed}"

    def _save_task0_warmup_ckpts(self) -> bool:
        if self._cur_task != 0:
            return False
        return bool(self.args.get("task0_save_warmup_ckpts", True))

    def _save_task0_warmup_state(
        self,
        tag: str,
        *,
        epoch: int,
        test_acc: float,
        train_acc: float,
        init_lr: float,
        state_dict: dict,
    ) -> Path:
        ckpt_dir = self._task0_warmup_ckpt_dir()
        assert ckpt_dir is not None
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{tag}.pt"
        torch.save(
            {
                "tag": tag,
                "epoch": int(epoch),
                "test_acc": float(test_acc),
                "train_acc": float(train_acc),
                "task0_init_lr": float(init_lr),
                "model_state_dict": copy.deepcopy(state_dict),
            },
            path,
        )
        logging.info(
            "Task 0 warmup checkpoint %s: epoch=%d test_acc=%.2f train_acc=%.2f",
            path,
            epoch,
            test_acc,
            train_acc,
        )
        return path

    def _finalize_task0_warmup_ckpts(
        self,
        *,
        best_epoch: int,
        best_test_acc: float,
        best_train_acc: float,
        last_epoch: int,
        last_test_acc: float,
        last_train_acc: float,
        init_lr: float,
        best_state: dict | None,
        last_state: dict,
    ) -> None:
        if not self._save_task0_warmup_ckpts():
            return
        self._save_task0_warmup_state(
            "last",
            epoch=last_epoch,
            test_acc=last_test_acc,
            train_acc=last_train_acc,
            init_lr=init_lr,
            state_dict=last_state,
        )
        if best_state is None:
            logging.warning("Task 0 warmup: no best checkpoint tracked; using last weights")
            self._save_task0_adapter_anchor()
            return
        self._save_task0_warmup_state(
            "best",
            epoch=best_epoch,
            test_acc=best_test_acc,
            train_acc=best_train_acc,
            init_lr=init_lr,
            state_dict=best_state,
        )
        if bool(self.args.get("task0_use_best_warmup", True)):
            self._unwrap_network().load_state_dict(best_state, strict=True)
            logging.info(
                "Task 0 warmup: restored best weights (epoch=%d test_acc=%.2f)",
                best_epoch,
                best_test_acc,
            )
        else:
            self._unwrap_network().load_state_dict(last_state, strict=True)
        self._save_task0_adapter_anchor()

    @staticmethod
    def _adapter_anchor_slice(state_dict: dict) -> dict:
        """Backbone-only weights for task-0 anchor (excludes ``fc`` head sized per task)."""
        return {
            k: copy.deepcopy(v)
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }

    def _save_task0_adapter_anchor(self) -> None:
        """Keep task-0 adapter weights to restore before increment warmup."""
        self._task0_best_adapter_state = self._adapter_anchor_slice(
            self._unwrap_network().state_dict()
        )
        logging.info("Task 0 adapter anchor saved for increment warmup restore")

    def _restore_task0_adapter_anchor_if_needed(self) -> None:
        """Start increment warmup from the task-0 best adapter, not drifted weights."""
        if self._cur_task <= 0:
            return
        if not bool(
            self.args.get("fscil_restore_task0_adapter_before_increment_warmup", True)
        ):
            return
        if self._task0_best_adapter_state is None:
            logging.warning(
                "Increment warmup task=%d: task-0 adapter anchor missing; "
                "continuing from current weights",
                self._cur_task,
            )
            return
        current = self._unwrap_network().state_dict()
        to_load = {
            k: v
            for k, v in self._task0_best_adapter_state.items()
            if k in current and current[k].shape == v.shape
        }
        if not to_load:
            logging.warning(
                "Increment warmup task=%d: no matching backbone keys in task-0 anchor",
                self._cur_task,
            )
            return
        self._unwrap_network().load_state_dict(to_load, strict=False)
        logging.info(
            "Increment warmup task=%d: restored task-0 best adapter anchor (%d keys)",
            self._cur_task,
            len(to_load),
        )

    def _increment_adapter_warmup_enabled(self) -> bool:
        if self._cur_task <= 0:
            return False
        if bool(self.args.get("fscil_adapter_base_only", False)):
            return bool(self.args.get("fscil_task1_adapter_warmup", False)) and (
                self._cur_task == 1
            )
        return bool(self.increment_adapter_warmup)

    def _init_adapter_train(
        self, train_loader, test_loader, tuned_epoch: int | None = None
    ) -> None:
        """Adapter SGD/AdamW on labeled train data before ridge/cache adaptation."""
        if tuned_epoch is None:
            tuned_epoch = int(self.args.get("tuned_epoch", 5))
        init_lr = self._warmup_lr()
        if self._cur_task == 0:
            optimizer_name = str(
                self.args.get("task0_optimizer") or self.args.get("optimizer", "sgd")
            ).lower()
            weight_decay = float(
                self.args.get("task0_weight_decay", self.args.get("weight_decay", 0.0005))
            )
        else:
            optimizer_name = str(self.args.get("optimizer", "sgd")).lower()
            weight_decay = float(self.args.get("weight_decay", 0.0005))
        min_lr = float(self.args.get("min_lr", 0.0))
        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=init_lr,
                weight_decay=weight_decay,
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tuned_epoch, eta_min=min_lr
        )

        use_point_aug = self._cur_task == 0 and bool(
            self.args.get("task0_point_augment", False)
        )
        trainable = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(
            "Adapter warmup task=%d: %d trainable params, %d epochs, n=%d, "
            "lr=%.6f, optimizer=%s, wd=%.1e, point_aug=%s",
            self._cur_task,
            trainable,
            tuned_epoch,
            len(train_loader.dataset),
            init_lr,
            optimizer_name,
            weight_decay,
            use_point_aug,
        )

        info = ""
        best_test_acc = -1.0
        best_epoch = 0
        best_train_acc = 0.0
        best_state = None
        last_train_acc = 0.0
        last_test_acc = 0.0
        prog_bar = tqdm(range(tuned_epoch))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for batch in train_loader:
                inputs, targets = _unpack_batch(batch, self._device)
                if use_point_aug:
                    from datasets.pointcloud_augment import augment_task0_point_cloud

                    inputs = augment_task0_point_cloud(
                        inputs,
                        scale_low=float(self.args.get("task0_aug_scale_low", 0.8)),
                        scale_high=float(self.args.get("task0_aug_scale_high", 1.25)),
                        shift_range=float(self.args.get("task0_aug_shift_range", 0.1)),
                        rotate_z=bool(self.args.get("task0_aug_rotate_z", True)),
                    )
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            test_acc = self._adapter_warmup_test_acc(test_loader)
            last_train_acc = float(train_acc)
            last_test_acc = float(test_acc)
            if (
                self._cur_task == 0
                and self._save_task0_warmup_ckpts()
                and test_acc >= best_test_acc
            ):
                best_test_acc = float(test_acc)
                best_epoch = epoch + 1
                best_train_acc = float(train_acc)
                best_state = copy.deepcopy(self._unwrap_network().state_dict())
            info = (
                f"Task {self._cur_task} warmup Epoch {epoch + 1}/{tuned_epoch} => "
                f"Loss {losses / max(len(train_loader), 1):.3f}, "
                f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
            )
            prog_bar.set_description(info)
        logging.info(info)
        if self._cur_task == 0 and self._save_task0_warmup_ckpts():
            self._finalize_task0_warmup_ckpts(
                best_epoch=best_epoch,
                best_test_acc=best_test_acc,
                best_train_acc=best_train_acc,
                last_epoch=tuned_epoch,
                last_test_acc=last_test_acc,
                last_train_acc=last_train_acc,
                init_lr=init_lr,
                best_state=best_state,
                last_state=copy.deepcopy(self._unwrap_network().state_dict()),
            )
        elif self._cur_task == 0 and self.task0_adapter_warmup:
            self._save_task0_adapter_anchor()
        if self._cur_task > 0:
            logging.info(
                "Increment warmup task=%d: using last-epoch adapter weights (epoch=%d)",
                self._cur_task,
                tuned_epoch,
            )

    def _init_classifier_warmup_train(
        self, train_loader, test_loader, tuned_epoch: int
    ) -> None:
        """Adapter SGD using the active ridge/cache classifier (FSCIL few-shot)."""
        init_lr = float(self.args.get("init_lr", 0.01))
        weight_decay = float(self.args.get("weight_decay", 0.0005))
        min_lr = float(self.args.get("min_lr", 0.0))
        optimizer_name = str(self.args.get("optimizer", "sgd")).lower()
        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=init_lr,
                weight_decay=weight_decay,
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tuned_epoch, eta_min=min_lr
        )

        trainable = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(
            "Classifier-aligned warmup task=%d: %d trainable params, %d epochs, n=%d",
            self._cur_task,
            trainable,
            tuned_epoch,
            len(train_loader.dataset),
        )

        info = ""
        prog_bar = tqdm(range(tuned_epoch))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for batch in train_loader:
                inputs, targets = _unpack_batch(batch, self._device)
                emb = self._unwrap_network().backbone(inputs)
                logits = self._classify(emb, self._total_classes)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            test_acc = self._classifier_warmup_test_acc(test_loader)
            info = (
                f"Task {self._cur_task} clf-warmup Epoch {epoch + 1}/{tuned_epoch} => "
                f"Loss {losses / max(len(train_loader), 1):.3f}, "
                f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
            )
            prog_bar.set_description(info)
        logging.info(info)
        if self._cur_task > 0:
            logging.info(
                "Increment clf-warmup task=%d: using last-epoch adapter weights (epoch=%d)",
                self._cur_task,
                tuned_epoch,
            )

    def _classifier_warmup_test_acc(self, test_loader) -> float:
        self._network.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = _unpack_batch(batch, self._device)
                emb = self._unwrap_network().backbone(inputs)
                logits = self._classify(emb, self._total_classes)
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
        return float(np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2))

    def _adapter_warmup_test_acc(self, test_loader) -> float:
        self._network.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = _unpack_batch(batch, self._device)
                logits = self._network(inputs)["logits"]
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
        return float(np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2))

    def after_task(self):
        self._known_classes = self._total_classes

    def _log_task_class_names(self, data_manager) -> None:
        """Log human-readable class names for the current incremental task."""
        protocol = str(self.args.get("fscil3d_protocol", "")).strip()
        class_indices = list(range(self._known_classes, self._total_classes))
        if not class_indices:
            return

        if protocol in ("modelnet_scanobjectnn", "shapenet_scanobjectnn"):
            from datasets.scanobjectnn_taxonomy import (
                cross_scanobjectnn_class_label,
                format_cross_scanobjectnn_task_classes,
            )

            datasets = {cross_scanobjectnn_class_label(protocol, i)[0] for i in class_indices}
            domain = datasets.pop() if len(datasets) == 1 else "mixed"
            logging.info(
                "Task %d classes (%d, %s):\n%s",
                self._cur_task + 1,
                len(class_indices),
                domain,
                format_cross_scanobjectnn_task_classes(protocol, class_indices),
            )
            return

        if protocol == "modelnet" and self.args.get("data_path"):
            from datasets.modelnet40 import iModelNet40

            names = iModelNet40.shape_names(str(self.args["data_path"]))
            order = data_manager._class_order
            lines = []
            for idx in class_indices:
                orig = order[idx]
                label = names[orig] if orig < len(names) else str(orig)
                lines.append(f"  [{idx:2d}] {label} (ModelNet)")
            logging.info("Task %d classes (%d, ModelNet):\n%s", self._cur_task + 1, len(class_indices), "\n".join(lines))
            return

        logging.info(
            "Task %d class indices: %s",
            self._cur_task + 1,
            class_indices,
        )

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        if self._network.fc is not None:
            for param in self._network.fc.parameters():
                param.requires_grad = False
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self._log_task_class_names(data_manager)

        if self.fscil_shot_protocol:
            train_dataset, proto_dataset = prepare_fscil_task_datasets(
                data_manager,
                self._known_classes,
                self._total_classes,
                self._cur_task,
                self.args,
            )
        else:
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
            )
            proto_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="test",
            )

        self.train_dataset = train_dataset
        self.data_manager = data_manager
        bs = self.args.get("batch_size", 128)
        self.train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
        )
        self.train_loader_for_protonet = DataLoader(
            proto_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
        )
        self._sync_text_prototypes(data_manager, self._total_classes)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _collect_old_class_embeddings(self, data_manager, bs: int) -> dict:
        """Old-class embedding proxies for adaptation / conflict replay (not raw buffer)."""
        old_embeddings = {}
        if self._known_classes <= 0:
            return old_embeddings

        proxies = self._modules.old_class_proxies
        if proxies.disabled:
            return old_embeddings

        if proxies.use_cache:
            if self.cache.size == 0:
                return old_embeddings
            labels = self.cache.labels.detach().cpu()
            keys = self.cache.keys.detach().cpu()
            for cls in range(self._known_classes):
                mask = labels == cls
                if mask.any():
                    old_embeddings[cls] = keys[mask]
            return old_embeddings

        buf = self._modules.exemplar_buffer
        if self.fscil_shot_protocol and buf.shots_per_old_class > 0:
            emb, lbl = self._collect_class_train_embeddings(
                data_manager,
                0,
                self._known_classes,
                buf.shots_per_old_class,
                cache_role="old_replay",
                role_seed=buf.shot_seed,
            )
            if emb.numel() == 0:
                return old_embeddings
            labels_cpu = lbl.detach().cpu().long()
            for cls in range(self._known_classes):
                mask = labels_cpu == cls
                if mask.any():
                    old_embeddings[cls] = emb[mask]
            return old_embeddings

        for cls in range(self._known_classes):
            ds = data_manager.get_dataset([cls], source="train", mode="test")
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
            emb, _ = self._collect_embeddings(loader)
            old_embeddings[cls] = emb
        return old_embeddings

    def _effective_ridge_alpha(self) -> float:
        """Ridge weight in ridge_cache fusion; lower on few-shot incremental tasks."""
        if not self.fscil_shot_protocol or self._cur_task <= 0:
            return self.ridge_cache_alpha
        base = float(self.args.get("ridge_cache_alpha", 0.85))
        floor = float(self.args.get("fscil_increment_ridge_alpha", 0.55))
        task1_alpha = self.args.get("fscil_task1_ridge_alpha")
        start = float(task1_alpha) if task1_alpha is not None else floor
        incr = int(self.args.get("increment_train_per_class", 5))
        t_shot = min(1.0, max(0.0, (incr - 1) / 19.0))
        target = floor + t_shot * (base - floor)
        last = int(self.args.get("fscil_last_task_index", 4))
        last_alpha = self.args.get("fscil_last_task_ridge_alpha")
        if last_alpha is not None and self._cur_task >= last:
            return float(last_alpha)
        if not bool(self.args.get("fscil_ridge_alpha_ramp", False)):
            if self._cur_task == 1 and task1_alpha is not None:
                return float(task1_alpha)
            return target
        if last <= 1 or self._cur_task >= last:
            return target
        ramp = (self._cur_task - 1) / max(last - 1, 1)
        return start + ramp * (target - start)

    def _increment_warmup_epochs(self) -> int:
        epochs = int(self.args.get("increment_tuned_epoch", 3))
        if self._cur_task == 1:
            epochs = int(self.args.get("fscil_task1_warmup_epochs", epochs))
        last_task = int(self.args.get("fscil_last_task_index", 4))
        if self._cur_task >= last_task:
            epochs = int(self.args.get("fscil_last_task_warmup_epochs", epochs))
        return epochs

    def _is_last_fscil_task(self) -> bool:
        """True on the final incremental session (few-shot tasks only)."""
        if not self.fscil_shot_protocol or self._cur_task <= 0:
            return False
        last_idx = int(self.args.get("fscil_last_task_index", 4))
        return self._cur_task >= last_idx

    def _augment_few_shot_for_ridge(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Light embedding noise doubling for ridge stats on few-shot tasks."""
        std = float(self.args.get("fscil_ridge_proto_noise_std", 0.0))
        if std <= 0.0 or self._cur_task <= 0:
            return embeddings, labels
        seed_val = self.args.get("seed", 1993)
        if isinstance(seed_val, list):
            seed_val = seed_val[0]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed_val) + self._cur_task * 31)
        noise = torch.randn(embeddings.shape, generator=gen) * std
        aug_emb = embeddings + noise
        return torch.cat([embeddings, aug_emb], dim=0), torch.cat([labels, labels], dim=0)

    def _uses_cache_inference(self) -> bool:
        return self._modules.classifier.uses_cache_inference

    def _ensure_ncm_prototypes(self) -> None:
        if self._ncm_prototypes is not None:
            return
        nb = int(self.args.get("num_category", 40))
        self._ncm_prototypes = torch.zeros(nb, self.feature_dim)

    def _update_ncm_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_ids: np.ndarray,
        *,
        freeze_base: bool = False,
    ) -> None:
        """Running mean prototypes; base classes can stay frozen after task 0."""
        self._ensure_ncm_prototypes()
        labels_cpu = labels.detach().cpu().long()
        emb_cpu = embeddings.detach().cpu().float()
        for cls in class_ids:
            cls = int(cls)
            if freeze_base and self._cur_task > 0 and cls < self._known_classes:
                continue
            mask = labels_cpu == cls
            if mask.any():
                self._ncm_prototypes[cls] = emb_cpu[mask].mean(dim=0)

    def _ncm_logits(self, embeddings: torch.Tensor, num_classes: int) -> torch.Tensor:
        if self._ncm_prototypes is None:
            return torch.zeros(
                embeddings.shape[0], num_classes, device=embeddings.device
            )
        means = self._ncm_prototypes[:num_classes].to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        feat = F.normalize(embeddings, dim=-1)
        proto = F.normalize(means, dim=-1)
        return feat @ proto.T

    def _exemplar_buffer_shot_seed(self) -> int:
        return self._modules.exemplar_buffer.shot_seed

    def _collect_class_train_embeddings(
        self,
        data_manager,
        class_lo: int,
        class_hi: int,
        shots_per_class: int,
        *,
        cache_role: str = "ridge_refresh",
        role_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample up to ``shots_per_class`` train embeddings per class."""
        if class_hi <= class_lo or shots_per_class <= 0:
            return torch.empty(0, self.feature_dim), torch.empty(0, dtype=torch.long)
        seed_val = self.args.get("seed", 1993)
        if isinstance(seed_val, list):
            seed_val = seed_val[0]
        emb_parts, label_parts = [], []
        bs = int(self.args.get("batch_size", 128))
        if role_seed is None:
            role_seed = (
                self._exemplar_buffer_shot_seed()
                if cache_role == "old_replay"
                else int(self.args.get("fscil_base_ridge_refresh_seed", int(seed_val)))
            )
        for cls in range(class_lo, class_hi):
            ds = data_manager.get_dataset([cls], source="train", mode="test")
            if len(ds) == 0:
                continue
            n = min(int(shots_per_class), len(ds))
            from datasets.fscil_protocol import _pick_class_indices

            cache_key = (cache_role, cls, role_seed, int(shots_per_class))
            idx = _pick_class_indices(len(ds), n, cache_key=cache_key)
            loader = DataLoader(
                Subset(ds, idx.tolist()),
                batch_size=bs,
                shuffle=False,
                num_workers=num_workers,
            )
            emb, lbl = self._collect_embeddings(loader)
            emb_parts.append(emb)
            label_parts.append(lbl)
        if not emb_parts:
            return torch.empty(0, self.feature_dim), torch.empty(0, dtype=torch.long)
        return torch.cat(emb_parts, dim=0), torch.cat(label_parts, dim=0)

    def _refresh_ridge_from_exemplar_buffer(self) -> int:
        """Re-anchor ridge on old-class embeddings from the exemplar buffer (disk)."""
        if (
            self.ridge_head is None
            or not self.fscil_shot_protocol
            or self._cur_task <= 0
            or self._known_classes <= 0
        ):
            return 0
        buf = self._modules.exemplar_buffer
        shots = buf.ridge_refresh_shots_per_old_class
        if shots <= 0:
            return 0
        shot_cap = buf.shots_per_old_class
        if shots > shot_cap and shot_cap > 0:
            logging.warning(
                "exemplar_buffer_ridge_refresh_shots=%d capped to %d "
                "(exemplar_buffer_shots)",
                shots,
                shot_cap,
            )
            shots = shot_cap
        old_emb, old_lbl = self._collect_class_train_embeddings(
            self.data_manager,
            0,
            self._known_classes,
            shots,
            cache_role="old_replay",
            role_seed=buf.shot_seed,
        )
        if old_emb.numel() == 0:
            return 0
        weight = buf.ridge_refresh_weight
        self.ridge_head.update(old_emb, old_lbl, weight=weight)
        logging.info(
            "Exemplar-buffer ridge refresh task=%d classes=0-%d shots/class=%d n=%d weight=%.2f",
            self._cur_task,
            self._known_classes - 1,
            shots,
            int(old_emb.shape[0]),
            weight,
        )
        return int(old_emb.shape[0])

    def _sync_text_prototypes(self, data_manager, num_classes: int) -> None:
        if self.text_head is None:
            return
        dataset = str(self.args.get("dataset", ""))
        if dataset != "modelnet40":
            logging.warning("Text prototypes: unsupported dataset %r", dataset)
            return
        from datasets.modelnet40 import iModelNet40

        shape_names = iModelNet40.shape_names(self.args.get("data_path"))
        names = ordered_class_names(shape_names, data_manager._class_order, num_classes)
        self.text_head.set_class_names(
            names,
            class_order=data_manager._class_order[:num_classes],
        )

    def _text_logits(self, embeddings: torch.Tensor, num_classes: int) -> torch.Tensor:
        if self.text_head is None:
            raise RuntimeError("text_head is not initialized")
        return self.text_head.classify(embeddings, num_classes)

    def _classify(self, embeddings: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Dispatch to cache, ridge, text, NCM blend, or fused classifier."""
        if self.classifier_mode == "text":
            return self._text_logits(embeddings, num_classes)

        if self.classifier_mode == "ridge" and self.ridge_head is not None:
            return self.ridge_head.classify(embeddings, num_classes)

        if self.classifier_mode == "ridge_ncm" and self.ridge_head is not None:
            ridge_logits = self.ridge_head.classify(embeddings, num_classes)
            ncm_logits = self._ncm_logits(embeddings, num_classes)
            beta = float(self.args.get("fscil_ridge_ncm_alpha", 0.70))
            return beta * ridge_logits + (1.0 - beta) * ncm_logits

        if self.classifier_mode == "ridge_text" and self.ridge_head is not None:
            ridge_logits = self.ridge_head.classify(embeddings, num_classes)
            text_logits = self._text_logits(embeddings, num_classes)
            gamma = self.text_fusion_gamma
            return (1.0 - gamma) * ridge_logits + gamma * text_logits

        cache_logits = self.cache.classify(embeddings, num_classes)
        if self.classifier_mode == "ridge_cache" and self.ridge_head is not None:
            ridge_logits = self.ridge_head.classify(embeddings, num_classes)
            alpha = self._effective_ridge_alpha()
            return alpha * ridge_logits + (1.0 - alpha) * cache_logits

        if self.classifier_mode == "ridge_cache_text" and self.ridge_head is not None:
            ridge_logits = self.ridge_head.classify(embeddings, num_classes)
            text_logits = self._text_logits(embeddings, num_classes)
            alpha = self._effective_ridge_alpha()
            gamma = min(self.text_fusion_gamma, max(0.0, 1.0 - alpha))
            cache_w = max(0.0, 1.0 - alpha - gamma)
            return alpha * ridge_logits + cache_w * cache_logits + gamma * text_logits

        return cache_logits

    def _build_test_confusion_matrix(self, test_loader) -> np.ndarray:
        """Confusion matrix on the held-out test split (row-normalized recalls)."""
        y_pred, y_true = self._eval_cnn(test_loader)
        preds = torch.tensor(y_pred[:, 0], dtype=torch.long)
        labels = torch.tensor(y_true, dtype=torch.long)
        return build_confusion_matrix(labels, preds, self._total_classes)

    def _class_embeddings_from_train(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_lo: int,
        class_hi: int,
    ) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        labels_cpu = labels.detach().cpu().long()
        emb_cpu = embeddings.detach().cpu().float()
        for cls in range(class_lo, class_hi):
            mask = labels_cpu == cls
            if mask.any():
                out[cls] = emb_cpu[mask]
        return out

    def _build_adaptation_batch(
        self, new_embeddings: torch.Tensor, new_labels: torch.Tensor, old_class_embeddings: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build train-side adaptation set (new train + old cache/train-source signals)."""
        emb_parts = [new_embeddings]
        label_parts = [new_labels]
        for cls in sorted(old_class_embeddings):
            cls_emb = old_class_embeddings[cls]
            if cls_emb is None or cls_emb.numel() == 0:
                continue
            cls_emb = cls_emb.detach().cpu().float()
            cls_labels = torch.full((cls_emb.shape[0],), int(cls), dtype=torch.long)
            emb_parts.append(cls_emb)
            label_parts.append(cls_labels)
        return torch.cat(emb_parts, dim=0), torch.cat(label_parts, dim=0)

    def _run_adapter_warmup_if_needed(self, train_loader, test_loader) -> None:
        """SGD on adapter+fc using the task's labeled train split."""
        run_task0 = self.task0_adapter_warmup and self._cur_task == 0
        run_increment = self._increment_adapter_warmup_enabled()
        if not (run_task0 or run_increment):
            return
        if (
            run_increment
            and bool(self.args.get("fscil_classifier_warmup_after_protos", False))
            and not bool(self.args.get("fscil_increment_warmup_pre_and_post", False))
        ):
            return
        if not hasattr(self._network.backbone, "adaptmlp"):
            logging.warning(
                "Adapter warmup skipped task=%d: backbone has no adaptmlp",
                self._cur_task,
            )
            return
        if run_increment:
            self._restore_task0_adapter_anchor_if_needed()
        epochs = int(
            self._increment_warmup_epochs()
            if self._cur_task > 0
            else self.args.get("tuned_epoch", 5)
        )
        self._set_adapter_warmup_trainable(train_fc=True)
        self._init_adapter_train(train_loader, test_loader, tuned_epoch=epochs)
        self._freeze_backbone()

    def _run_classifier_warmup_after_protos(
        self, train_loader, test_loader
    ) -> None:
        """Few-shot adapter SGD on ridge/cache logits after prototypes exist."""
        if not (
            self.fscil_shot_protocol
            and self._increment_adapter_warmup_enabled()
            and bool(self.args.get("fscil_classifier_warmup_after_protos", False))
        ):
            return
        if self._is_last_fscil_task() and bool(
            self.args.get("fscil_last_task_skip_clf_warmup", False)
        ):
            logging.info(
                "Skipping classifier warmup on last FSCIL task=%d", self._cur_task
            )
            return
        if not hasattr(self._network.backbone, "adaptmlp"):
            logging.warning(
                "Classifier warmup skipped task=%d: backbone has no adaptmlp",
                self._cur_task,
            )
            return
        self._restore_task0_adapter_anchor_if_needed()
        epochs = self._increment_warmup_epochs()
        self._set_adapter_warmup_trainable(train_fc=False)
        self._init_classifier_warmup_train(train_loader, test_loader, tuned_epoch=epochs)
        self._freeze_backbone()

    def _fscil_prototype_settings(self) -> tuple[str, int]:
        """Return (insert_strategy, max_prototypes_per_class) for this task."""
        strategy = self.cache.insert_strategy
        max_proto = self.cache.max_prototypes_per_class
        if not self.fscil_shot_protocol or self._cur_task <= 0:
            return strategy, max_proto
        if bool(self.args.get("fscil_shot_prototypes", False)):
            strategy = "per_sample"
        elif bool(self.args.get("fscil_mean_close_far_prototypes", False)):
            strategy = "mean_close_far"
        elif bool(self.args.get("fscil_mean_far2_prototypes", False)):
            strategy = "mean_far2"
        elif bool(self.args.get("fscil_mean_nearmean_prototypes", False)):
            strategy = "mean_nearmean"
        elif bool(self.args.get("fscil_mean_best2_prototypes", False)):
            strategy = "mean_best2"
        elif bool(self.args.get("fscil_cluster_new_classes", True)):
            strategy = "cluster"
        incr_cap = int(self.args.get("increment_train_per_class", 5))
        max_proto = int(self.args.get("fscil_increment_max_prototypes", incr_cap))
        return strategy, max_proto

    def _insert_task_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_ids: np.ndarray,
        *,
        replace: bool = False,
    ) -> int:
        cache_insert_strategy = self.cache.insert_strategy
        cache_max_proto = self.cache.max_prototypes_per_class
        strategy, max_proto = self._fscil_prototype_settings()
        self.cache.insert_strategy = strategy
        self.cache.max_prototypes_per_class = max_proto
        try:
            if replace:
                inserted = self.cache.replace_class_prototypes(
                    embeddings, labels, class_ids
                )
            else:
                inserted = self.cache.insert_prototypes(embeddings, labels, class_ids)
        finally:
            self.cache.insert_strategy = cache_insert_strategy
            self.cache.max_prototypes_per_class = cache_max_proto
        return inserted

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        self._run_adapter_warmup_if_needed(train_loader, test_loader)
        self._network.eval()

        embeddings, labels = self._collect_embeddings(train_loader_for_protonet)
        class_ids = np.unique(labels.numpy())

        if self.classifier_mode == "ridge_ncm":
            self._update_ncm_prototypes(
                embeddings,
                labels,
                class_ids,
                freeze_base=bool(self.args.get("fscil_freeze_base_means", True)),
            )

        if self._uses_cache_inference():
            inserted = self._insert_task_prototypes(embeddings, labels, class_ids)
            logging.info(
                "KeyCache inserted %d prototypes for task %d", inserted, self._cur_task
            )
        else:
            logging.info(
                "Cache disabled for inference (classifier_mode=%s) task=%d",
                self.classifier_mode,
                self._cur_task,
            )

        if self.ridge_head is not None:
            self._refresh_ridge_from_exemplar_buffer()
            ridge_weight = 1.0
            ridge_embeddings, ridge_labels = embeddings, labels
            if self.fscil_shot_protocol and self._cur_task > 0:
                ridge_weight = float(self.args.get("fscil_novel_ridge_weight", 2.0))
                ridge_embeddings, ridge_labels = self._augment_few_shot_for_ridge(
                    embeddings, labels
                )
            self.ridge_head.update(
                ridge_embeddings,
                ridge_labels,
                weight=ridge_weight,
                fit_ridge=(self._cur_task == 0 and self.ridge_head.ridge is None),
            )

        self._run_classifier_warmup_after_protos(train_loader, test_loader)
        if (
            self.fscil_shot_protocol
            and self._cur_task > 0
            and bool(self.args.get("fscil_classifier_warmup_after_protos", False))
        ):
            self._network.eval()
            embeddings, labels = self._collect_embeddings(train_loader_for_protonet)
            if self.classifier_mode == "ridge_ncm":
                novel_ids = np.arange(self._known_classes, self._total_classes)
                self._update_ncm_prototypes(
                    embeddings,
                    labels,
                    novel_ids,
                    freeze_base=bool(self.args.get("fscil_freeze_base_means", True)),
                )
            if self._uses_cache_inference():
                novel_ids = np.arange(self._known_classes, self._total_classes)
                refreshed = self._insert_task_prototypes(
                    embeddings, labels, novel_ids, replace=True
                )
                logging.info(
                    "KeyCache refreshed %d few-shot prototypes after clf-warmup task=%d",
                    refreshed,
                    self._cur_task,
                )

        bs = self.args.get("batch_size", 128)
        old_class_embeddings = self._collect_old_class_embeddings(self.data_manager, bs)

        adapt_embeddings, adapt_labels = self._build_adaptation_batch(
            embeddings, labels, old_class_embeddings
        )
        adapt_embeddings = adapt_embeddings.to(self._device)
        adapt_labels = adapt_labels.to(self._device)
        adapt_logits = self._classify(adapt_embeddings, self._total_classes)
        adapt_preds = adapt_logits.argmax(dim=1)

        cm = build_confusion_matrix(adapt_labels, adapt_preds, self._total_classes)
        old_recall = (
            float(np.diag(cm[: self._known_classes, : self._known_classes]).mean())
            if self._known_classes > 0
            else 1.0
        )
        logging.info(
            "Task %d confusion peak=%.3f old_recall=%.3f classifier=%s",
            self._cur_task,
            float(np.max(cm - np.diag(np.diag(cm)))),
            old_recall,
            self.classifier_mode,
        )

        if self._uses_cache_inference():
            if self._is_last_fscil_task() and bool(
                self.args.get("fscil_last_task_skip_evolve", False)
            ):
                logging.info(
                    "Skipping cache evolve on last FSCIL task=%d", self._cur_task
                )
            else:
                self.cache.evolve(
                    self._cur_task,
                    adapt_embeddings,
                    adapt_labels,
                    adapt_preds,
                    self._total_classes,
                    known_classes=self._known_classes,
                    confusion_matrix=cm,
                    old_class_embeddings=old_class_embeddings,
                )

        if self.conflict_replay and self.ridge_head is not None:
            if self._is_last_fscil_task() and bool(
                self.args.get("fscil_last_task_skip_conflict_replay", False)
            ):
                logging.info(
                    "Skipping conflict replay on last FSCIL task=%d", self._cur_task
                )
            else:
                replay_cm = cm
                if self.conflict_replay_on_test:
                    replay_cm = self._build_test_confusion_matrix(test_loader)
                    logging.info(
                        "Test-aware conflict replay task=%d (classes=%d)",
                        self._cur_task,
                        self._total_classes,
                    )

                cr = self._modules.conflict_replay
                old_replay_max = cr.max_old_classes
                novel_replay_max = cr.max_novel_classes
                if self._is_last_fscil_task():
                    old_replay_max = int(
                        self.args.get(
                            "fscil_last_task_replay_max_classes", old_replay_max
                        )
                    )
                    novel_replay_max = int(
                        self.args.get(
                            "fscil_last_task_novel_replay_max_classes",
                            novel_replay_max,
                        )
                    )

                n_old = 0
                if self._known_classes > 0 and old_replay_max > 0:
                    n_old = self.ridge_head.replay_classes(
                        old_class_embeddings,
                        replay_cm,
                        recall_threshold=self.replay_recall_threshold,
                        replay_weight=self.replay_weight,
                        max_classes=old_replay_max,
                        class_range=(0, self._known_classes),
                    )

                n_novel = 0
                if self._known_classes < self._total_classes and novel_replay_max > 0:
                    novel_embeddings = self._class_embeddings_from_train(
                        embeddings,
                        labels,
                        self._known_classes,
                        self._total_classes,
                    )
                    novel_weight = float(self.novel_replay_weight)
                    novel_thresh = float(self.args.get("novel_replay_threshold", 0.70))
                    novel_max = novel_replay_max
                    if self.fscil_shot_protocol and self._cur_task == 1:
                        novel_weight = float(
                            self.args.get(
                                "fscil_task1_novel_replay_weight", novel_weight
                            )
                        )
                        novel_thresh = float(
                            self.args.get(
                                "fscil_task1_novel_replay_threshold", novel_thresh
                            )
                        )
                        novel_max = int(
                            self.args.get(
                                "fscil_task1_novel_replay_max_classes", novel_max
                            )
                        )
                    n_novel = self.ridge_head.replay_classes(
                        novel_embeddings,
                        replay_cm,
                        recall_threshold=novel_thresh,
                        replay_weight=novel_weight,
                        max_classes=novel_max,
                        class_range=(self._known_classes, self._total_classes),
                    )
                blend_alpha = (
                    self._effective_ridge_alpha()
                    if self.classifier_mode == "ridge_cache"
                    else float(self.args.get("fscil_ridge_ncm_alpha", 1.0))
                    if self.classifier_mode == "ridge_ncm"
                    else 1.0
                )
                logging.info(
                    "Conflict replay task=%d old=%d novel=%d total_replays=%d alpha=%.3f",
                    self._cur_task,
                    n_old,
                    n_novel,
                    self.ridge_head.replay_updates,
                    blend_alpha,
                )

        if self._uses_cache_inference():
            self.cache.update_margins(adapt_embeddings, adapt_labels, self._total_classes)
            self.cache.sync_fc_weights(self._unwrap_network().fc)

    @torch.no_grad()
    def _collect_embeddings(self, loader):
        self._network.eval()
        embedding_list, label_list = [], []
        for batch in loader:
            data, label = _unpack_batch(batch, self._device)
            emb = self._unwrap_network().backbone(data)
            embedding_list.append(emb.cpu())
            label_list.append(label.cpu())
        return torch.cat(embedding_list, dim=0), torch.cat(label_list, dim=0)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for batch in loader:
            data, label = _unpack_batch(batch, self._device)
            with torch.no_grad():
                emb = self._unwrap_network().backbone(data)
                logits = self._classify(emb, self._total_classes)
            predicts = safe_topk_indices(logits, self.topk)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(label.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _checkpoint_extra(self):
        extra = {"key_cache": self.cache.state_dict()}
        if self.rl_agent is not None:
            extra["rl_agent"] = self.rl_agent.state_dict()
        if self.ridge_head is not None:
            extra["ridge_head"] = self.ridge_head.state_dict()
        if self._ncm_prototypes is not None:
            extra["ncm_prototypes"] = self._ncm_prototypes.cpu()
        return extra

    def _load_checkpoint_extra(self, extra):
        if "key_cache" in extra:
            self.cache.load_state_dict(extra["key_cache"])
        if self.rl_agent is not None and "rl_agent" in extra:
            self.rl_agent.load_state_dict(extra["rl_agent"])
        if self.ridge_head is not None and "ridge_head" in extra:
            self.ridge_head.load_state_dict(extra["ridge_head"])
        if "ncm_prototypes" in extra:
            self._ncm_prototypes = extra["ncm_prototypes"]
        elif "class_means" in extra:
            self._ncm_prototypes = extra["class_means"]

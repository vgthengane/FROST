"""Uni3D + SGD linear head for FSCIL Joint / naive Fine-Tune baselines.

Joint (``linprobe_mode=joint``):
  session 0 = full base; session t>0 = full base + 5-shot novels seen so far.

Fine-Tune (``linprobe_mode=finetune``):
  session 0 = full base; session t>0 = current session 5-shot novels + old-class replay.

By default no adapter is attached: the backbone is frozen and only a plain
``SimpleLinear`` head is trained with CE.  ``linprobe_use_adapter=true`` is an
opt-in ablation path: the adapter and linear head train on session 0, then the
adapter is frozen while the expanded linear head continues training.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.fscil_protocol import (
    prepare_fscil_task_datasets,
    prepare_joint_fscil_train_dataset,
)
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

num_workers_default = 4


def _unpack_batch(batch, device):
    if not isinstance(batch, (list, tuple)) or len(batch) < 3:
        raise ValueError(f"Unexpected batch format: {type(batch)}")
    has_type = not torch.is_tensor(batch[-1])
    label = batch[-2] if has_type else batch[-1]
    data = batch[1]
    return data.to(device), label.to(device)


def _resolve_mode(args: dict) -> str:
    mode = str(args.get("linprobe_mode", "")).strip().lower()
    if mode in {"joint", "finetune", "ft"}:
        return "finetune" if mode == "ft" else mode
    name = str(args.get("model_name", "")).lower()
    if "joint" in name:
        return "joint"
    return "finetune"


class Learner(BaseLearner):
    """Uni3D linear probe with an optional base-session adapter."""

    def __init__(self, args):
        super().__init__(args)
        self.args = dict(args)
        self.use_adapter = bool(self.args.get("linprobe_use_adapter", False))
        bt = str(self.args.get("backbone_type", "uni3d_base_patch14_448"))
        if self.use_adapter and not bt.endswith("_adapter"):
            bt = f"{bt}_adapter"
            self.args.setdefault("ffn_num", 64)
        elif not self.use_adapter and bt.endswith("_adapter"):
            bt = bt[: -len("_adapter")]
            logging.info("fscil_linprobe: stripping adapter → backbone_type=%s", bt)
        self.args["backbone_type"] = bt or "uni3d_base_patch14_448"

        self.linprobe_mode = _resolve_mode(self.args)
        self._network = IncrementalNet(self.args, True)
        self._configure_trainable_parameters()
        self.batch_size = int(self.args.get("batch_size", 8))
        self.num_workers = int(self.args.get("num_workers", num_workers_default))
        self.init_lr = float(self.args.get("init_lr", 0.01))
        self.weight_decay = float(self.args.get("weight_decay", 5e-4))
        self.min_lr = float(self.args.get("min_lr", 0.0))
        self.optimizer_name = str(self.args.get("optimizer", "sgd")).lower()
        self.base_epoch = int(self.args.get("base_epoch", self.args.get("tuned_epoch", 10)))
        self.increment_epoch = int(
            self.args.get("increment_epoch", self.args.get("increment_tuned_epoch", 5))
        )
        logging.info(
            "fscil_linprobe mode=%s backbone=%s base_epoch=%d increment_epoch=%d "
            "adapter_base_only=%s",
            self.linprobe_mode,
            self.args["backbone_type"],
            self.base_epoch,
            self.increment_epoch,
            self.use_adapter,
        )

    def _configure_trainable_parameters(self) -> None:
        """Freeze Uni3D; optionally train its adapter on session 0; always train FC."""
        for param in self._network.backbone.parameters():
            param.requires_grad = False
        if self.use_adapter and self._cur_task == 0:
            if not hasattr(self._network.backbone, "adaptmlp"):
                raise RuntimeError(
                    "linprobe_use_adapter=true requires a backbone with adaptmlp"
                )
            for param in self._network.backbone.adaptmlp.parameters():
                param.requires_grad = True
        if self._network.fc is not None:
            for param in self._network.fc.parameters():
                param.requires_grad = True

    def _set_train_modes(self) -> None:
        """Keep frozen Uni3D deterministic while allowing the base adapter to train."""
        self._network.train()
        self._network.backbone.eval()
        if self.use_adapter and self._cur_task == 0:
            self._network.backbone.adaptmlp.train()

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._configure_trainable_parameters()

        logging.info(
            "Learning on %d-%d [%s]",
            self._known_classes,
            self._total_classes,
            self.linprobe_mode,
        )

        if self.linprobe_mode == "joint":
            train_dataset = prepare_joint_fscil_train_dataset(
                data_manager,
                total_classes=self._total_classes,
                cur_task=self._cur_task,
                args=self.args,
            )
        else:
            # Naive FT: current-session 5-shot novels + configured old-class replay.
            train_dataset, _ = prepare_fscil_task_datasets(
                data_manager,
                known_classes=self._known_classes,
                total_classes=self._total_classes,
                cur_task=self._cur_task,
                args=self.args,
            )

        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _epochs_for_task(self) -> int:
        if self._cur_task == 0:
            return max(1, self.base_epoch)
        return max(1, self.increment_epoch)

    def _build_optimizer(self, epochs: int):
        params = [p for p in self._network.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in params)
        total = sum(p.numel() for p in self._network.parameters())
        lr = (
            float(self.args.get("task0_init_lr", self.init_lr))
            if self._cur_task == 0
            else self.init_lr
        )
        weight_decay = (
            float(self.args.get("task0_weight_decay", self.weight_decay))
            if self._cur_task == 0
            else self.weight_decay
        )
        optimizer_name = (
            str(self.args.get("task0_optimizer", self.optimizer_name)).lower()
            if self._cur_task == 0
            else self.optimizer_name
        )
        logging.info(
            "Trainable params=%d / total=%d (%s), lr=%.6f wd=%.1e optimizer=%s",
            trainable,
            total,
            "base adapter + FC" if self.use_adapter and self._cur_task == 0 else "FC only",
            lr,
            weight_decay,
            optimizer_name,
        )
        if optimizer_name == "adam":
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                params,
                momentum=0.9,
                lr=lr,
                weight_decay=weight_decay,
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.min_lr
        )
        return optimizer, scheduler

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self._configure_trainable_parameters()
        epochs = self._epochs_for_task()
        optimizer, scheduler = self._build_optimizer(epochs)

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._set_train_modes()
            losses = 0.0
            correct, total = 0, 0
            for batch in train_loader:
                inputs, targets = _unpack_batch(batch, self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += float(loss.item())
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / max(len(train_loader), 1),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for batch in loader:
            inputs, targets = _unpack_batch(batch, self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = outputs.argmax(dim=1)
            correct += (predicts.cpu() == targets.cpu()).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        from utils.toolkit import safe_topk_indices

        for batch in loader:
            inputs, targets = _unpack_batch(batch, self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = safe_topk_indices(outputs, self.topk)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

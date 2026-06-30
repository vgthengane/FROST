"""Uni3D + MOS (momentum-based adapter merging) for 3D continual learning."""

from __future__ import annotations

import copy
from typing import Optional

import timm
import torch
import torch.nn as nn
from easydict import EasyDict

from .uni3d_l2p import Group, Encoder, PatchDropout, _load_uni3d_weights
from .uni3d_adapter import AdaptFormerAdapter


class Uni3DMOSBackbone(nn.Module):
    def __init__(
        self,
        point_transformer: nn.Module,
        *,
        num_classes: int,
        trans_dim: int = 768,
        num_group: int = 512,
        group_size: int = 64,
        encoder_dim: int = 512,
        patch_dropout: float = 0.0,
        tuning_config: Optional[EasyDict] = None,
    ):
        super().__init__()
        self.out_dim = trans_dim
        self.embed_dim = trans_dim
        self.num_classes = num_classes

        # Uni3D tokenization
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dim)
        self.encoder2trans = nn.Linear(encoder_dim, trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        self.visual = point_transformer
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        # MOS adapter config
        tuning_config = tuning_config or EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=16,
            d_model=trans_dim,
            adapter_momentum=0.0,
        )
        self.config = tuning_config

        # Classifier head inside backbone (used when fc_only=True in MOS learner)
        self.head = nn.Linear(trans_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Adapter state
        self.adapter_list = nn.ModuleList()  # list of ModuleList (one per task snapshot)
        self.cur_adapter = nn.ModuleList()

        depth = len(self.visual.blocks)
        self.down_weight_sum = [[] for _ in range(depth)]
        self.down_bias_sum = [[] for _ in range(depth)]
        self.up_weight_sum = [[] for _ in range(depth)]
        self.up_bias_sum = [[] for _ in range(depth)]

        self.init_adapters()

    def init_adapters(self):
        self.cur_adapter = nn.ModuleList()
        depth = len(self.visual.blocks)
        device = self.cls_token.device
        for _ in range(depth):
            adapter = AdaptFormerAdapter(
                d_model=int(self.config.d_model),
                bottleneck=int(self.config.ffn_num),
                dropout=0.1,
                init_option=self.config.ffn_adapter_init_option,
                adapter_scalar=self.config.ffn_adapter_scalar,
                adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
            ).to(device)
            self.cur_adapter.append(adapter)
        self.cur_adapter.requires_grad_(True)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        # train only current adapters and head
        for p in self.cur_adapter.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def sum_adapter_param(self):
        depth = len(self.visual.blocks)
        for layer_idx in range(depth):
            dw = self.cur_adapter[layer_idx].down_proj.weight.data.detach()
            db = self.cur_adapter[layer_idx].down_proj.bias.data.detach()
            uw = self.cur_adapter[layer_idx].up_proj.weight.data.detach()
            ub = self.cur_adapter[layer_idx].up_proj.bias.data.detach()

            if len(self.down_weight_sum[layer_idx]) > 0:
                self.down_weight_sum[layer_idx].append(dw + self.down_weight_sum[layer_idx][-1])
                self.down_bias_sum[layer_idx].append(db + self.down_bias_sum[layer_idx][-1])
                self.up_weight_sum[layer_idx].append(uw + self.up_weight_sum[layer_idx][-1])
                self.up_bias_sum[layer_idx].append(ub + self.up_bias_sum[layer_idx][-1])
            else:
                self.down_weight_sum[layer_idx].append(dw)
                self.down_bias_sum[layer_idx].append(db)
                self.up_weight_sum[layer_idx].append(uw)
                self.up_bias_sum[layer_idx].append(ub)

    def adapter_update(self):
        # snapshot current adapter for this task
        self.adapter_list.append(copy.deepcopy(self.cur_adapter))
        self.sum_adapter_param()

    def reweight_adapter(self, adapter: nn.ModuleList, idx: int) -> nn.ModuleList:
        momentum = float(self.config.adapter_momentum)
        if momentum == 0.0 or idx == 0:
            return adapter

        depth = len(self.visual.blocks)
        for layer_idx in range(depth):
            adapter[layer_idx].down_proj.weight.data = (1 - momentum) * adapter[layer_idx].down_proj.weight.data + momentum * (
                self.down_weight_sum[layer_idx][idx - 1] / idx
            )
            adapter[layer_idx].down_proj.bias.data = (1 - momentum) * adapter[layer_idx].down_proj.bias.data + momentum * (
                self.down_bias_sum[layer_idx][idx - 1] / idx
            )
            adapter[layer_idx].up_proj.weight.data = (1 - momentum) * adapter[layer_idx].up_proj.weight.data + momentum * (
                self.up_weight_sum[layer_idx][idx - 1] / idx
            )
            adapter[layer_idx].up_proj.bias.data = (1 - momentum) * adapter[layer_idx].up_proj.bias.data + momentum * (
                self.up_bias_sum[layer_idx][idx - 1] / idx
            )
        return adapter

    def adapter_merge(self):
        # EMA merge current adapter towards historical mean (called during training)
        if len(self.adapter_list) == 0 or float(self.config.adapter_momentum) == 0.0:
            return
        self.cur_adapter = self.reweight_adapter(self.cur_adapter, len(self.adapter_list))

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        pts = x[:, :, :3].contiguous()
        colors = x[:, :, 3:].contiguous()
        _, center, features = self.group_divider(pts, colors)

        group_tokens = self.encoder(features)
        group_tokens = self.encoder2trans(group_tokens)

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, group_tokens), dim=1)

        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        group_pos = self.pos_embed(center)
        pos = torch.cat((cls_pos, group_pos), dim=1)
        tokens = tokens + pos

        tokens = self.patch_dropout(tokens)
        tokens = self.visual.pos_drop(tokens)
        return tokens

    def _forward_blocks(self, tokens: torch.Tensor, adapt: Optional[nn.ModuleList]) -> torch.Tensor:
        if adapt is not None:
            assert len(adapt) == len(self.visual.blocks)

        for i, blk in enumerate(self.visual.blocks):
            # attention
            if getattr(blk, "gamma_1", None) is None:
                attn_out = blk.attn(blk.norm1(tokens))
            else:
                attn_out = blk.gamma_1 * blk.attn(blk.norm1(tokens))
            tokens = tokens + blk.drop_path1(attn_out)

            adapt_x = None
            if adapt is not None and self.config.ffn_adapt and self.config.ffn_option == "parallel":
                adapt_x = adapt[i](tokens, add_residual=False)

            # FFN
            if getattr(blk, "gamma_2", None) is None:
                ffn_out = blk.mlp(blk.norm2(tokens))
            else:
                ffn_out = blk.gamma_2 * blk.mlp(blk.norm2(tokens))
            ffn_out = blk.drop_path2(ffn_out)

            if adapt is not None and self.config.ffn_adapt:
                if self.config.ffn_option == "sequential":
                    ffn_out = adapt[i](ffn_out)  # adds residual inside
                elif self.config.ffn_option == "parallel":
                    ffn_out = ffn_out + adapt_x
                else:
                    raise ValueError(self.config.ffn_option)

            tokens = tokens + ffn_out

        return tokens

    def _cls_feature(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = self.visual.norm(tokens[:, 0, :])
        if hasattr(self.visual, "fc_norm") and self.visual.fc_norm is not None:
            cls = self.visual.fc_norm(cls)
        return cls

    def forward_features(self, x: torch.Tensor, adapter_id: int, train: bool):
        tokens = self._tokenize(x)

        if adapter_id == -1:
            tokens = self._forward_blocks(tokens, adapt=None)
        else:
            if train:
                orig_adapter = self.cur_adapter
                momentum_adapter = self.reweight_adapter(orig_adapter, adapter_id)
                tokens = self._forward_blocks(tokens, adapt=momentum_adapter)
            else:
                if adapter_id == len(self.adapter_list):
                    orig_adapter = self.cur_adapter
                    momentum_adapter = self.reweight_adapter(orig_adapter, adapter_id)
                    tokens = self._forward_blocks(tokens, adapt=momentum_adapter)
                elif adapter_id < len(self.adapter_list):
                    orig_adapter = self.adapter_list[adapter_id]
                    momentum_adapter = self.reweight_adapter(orig_adapter, adapter_id)
                    tokens = self._forward_blocks(tokens, adapt=momentum_adapter)
                else:
                    raise ValueError("adapter_id is wrong.")

        feat = self._cls_feature(tokens)
        res = {"x": feat, "pre_logits": feat, "features": feat}
        return res

    def forward_head(self, res: dict) -> dict:
        x = res["x"]
        res["logits"] = self.head(x)
        return res

    def forward(self, x: torch.Tensor, adapter_id: int = -1, train: bool = False, fc_only: bool = False):
        if fc_only:
            # x is already a feature tensor [B, trans_dim]
            return {"logits": self.head(x)}
        res = self.forward_features(x, adapter_id, train)
        res = self.forward_head(res)
        return res


def create_uni3d_mos(
    *,
    pc_model: str = "eva02_base_patch14_448",
    pretrained_pc: Optional[str] = None,
    drop_path_rate: float = 0.0,
    trans_dim: int = 768,
    num_group: int = 512,
    group_size: int = 64,
    encoder_dim: int = 512,
    patch_dropout: float = 0.0,
    num_classes: int = 0,
    tuning_config: Optional[EasyDict] = None,
) -> Uni3DMOSBackbone:
    is_uni3d_pt = pretrained_pc and pretrained_pc.endswith(".pt")
    checkpoint_path = pretrained_pc if (pretrained_pc and not is_uni3d_pt) else ""

    try:
        point_transformer = timm.create_model(
            pc_model,
            checkpoint_path=checkpoint_path if checkpoint_path else None,
            drop_path_rate=drop_path_rate,
        )
    except RuntimeError as e:
        if "Unknown model" in str(e):
            if not checkpoint_path:
                raise RuntimeError(
                    f"Model '{pc_model}' not found in timm. "
                    f"Please provide a checkpoint_path via pretrained_pc, or ensure timm>=0.9.7. "
                    f"Current timm: {timm.__version__}"
                ) from e
            point_transformer = timm.create_model(pc_model, pretrained=False, drop_path_rate=drop_path_rate)
        else:
            raise

    model = Uni3DMOSBackbone(
        point_transformer,
        num_classes=num_classes,
        trans_dim=trans_dim,
        num_group=num_group,
        group_size=group_size,
        encoder_dim=encoder_dim,
        patch_dropout=patch_dropout,
        tuning_config=tuning_config,
    )

    if is_uni3d_pt and pretrained_pc:
        _load_uni3d_weights(model, pretrained_pc)

    return model



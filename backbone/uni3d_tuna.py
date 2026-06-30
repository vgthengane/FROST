"""Uni3D + TUNA for 3D continual learning."""

from __future__ import annotations

import copy
from typing import Optional

import timm
import torch
import torch.nn as nn
from easydict import EasyDict

from .uni3d_l2p import Group, Encoder, PatchDropout, _load_uni3d_weights
from .uni3d_adapter import AdaptFormerAdapter


class TaskVector:
    """
    TaskVector utility used for EMR merge.
    Here we use it to hold adapter state dict tensors (float only).
    """

    def __init__(self, pretrained_state_dict=None, vector=None):
        if vector is not None:
            self.vector = vector
        else:
            with torch.no_grad():
                self.vector = {}
                for key, val in pretrained_state_dict.items():
                    if getattr(val, "dtype", None) in (torch.int64, torch.uint8):
                        continue
                    self.vector[key] = val


def emr_merge(task_vectors: list[TaskVector]):
    """
    EMR merge over a list of adapter snapshots.
    Returns:
      vector_unified: dict[name -> merged tensor]
      masks: dict[name -> list[bool mask per task]]
      rescalers: tensor[len(task_vectors)]
    """
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}

    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n] > 0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask * param)
            param_max = torch.where(param_abs > param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] = param_max * flag

    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers


class Uni3DTUNABackbone(nn.Module):
    def __init__(
        self,
        point_transformer: nn.Module,
        *,
        trans_dim: int = 768,
        num_group: int = 512,
        group_size: int = 64,
        encoder_dim: int = 512,
        patch_dropout: float = 0.0,
        tuning_config: Optional[EasyDict] = None,
    ):
        super().__init__()
        self.out_dim = trans_dim

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

        tuning_config = tuning_config or EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=16,
            d_model=trans_dim,
        )
        self.config = tuning_config

        # Adapter state
        self.adapter_list = nn.ModuleList()  # historical adapters (ModuleList per task)
        self.cur_adapter = nn.ModuleList()
        self.merged_adapter: Optional[nn.ModuleList] = None

        self.init_adapters()

    def init_adapters(self):
        self.cur_adapter = nn.ModuleList()
        device = self.cls_token.device
        for _ in range(len(self.visual.blocks)):
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
        for p in self.cur_adapter.parameters():
            p.requires_grad = True

    def adapter_update(self):
        self.adapter_list.append(copy.deepcopy(self.cur_adapter))

    def merge(self):
        # Build a unified adapter by EMR merge over historical adapters.
        if len(self.adapter_list) == 0:
            self.merged_adapter = copy.deepcopy(self.cur_adapter)
            return

        merged_adapter = copy.deepcopy(self.cur_adapter).cpu()
        task_vectors = [
            TaskVector(pretrained_state_dict=self.adapter_list[i].cpu().state_dict())
            for i in range(len(self.adapter_list))
        ]
        vector_unified, _masks, _rescalers = emr_merge(task_vectors)
        merged_adapter.load_state_dict({k: v for k, v in vector_unified.items()}, strict=False)

        self.merged_adapter = merged_adapter.to(self.cls_token.device)
        # keep historical adapters on current device too
        for i in range(len(self.adapter_list)):
            self.adapter_list[i] = self.adapter_list[i].to(self.cls_token.device)

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
        tokens = tokens + torch.cat((cls_pos, group_pos), dim=1)

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
                    ffn_out = adapt[i](ffn_out)
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
                tokens = self._forward_blocks(tokens, adapt=self.cur_adapter)
            else:
                if adapter_id == len(self.adapter_list):
                    tokens = self._forward_blocks(tokens, adapt=self.cur_adapter)
                elif adapter_id < len(self.adapter_list):
                    tokens = self._forward_blocks(tokens, adapt=self.adapter_list[adapter_id])
                else:
                    if self.merged_adapter is None:
                        raise RuntimeError("merged_adapter is None; call merge() after adapter_update().")
                    tokens = self._forward_blocks(tokens, adapt=self.merged_adapter)

        feat = self._cls_feature(tokens)
        return {"x": feat, "pre_logits": feat, "features": feat}

    def forward(self, x: torch.Tensor, adapter_id: int = -1, train: bool = False, fc_only: bool = False):
        # TUNA learner uses external `TunaLinear`, so backbone logits are not required.
        if fc_only:
            return {"logits": x}
        return self.forward_features(x, adapter_id, train)


def create_uni3d_tuna(
    *,
    pc_model: str = "eva02_base_patch14_448",
    pretrained_pc: Optional[str] = None,
    drop_path_rate: float = 0.0,
    trans_dim: int = 768,
    num_group: int = 512,
    group_size: int = 64,
    encoder_dim: int = 512,
    patch_dropout: float = 0.0,
    tuning_config: Optional[EasyDict] = None,
) -> Uni3DTUNABackbone:
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

    model = Uni3DTUNABackbone(
        point_transformer,
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



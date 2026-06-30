"""Uni3D + EASE for 3D continual learning."""

from __future__ import annotations

from typing import Optional

import copy

import timm
import torch
import torch.nn as nn
from easydict import EasyDict

from .uni3d_l2p import Group, Encoder, PatchDropout, _load_uni3d_weights
from .uni3d_adapter import AdaptFormerAdapter


class Uni3DEaseBackbone(nn.Module):
    """Uni3D backbone with per-task adapters for EASE-style continual learning."""

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
        self.trans_dim = trans_dim
        self.out_dim = trans_dim  # EASE uses 768-dim features per adapter

        # Uni3D tokenization
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dim)
        self.encoder2trans = nn.Linear(encoder_dim, trans_dim)

        # CLS + geometric positional embedding (same as other Uni3D backbones in this repo)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        self.visual = point_transformer
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        # Adapter config (match `inc_net.py` EASE defaults)
        tuning_config = tuning_config or EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=64,
            d_model=trans_dim,
            _device=torch.device("cpu"),
        )
        self.config = tuning_config

        # Multi-adapter state
        self.adapter_list: list[nn.ModuleList] = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 6] -> tokens: [B, 1+G, trans_dim]"""
        pts = x[:, :, :3].contiguous()
        colors = x[:, :, 3:].contiguous()
        _, center, features = self.group_divider(pts, colors)

        group_tokens = self.encoder(features)  # [B,G,encoder_dim]
        group_tokens = self.encoder2trans(group_tokens)  # [B,G,trans_dim]

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
            assert len(adapt) == len(self.visual.blocks), "adapter length must match transformer depth"

        for i, blk in enumerate(self.visual.blocks):
            if getattr(blk, "gamma_1", None) is None:
                attn_out = blk.attn(blk.norm1(tokens))
            else:
                attn_out = blk.gamma_1 * blk.attn(blk.norm1(tokens))
            tokens = tokens + blk.drop_path1(attn_out)

            adapt_x = None
            if adapt is not None and self.config.ffn_adapt and self.config.ffn_option == "parallel":
                adapt_x = adapt[i](tokens, add_residual=False)

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

    def _norm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # EVA02 in timm usually has `norm` and `fc_norm`.
        tokens = self.visual.norm(tokens)
        if hasattr(self.visual, "fc_norm") and self.visual.fc_norm is not None:
            # fc_norm expects [B, C], so apply on CLS only if needed by caller.
            return tokens
        return tokens

    def _cls_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.visual.norm(tokens)
        cls = tokens[:, 0, :]
        if hasattr(self.visual, "fc_norm") and self.visual.fc_norm is not None:
            cls = self.visual.fc_norm(cls)
        return cls

    def get_new_adapter(self):
        self.cur_adapter = nn.ModuleList()
        if not self.config.ffn_adapt:
            return

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

    def add_adapter_to_list(self):
        # freeze and save current adapter, then create a fresh adapter for next task
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.cur_adapter.parameters():
            p.requires_grad = True

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokenize(x)
        tokens = self._forward_blocks(tokens, self.cur_adapter)
        return self._cls_from_tokens(tokens)

    def forward_test(self, x: torch.Tensor, use_init_ptm: bool = False) -> list[torch.Tensor]:
        tokens_init = self._tokenize(x)
        features: list[torch.Tensor] = []

        if use_init_ptm:
            t = copy.deepcopy(tokens_init)
            t = self._forward_blocks(t, adapt=None)
            features.append(self._cls_from_tokens(t))

        for adapt in self.adapter_list:
            t = copy.deepcopy(tokens_init)
            t = self._forward_blocks(t, adapt=adapt)
            features.append(self._cls_from_tokens(t))

        t = copy.deepcopy(tokens_init)
        t = self._forward_blocks(t, adapt=self.cur_adapter)
        features.append(self._cls_from_tokens(t))

        return features

    def forward(self, x: torch.Tensor, test: bool = False, use_init_ptm: bool = False) -> torch.Tensor:
        if not test:
            return self.forward_train(x)

        feats = self.forward_test(x, use_init_ptm=use_init_ptm)
        return torch.cat(feats, dim=1)

    def forward_proto(self, x: torch.Tensor, adapt_index: int) -> torch.Tensor:
        """Return a single [B, out_dim] vector for a specific adapter index.

        - ``adapt_index == -1``: no adapter
        - ``0 <= adapt_index < len(adapter_list)``: historical adapter
        - otherwise: current adapter
        """
        tokens = self._tokenize(x)
        if adapt_index == -1:
            tokens = self._forward_blocks(tokens, adapt=None)
            return self._cls_from_tokens(tokens)

        if adapt_index < len(self.adapter_list):
            adapt = self.adapter_list[adapt_index]
        else:
            adapt = self.cur_adapter

        tokens = self._forward_blocks(tokens, adapt=adapt)
        return self._cls_from_tokens(tokens)


def create_uni3d_ease(
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
) -> Uni3DEaseBackbone:
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
            point_transformer = timm.create_model(
                pc_model,
                pretrained=False,
                drop_path_rate=drop_path_rate,
            )
        else:
            raise

    model = Uni3DEaseBackbone(
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

    model.freeze()
    return model



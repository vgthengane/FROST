"""Uni3D + AdaptFormer adapter for parameter-efficient 3D continual learning."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from easydict import EasyDict

from .uni3d_l2p import Group, Encoder, PatchDropout, _load_uni3d_weights


class AdaptFormerAdapter(nn.Module):
    """AdaptFormer adapter with LoRA init, optional layer-norm in/out, and parallel mode support."""

    def __init__(
        self,
        d_model: int,
        bottleneck: int,
        dropout: float = 0.1,
        init_option: str = "lora",
        adapter_scalar: str | float = "0.1",
        adapter_layernorm_option: str = "none",
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        if init_option != "lora":
            raise NotImplementedError("Only 'lora' init is supported.")

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, add_residual: bool = True, residual: torch.Tensor | None = None) -> torch.Tensor:
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == "in" and self.adapter_layer_norm_before is not None:
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == "out" and self.adapter_layer_norm_before is not None:
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            return up + residual
        return up


class Uni3DAdapter(nn.Module):
    def __init__(
        self,
        point_transformer: nn.Module,
        *,
        trans_dim: int = 768,
        embed_dim: int = 1024,
        num_group: int = 512,
        group_size: int = 64,
        encoder_dim: int = 512,
        patch_dropout: float = 0.0,
        tuning_config: Optional[EasyDict] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.trans_dim = trans_dim
        self.embed_dim = embed_dim
        self.out_dim = embed_dim  # required by BaseNet/SimpleVitNet

        # Uni3D tokenization
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dim)
        self.encoder2trans = nn.Linear(encoder_dim, trans_dim)

        # CLS + geometric positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        self.visual = point_transformer
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        # Adapter config
        tuning_config = tuning_config or EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_num=64,
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_adapter_layernorm_option="none",
        )
        self.tuning_config = tuning_config

        # One AdaptFormer adapter per transformer block (inserted inside FFN)
        self.adaptmlp = nn.ModuleList(
            [
                AdaptFormerAdapter(
                    d_model=trans_dim,
                    bottleneck=int(tuning_config.ffn_num),
                    dropout=0.1,
                    init_option=tuning_config.ffn_adapter_init_option,
                    adapter_scalar=tuning_config.ffn_adapter_scalar,
                    adapter_layernorm_option=tuning_config.ffn_adapter_layernorm_option,
                )
                for _ in range(len(self.visual.blocks))
            ]
        )

        # Final projection to embed_dim
        self.trans2embed = nn.Linear(trans_dim, embed_dim)

        if freeze_backbone:
            for n, p in self.named_parameters():
                p.requires_grad = False
            # Enable adapters only (and optionally projection/head if desired)
            for n, p in self.adaptmlp.named_parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pts = x[:, :, :3].contiguous()
        colors = x[:, :, 3:].contiguous()
        _, center, features = self.group_divider(pts, colors)

        group_tokens = self.encoder(features)           # [B,G,encoder_dim]
        group_tokens = self.encoder2trans(group_tokens) # [B,G,trans_dim]

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, group_tokens), dim=1)

        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        group_pos = self.pos_embed(center)
        pos = torch.cat((cls_pos, group_pos), dim=1)
        tokens = tokens + pos

        tokens = self.patch_dropout(tokens)
        tokens = self.visual.pos_drop(tokens)

        for i, blk in enumerate(self.visual.blocks):
            if getattr(blk, "gamma_1", None) is None:
                attn_out = blk.attn(blk.norm1(tokens))
            else:
                attn_out = blk.gamma_1 * blk.attn(blk.norm1(tokens))
            tokens = tokens + blk.drop_path1(attn_out)

            adapt_x = None
            if self.tuning_config.ffn_adapt and self.tuning_config.ffn_option == "parallel":
                adapt_x = self.adaptmlp[i](tokens, add_residual=False)

            if getattr(blk, "gamma_2", None) is None:
                ffn_out = blk.mlp(blk.norm2(tokens))
            else:
                ffn_out = blk.gamma_2 * blk.mlp(blk.norm2(tokens))
            ffn_out = blk.drop_path2(ffn_out)

            if self.tuning_config.ffn_adapt:
                if self.tuning_config.ffn_option == "sequential":
                    ffn_out = self.adaptmlp[i](ffn_out)
                elif self.tuning_config.ffn_option == "parallel":
                    ffn_out = ffn_out + adapt_x
                else:
                    raise ValueError(self.tuning_config.ffn_option)

            tokens = tokens + ffn_out

        x_cls = self.visual.norm(tokens[:, 0, :])
        x_cls = self.visual.fc_norm(x_cls)
        return self.trans2embed(x_cls)


def create_uni3d_adapter(
    *,
    pc_model: str = "eva02_base_patch14_448",
    pretrained_pc: Optional[str] = None,
    drop_path_rate: float = 0.0,
    trans_dim: int = 768,
    embed_dim: int = 1024,
    num_group: int = 512,
    group_size: int = 64,
    encoder_dim: int = 512,
    patch_dropout: float = 0.0,
    tuning_config: Optional[EasyDict] = None,
):
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
                checkpoint_path=checkpoint_path,
                drop_path_rate=drop_path_rate,
                pretrained=False,
            )
        else:
            raise

    model = Uni3DAdapter(
        point_transformer=point_transformer,
        trans_dim=trans_dim,
        embed_dim=embed_dim,
        num_group=num_group,
        group_size=group_size,
        encoder_dim=encoder_dim,
        patch_dropout=patch_dropout,
        tuning_config=tuning_config,
        freeze_backbone=True,
    )

    if is_uni3d_pt:
        _load_uni3d_weights(model, pretrained_pc)

    return model



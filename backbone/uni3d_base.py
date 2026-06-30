"""Uni3D base backbone (no prompt) for 3D point clouds."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from .uni3d_l2p import Group, Encoder, PatchDropout, _load_uni3d_weights


class Uni3DBase(nn.Module):
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
    ):
        super().__init__()
        self.trans_dim = trans_dim
        self.embed_dim = embed_dim
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_dim = encoder_dim

        # Match BaseNet expectations
        self.out_dim = embed_dim

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

        # Transformer backbone (timm EVA02 / ViT-like)
        self.visual = point_transformer
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        # Final embedding projection
        self.trans2embed = nn.Linear(trans_dim, embed_dim)

    def forward(self, x: torch.Tensor, layer_feat: bool = False):
        """
        Args:
            x: [B, N, 6] float tensor
            layer_feat: if True, return per-block token features for SLCA
        Returns:
            features: [B, embed_dim] or {"features": list of [B, T, trans_dim]}
        """
        pts = x[:, :, :3].contiguous()
        colors = x[:, :, 3:].contiguous()
        _, center, features = self.group_divider(pts, colors)
        # FPS/KNN 为异步 CUDA；若此处已失败，同步后报错位置更贴近真因，避免误报在后续 conv1d
        if features.is_cuda:
            torch.cuda.synchronize(features.device)

        group_tokens = self.encoder(features)                # [B,G,encoder_dim]
        group_tokens = self.encoder2trans(group_tokens)      # [B,G,trans_dim]

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, group_tokens), dim=1)  # [B,1+G,trans_dim]

        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        group_pos = self.pos_embed(center)
        pos = torch.cat((cls_pos, group_pos), dim=1)
        tokens = tokens + pos

        tokens = self.patch_dropout(tokens)
        tokens = self.visual.pos_drop(tokens)

        layer_outputs = []
        for blk in self.visual.blocks:
            tokens = blk(tokens)
            if layer_feat:
                layer_outputs.append(tokens)

        x_cls = self.visual.norm(tokens[:, 0, :])
        x_cls = self.visual.fc_norm(x_cls)
        out = self.trans2embed(x_cls)

        if layer_feat:
            return {"features": layer_outputs}
        return out


def create_uni3d_base(
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

    model = Uni3DBase(
        point_transformer=point_transformer,
        trans_dim=trans_dim,
        embed_dim=embed_dim,
        num_group=num_group,
        group_size=group_size,
        encoder_dim=encoder_dim,
        patch_dropout=patch_dropout,
    )

    if is_uni3d_pt:
        _load_uni3d_weights(model, pretrained_pc)

    return model



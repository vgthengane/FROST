"""Uni3D + CODA-Prompt for 3D continual learning."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from .prompt import CodaPrompt
from .uni3d_l2p import Group, Encoder, PatchDropout, _load_uni3d_weights


class Uni3DCodaPromptBackbone(nn.Module):
    """``forward(x, prompt=None, q=None, train=False, task_id=None) -> (tokens, prompt_loss)``."""

    def __init__(
        self,
        point_transformer: nn.Module,
        *,
        trans_dim: int = 768,
        num_group: int = 512,
        group_size: int = 64,
        encoder_dim: int = 512,
        patch_dropout: float = 0.0,
    ):
        super().__init__()
        self.trans_dim = trans_dim
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_dim = encoder_dim

        # Uni3D tokenization
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dim)
        self.encoder2trans = nn.Linear(encoder_dim, trans_dim)

        # CLS + geometric positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        # Transformer from Uni3D (EVA02 / ViT-like)
        self.visual = point_transformer
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

    def _build_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build initial token sequence with geometry positional embedding.
        Input x: [B, N, 6] (xyz+rgb)
        Output: [B, 1+G, C]
        """
        pts = x[:, :, :3].contiguous()
        colors = x[:, :, 3:].contiguous()
        _, center, features = self.group_divider(pts, colors)  # center: [B,G,3]

        group_tokens = self.encoder(features)                # [B,G,encoder_dim]
        group_tokens = self.encoder2trans(group_tokens)      # [B,G,C]

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, group_tokens), dim=1)  # [B,1+G,C]

        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        group_pos = self.pos_embed(center)
        pos = torch.cat((cls_pos, group_pos), dim=1)
        tokens = tokens + pos
        return tokens

    def forward(
        self,
        x: torch.Tensor,
        register_blk: int = -1,
        prompt: Optional[CodaPrompt] = None,
        q: Optional[torch.Tensor] = None,
        train: bool = False,
        task_id: Optional[int] = None,
    ):
        tokens = self._build_tokens(x)
        tokens = self.patch_dropout(tokens)
        tokens = self.visual.pos_drop(tokens)

        prompt_loss = torch.zeros((1,), device=tokens.device)

        for i, blk in enumerate(self.visual.blocks):
            p_list = None
            if prompt is not None and q is not None:
                if train:
                    p_list, loss, tokens = prompt.forward(q, i, tokens, train=True)
                    prompt_loss = prompt_loss + loss
                else:
                    p_list, _, tokens = prompt.forward(q, i, tokens, train=False)

            if p_list is not None:
                prompt_tokens = torch.cat(p_list, dim=1)
                L = prompt_tokens.size(1)
                tmp = torch.cat((tokens[:, :1, :], prompt_tokens, tokens[:, 1:, :]), dim=1)
                tmp = blk(tmp)
                tokens = torch.cat((tmp[:, :1, :], tmp[:, 1 + L :, :]), dim=1)
            else:
                tokens = blk(tokens)

        tokens = self.visual.norm(tokens)
        return tokens, prompt_loss


def create_uni3d_coda_prompt(
    *,
    pc_model: str = "eva02_base_patch14_448",
    pretrained_pc: Optional[str] = None,
    drop_path_rate: float = 0.0,
    trans_dim: int = 768,
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

    model = Uni3DCodaPromptBackbone(
        point_transformer=point_transformer,
        trans_dim=trans_dim,
        num_group=num_group,
        group_size=group_size,
        encoder_dim=encoder_dim,
        patch_dropout=patch_dropout,
    )

    if is_uni3d_pt:
        _load_uni3d_weights(model, pretrained_pc)

    return model



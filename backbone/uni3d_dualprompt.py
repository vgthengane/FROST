"""Uni3D with DualPrompt for 3D continual learning (prompt-tuning mode)."""

from __future__ import annotations

from typing import Optional, Sequence

import timm
import torch
import torch.nn as nn

from .prompt import EPrompt
from .uni3d_l2p import Group, Encoder, PatchDropout, _load_uni3d_weights


class PointcloudEncoderDualPrompt(nn.Module):
    """Point cloud encoder + transformer + DualPrompt (prompt-tuning mode).

    Returns a dict with keys ``logits``, ``pre_logits`` and optionally ``reduce_sim``.
    """

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
        num_classes: int = 1000,
        # DualPrompt (E-Prompt pool)
        prompt_length: int = 5,
        embedding_key: str = "cls",
        prompt_init: str = "uniform",
        prompt_pool: bool = True,
        prompt_key: bool = True,
        pool_size: int = 10,
        top_k: int = 1,
        batchwise_prompt: bool = True,
        prompt_key_init: str = "uniform",
        use_prompt_mask: bool = True,
        # DualPrompt (G-Prompt / E-Prompt layer placement)
        use_g_prompt: bool = True,
        g_prompt_length: int = 5,
        g_prompt_layer_idx: Optional[Sequence[int]] = (0, 1),
        use_prefix_tune_for_g_prompt: bool = False,
        use_e_prompt: bool = True,
        e_prompt_layer_idx: Optional[Sequence[int]] = (2, 3, 4),
        use_prefix_tune_for_e_prompt: bool = False,
        head_type: str = "token",
    ):
        super().__init__()

        if use_prefix_tune_for_g_prompt or use_prefix_tune_for_e_prompt:
            raise NotImplementedError(
                "Prompt-tuning (token concatenation) mode only. "
                "Please set use_prefix_tune_for_g_prompt=false and use_prefix_tune_for_e_prompt=false."
            )

        self.trans_dim = trans_dim
        self.embed_dim = embed_dim
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_dim = encoder_dim

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

        # Transformer backend (timm EVA02 / ViT-like)
        self.visual = point_transformer
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        # DualPrompt config
        self.prompt_pool = prompt_pool
        self.use_prompt_mask = use_prompt_mask
        self.head_type = head_type

        self.use_g_prompt = use_g_prompt
        self.g_prompt_length = g_prompt_length
        self.g_prompt_layer_idx = list(g_prompt_layer_idx) if g_prompt_layer_idx is not None else []

        self.use_e_prompt = use_e_prompt
        self.e_prompt_layer_idx = list(e_prompt_layer_idx) if e_prompt_layer_idx is not None else []

        # G-prompt tokens (prompt-tuning): [num_g_layers, g_len, C]
        if self.use_g_prompt and self.g_prompt_length is not None and len(self.g_prompt_layer_idx) > 0:
            g_shape = (len(self.g_prompt_layer_idx), int(self.g_prompt_length), trans_dim)
            if prompt_init == "zero":
                self.g_prompt = nn.Parameter(torch.zeros(g_shape))
            else:
                self.g_prompt = nn.Parameter(torch.randn(g_shape))
                nn.init.uniform_(self.g_prompt, -1, 1)
        else:
            self.g_prompt = None

        # E-prompt pool (selection) -> batched prompts per-layer
        if self.use_e_prompt:
            self.e_prompt = EPrompt(
                length=prompt_length,
                embed_dim=trans_dim,
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                prompt_key=prompt_key,
                pool_size=pool_size,
                top_k=top_k,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
                num_layers=len(self.e_prompt_layer_idx),
                use_prefix_tune_for_e_prompt=False,  # enforced above
                num_heads=-1,
                same_key_value=False,
            )
        else:
            self.e_prompt = None

        total = 0
        if self.use_g_prompt and self.g_prompt is not None:
            total += int(self.g_prompt_length) * len(self.g_prompt_layer_idx)
        if self.use_e_prompt and self.e_prompt is not None and self.prompt_pool:
            total += int(prompt_length) * int(top_k) * len(self.e_prompt_layer_idx)
        self.total_prompt_len = total

        self.trans2embed = nn.Linear(trans_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _build_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build initial token sequence x_tokens and group centers for position encoding.
        Returns:
          - x_tokens: [B, 1+G, trans_dim] (CLS + group tokens)
          - center: [B, G, 3]
        """
        pts = x[:, :, :3].contiguous()
        colors = x[:, :, 3:].contiguous()
        _, center, features = self.group_divider(pts, colors)

        group_tokens = self.encoder(features)          # [B, G, encoder_dim]
        group_tokens = self.encoder2trans(group_tokens)  # [B, G, trans_dim]

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        x_tokens = torch.cat((cls_tokens, group_tokens), dim=1)  # [B, 1+G, C]

        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        group_pos = self.pos_embed(center)
        pos = torch.cat((cls_pos, group_pos), dim=1)
        x_tokens = x_tokens + pos
        return x_tokens, center

    def extract_cls_transdim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return CLS feature in trans_dim (768) for prompt selection (original_backbone).
        """
        x_tokens, _ = self._build_tokens(x)
        x_tokens = self.patch_dropout(x_tokens)
        x_tokens = self.visual.pos_drop(x_tokens)
        for blk in self.visual.blocks:
            x_tokens = blk(x_tokens)
        x_cls = self.visual.norm(x_tokens[:, 0, :])
        x_cls = self.visual.fc_norm(x_cls)
        return x_cls

    def forward(self, x: torch.Tensor, task_id: int = -1, cls_features: Optional[torch.Tensor] = None, train: bool = False):
        # Initial tokens (CLS + groups) with geometry positional encoding
        x_tokens, _ = self._build_tokens(x)
        x_tokens = self.patch_dropout(x_tokens)
        x_tokens = self.visual.pos_drop(x_tokens)

        # Prompt mask (task-aware selection) for EPrompt
        prompt_mask = None
        if self.use_prompt_mask and train and self.e_prompt is not None:
            start = task_id * self.e_prompt.top_k
            end = (task_id + 1) * self.e_prompt.top_k
            if end <= self.e_prompt.pool_size:
                single_prompt_mask = torch.arange(start, end, device=x_tokens.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x_tokens.shape[0], -1)

        # Pre-compute E-prompt tokens for all E layers
        reduce_sim = None
        e_prompts = None
        if self.e_prompt is not None:
            res = self.e_prompt(x_tokens, prompt_mask=prompt_mask, cls_features=cls_features)
            e_prompts = res.get("batched_prompt", None)  # [num_e_layers, B, top_k*len, C]
            reduce_sim = res.get("reduce_sim", None)

        g_counter = -1
        e_counter = -1

        # Iterate transformer blocks and insert prompts at specified layers by concatenation
        for i, blk in enumerate(self.visual.blocks):
            if self.use_g_prompt and (i in self.g_prompt_layer_idx) and (self.g_prompt is not None):
                g_counter += 1
                g_tok = self.g_prompt[g_counter].unsqueeze(0).expand(x_tokens.size(0), -1, -1)
                x_tokens = torch.cat([g_tok, x_tokens], dim=1)
                x_tokens = blk(x_tokens)
                continue

            if self.use_e_prompt and (i in self.e_prompt_layer_idx) and (e_prompts is not None):
                e_counter += 1
                e_tok = e_prompts[e_counter]  # [B, top_k*len, C]
                x_tokens = torch.cat([e_tok, x_tokens], dim=1)
                x_tokens = blk(x_tokens)
                continue

            x_tokens = blk(x_tokens)

        # Final norm
        x_tokens = self.visual.norm(x_tokens)

        # Head selection (token) - CLS is shifted by total_prompt_len
        if self.head_type != "token":
            raise NotImplementedError("Only head_type='token' is supported.")

        cls_index = self.total_prompt_len  # CLS token position after all prompt insertions
        x_cls = x_tokens[:, cls_index, :]
        x_cls = self.visual.fc_norm(x_cls)

        pre_logits = self.trans2embed(x_cls)
        logits = self.head(pre_logits)

        out = {"pre_logits": pre_logits, "logits": logits}
        if reduce_sim is not None:
            out["reduce_sim"] = reduce_sim
        return out


def create_uni3d_dualprompt(
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
    num_classes: int = 1000,
    # DualPrompt args
    prompt_length: int = 5,
    embedding_key: str = "cls",
    prompt_init: str = "uniform",
    prompt_pool: bool = True,
    prompt_key: bool = True,
    pool_size: int = 10,
    top_k: int = 1,
    batchwise_prompt: bool = True,
    prompt_key_init: str = "uniform",
    head_type: str = "token",
    use_prompt_mask: bool = True,
    use_g_prompt: bool = True,
    g_prompt_length: int = 5,
    g_prompt_layer_idx: Optional[Sequence[int]] = (0, 1),
    use_prefix_tune_for_g_prompt: bool = False,
    use_e_prompt: bool = True,
    e_prompt_layer_idx: Optional[Sequence[int]] = (2, 3, 4),
    use_prefix_tune_for_e_prompt: bool = False,
    same_key_value: bool = False,
):
    # Uni3D uses a timm transformer backbone; if pretrained_pc is a Uni3D .pt, do not pass to timm.
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
                    f"Please provide a checkpoint_path via pretrained_pc parameter, "
                    f"or ensure timm version >= 0.9.7 is installed. Current timm: {timm.__version__}"
                ) from e
            point_transformer = timm.create_model(
                pc_model,
                checkpoint_path=checkpoint_path,
                drop_path_rate=drop_path_rate,
                pretrained=False,
            )
        else:
            raise

    model = PointcloudEncoderDualPrompt(
        point_transformer=point_transformer,
        trans_dim=trans_dim,
        embed_dim=embed_dim,
        num_group=num_group,
        group_size=group_size,
        encoder_dim=encoder_dim,
        patch_dropout=patch_dropout,
        num_classes=num_classes,
        prompt_length=prompt_length,
        embedding_key=embedding_key,
        prompt_init=prompt_init,
        prompt_pool=prompt_pool,
        prompt_key=prompt_key,
        pool_size=pool_size,
        top_k=top_k,
        batchwise_prompt=batchwise_prompt,
        prompt_key_init=prompt_key_init,
        head_type=head_type,
        use_prompt_mask=use_prompt_mask,
        use_g_prompt=use_g_prompt,
        g_prompt_length=g_prompt_length,
        g_prompt_layer_idx=g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=use_prefix_tune_for_g_prompt,
        use_e_prompt=use_e_prompt,
        e_prompt_layer_idx=e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
    )

    if is_uni3d_pt:
        _load_uni3d_weights(model, pretrained_pc)

    return model



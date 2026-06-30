"""
Uni3D with L2P (Learning to Prompt) for Continual Learning
Based on Uni3D architecture with integrated prompt mechanism
"""
import torch
import torch.nn as nn
import timm
import numpy as np
from typing import Optional
from functools import partial

from .prompt import Prompt

# Import pointnet2 utilities (need to be installed)
try:
    from pointnet2_ops import pointnet2_utils
except ImportError:
    print("Warning: pointnet2_ops not found. Please install it for Uni3D to work.")
    pointnet2_utils = None

# NOTE:
# We have observed an intermittent CUDA failure where calling pointnet2_ops FPS kernels before
# the first cuBLAS matmul can lead to `cublasCreate(handle)` / `CUBLAS_STATUS_*` errors later
# (the error surfaces at the first torch.matmul used in KNN distance computation).
# A robust workaround is to "warm up" cuBLAS once per CUDA device before invoking FPS.
_CUBLAS_WARMED_DEVICES = set()


def _warmup_cublas_if_needed(device: torch.device) -> None:
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _CUBLAS_WARMED_DEVICES:
        return
    # Small matmul to force cuBLAS handle initialization on this device.
    # IMPORTANT: do NOT consume RNG here (to avoid affecting experiment reproducibility).
    a = torch.ones((4, 4), device=device, dtype=torch.float32)
    b = torch.ones((4, 4), device=device, dtype=torch.float32)
    _ = a @ b
    torch.cuda.synchronize(device)
    _CUBLAS_WARMED_DEVICES.add(idx)


def _fps_torch(data, number):
    """Batched FPS without pointnet2_ops (slower; for dev / CPU-only setups)."""
    b, n, _ = data.shape
    centroids = torch.zeros(b, number, dtype=torch.long, device=data.device)
    distance = torch.ones(b, n, device=data.device) * 1e10
    farthest = torch.randint(0, n, (b,), device=data.device)
    batch_idx = torch.arange(b, device=data.device)
    for i in range(number):
        centroids[:, i] = farthest
        centroid = data[batch_idx, farthest, :].unsqueeze(1)
        dist = torch.sum((data - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return torch.gather(data, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))


def fps(data, number):
    """
    Farthest Point Sampling
    data: B N 3
    number: int
    """
    if pointnet2_utils is None:
        return _fps_torch(data, number)
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


def knn_point(nsample, xyz, new_xyz):
    """
    K-Nearest Neighbors
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    def square_distance(src, dst):
        """
        Calculate Euclid distance between each two points.
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class PatchDropout(nn.Module):
    """Patch Dropout for regularization"""
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch, device=x.device)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens, device=x.device)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    """Group points into local neighborhoods"""
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        """
        input: B N 3 (xyz), B N 3 (color)
        output: B G M 3, center: B G 3, features: B G M 6
        """
        batch_size, num_points, _ = xyz.shape
        # Warm up cuBLAS before using pointnet2_ops kernels (see note above).
        _warmup_cublas_if_needed(xyz.device)
        # FPS the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # KNN to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class Encoder(nn.Module):
    """Encoder for point cloud groups"""
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
        point_groups : B G M 6
        output: B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder — Conv1d/cuDNN 对非 contiguous 输入在部分环境下会异常，显式 contiguous
        feature = self.first_conv(point_groups.transpose(2, 1).contiguous())  # [BG, 6, n]
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG encoder_channel n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointcloudEncoderL2P(nn.Module):
    """Pointcloud Encoder with L2P prompt mechanism"""
    def __init__(
        self,
        point_transformer,
        trans_dim=768,
        embed_dim=1024,
        num_group=512,
        group_size=64,
        encoder_dim=256,
        patch_dropout=0.0,
        num_classes=1000,
        prompt_length=5,
        embedding_key='cls',
        prompt_init='uniform',
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init='uniform',
        use_prompt_mask: bool = False,
    ):
        super().__init__()
        self.trans_dim = trans_dim
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.num_group = num_group
        
        # Grouper
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        
        # Encoder
        self.encoder_dim = encoder_dim
        self.encoder = Encoder(encoder_channel=encoder_dim)
        
        # Bridge encoder and transformer
        self.encoder2trans = nn.Linear(encoder_dim, trans_dim)
        
        # Bridge transformer and final embedding
        self.trans2embed = nn.Linear(trans_dim, embed_dim)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )
        
        # Patch dropout
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        
        # Transformer (from timm)
        self.visual = point_transformer
        
        # L2P Prompt
        self.prompt_pool = prompt_pool
        self.use_prompt_mask = use_prompt_mask
        self.prompt_length = prompt_length
        self.top_k = top_k
        if prompt_length is not None and pool_size is not None and prompt_pool:
            self.prompt = Prompt(
                length=prompt_length,
                embed_dim=trans_dim,  # Use trans_dim (768) for prompt
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                prompt_key=prompt_key,
                pool_size=pool_size,
                top_k=top_k,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
            )
        else:
            self.prompt = None
            self.total_prompt_len = 0

        if self.prompt is not None and self.prompt_length is not None and self.top_k is not None:
            total_prompt_tokens = int(self.prompt_length) * int(self.top_k)
            self.prompt_pos_embed = nn.Parameter(torch.zeros(1, total_prompt_tokens, trans_dim))
            nn.init.trunc_normal_(self.prompt_pos_embed, std=0.02)
        else:
            self.prompt_pos_embed = None
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        """Forward pass.

        Args:
            x: ``[B, N, 6]`` point cloud (xyz + rgb concatenated)
            task_id: task id for continual learning (for ``prompt_mask``)
            cls_features: CLS features from original backbone (used for prompt selection)
            train: whether in training mode
        Returns:
            dict with 'logits', 'pre_logits', and optionally 'reduce_sim'
        """
        # Split xyz and rgb
        pts = x[:, :, :3].contiguous()  # [B, N, 3]
        colors = x[:, :, 3:].contiguous()  # [B, N, 3]
        # Divide the point cloud into groups
        _, center, features = self.group_divider(pts, colors)
        
        # Encode the input cloud patches
        group_input_tokens = self.encoder(features)  # B G encoder_dim
        group_input_tokens = self.encoder2trans(group_input_tokens)  # B G trans_dim
        
        prompt_out = {}
        if self.prompt is not None:
            prompt_mask = None
            if self.use_prompt_mask and train and task_id >= 0:
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                if end <= self.prompt.pool_size:
                    single_prompt_mask = torch.arange(start, end, device=group_input_tokens.device)
                    prompt_mask = single_prompt_mask.unsqueeze(0).expand(group_input_tokens.size(0), -1)

            prompt_out = self.prompt(group_input_tokens, prompt_mask=prompt_mask, cls_features=cls_features)
            group_input_tokens = prompt_out['prompted_embedding']
            self.total_prompt_len = prompt_out.get('total_prompt_len', 0)
        else:
            self.total_prompt_len = 0

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)

        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        group_pos = self.pos_embed(center)

        if self.total_prompt_len > 0:
            if self.prompt_pos_embed is not None:
                prompt_pos = self.prompt_pos_embed[:, :self.total_prompt_len, :].expand(
                    group_input_tokens.size(0), -1, -1
                )
            else:
                prompt_pos = torch.zeros(
                    group_input_tokens.size(0), self.total_prompt_len, self.trans_dim,
                    device=group_input_tokens.device, dtype=group_input_tokens.dtype
                )
            pos = torch.cat((cls_pos, prompt_pos, group_pos), dim=1)
        else:
            pos = torch.cat((cls_pos, group_pos), dim=1)

        x = x + pos
        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)

        for blk in self.visual.blocks:
            x = blk(x)

        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)
        x = self.trans2embed(x)
        logits = self.head(x)

        result = {'pre_logits': x, 'logits': logits}
        if 'reduce_sim' in prompt_out:
            result['reduce_sim'] = prompt_out['reduce_sim']
        return result
    
    def extract_cls_transdim(self, x):
        """Extract CLS features in trans_dim (used by ``original_backbone`` for prompt selection)."""
        pts = x[:, :, :3].contiguous()
        colors = x[:, :, 3:].contiguous()
        _, center, features = self.group_divider(pts, colors)

        group_input_tokens = self.encoder(features)
        group_input_tokens = self.encoder2trans(group_input_tokens)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = x + pos

        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)

        for blk in self.visual.blocks:
            x = blk(x)
        
        # Get CLS token output (trans_dim, not embed_dim)
        x = self.visual.norm(x[:, 0, :])  # B trans_dim
        x = self.visual.fc_norm(x)  # B trans_dim
        
        return x  # B trans_dim (768)


def _load_uni3d_weights(model, checkpoint_path):
    """
    Load Uni3D pretrained weights from checkpoint file.
    
    Args:
        model: PointcloudEncoderL2P model instance
        checkpoint_path: Path to .pt checkpoint file (DeepSpeed format)
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading Uni3D pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state_dict from checkpoint
    # DeepSpeed format has weights in 'module' key
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
    
    # Remove 'point_encoder.' prefix from keys (if present)
    # Our model structure doesn't have this prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'point_encoder.' prefix if present
        if key.startswith('point_encoder.'):
            new_key = key[len('point_encoder.'):]
        else:
            new_key = key
        
        # Skip keys that don't exist in our model (e.g., logit_scale, head weights if num_classes differs)
        # We'll use strict=False anyway, but this helps with logging
        new_state_dict[new_key] = value
    
    # Load weights with strict=False to allow for differences in classification head, etc.
    # Note: strict=False still errors on shape mismatch, so we catch and re-raise with a clearer message.
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        # Common case: encoder_dim mismatch between checkpoint and current config
        # Example:
        # - checkpoint has encoder.second_conv.3.weight: [512, 512, 1]
        # - model has encoder.second_conv.3.weight: [256, 512, 1]
        msg = str(e)
        if "size mismatch for encoder.second_conv.3.weight" in msg or "size mismatch for encoder2trans.weight" in msg:
            raise RuntimeError(
                "Uni3D checkpoint load failed due to shape mismatch. "
                "This almost always means your config 'encoder_dim' does not match the checkpoint. "
                "For /weights/uni3d_base.pt the checkpoint expects encoder_dim=512 (your current encoder_dim likely is 256). "
                "Please set args['encoder_dim']=512 (e.g., in exps/l2p.json) and retry."
            ) from e
        raise
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys not found in model (this is normal for classification head):")
        for key in list(missing_keys)[:10]:  # Print first 10
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint (ignored):")
        for key in list(unexpected_keys)[:10]:  # Print first 10
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")
    
    print("Uni3D pretrained weights loaded successfully")


def create_uni3d_original(
    pc_model='eva02_base_patch14_448',
    pretrained_pc=None,
    drop_path_rate=0.0,
    trans_dim=768,
    embed_dim=1024,
    num_group=512,
    group_size=64,
    encoder_dim=256,
    patch_dropout=0.0,
    num_classes=0,
):
    """
    Create Uni3D model WITHOUT prompt (for extracting cls_features)
    This is equivalent to original_backbone in ViT-L2P
    """
    # Create transformer blocks for point cloud via timm
    # Handle pretrained_pc: 
    # - If it's a .pt file (Uni3D format), don't pass to timm (we'll load manually later)
    # - Otherwise, pass to timm as checkpoint_path
    is_uni3d_pt = pretrained_pc and pretrained_pc.endswith('.pt')
    checkpoint_path = pretrained_pc if (pretrained_pc and not is_uni3d_pt) else ''
    
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
                    f"or ensure timm version >= 0.9.7 is installed. "
                    f"Current timm version: {timm.__version__}"
                ) from e
            else:
                point_transformer = timm.create_model(
                    pc_model,
                    checkpoint_path=checkpoint_path,
                    drop_path_rate=drop_path_rate,
                    pretrained=False,
                )
        else:
            raise
    
    # Create point cloud encoder WITHOUT prompt
    point_encoder = PointcloudEncoderL2P(
        point_transformer=point_transformer,
        trans_dim=trans_dim,
        embed_dim=embed_dim,
        num_group=num_group,
        group_size=group_size,
        encoder_dim=encoder_dim,
        patch_dropout=patch_dropout,
        prompt_length=None,  # No prompt
        embedding_key=None,
        prompt_init=None,
        prompt_pool=False,  # No prompt pool
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init=None,
        num_classes=num_classes,
    )
    
    # Load pretrained weights if provided (Uni3D .pt format)
    if is_uni3d_pt:
        _load_uni3d_weights(point_encoder, pretrained_pc)
    
    return point_encoder


def create_uni3d_l2p(
    pc_model='eva02_base_patch14_448',
    pretrained_pc=None,
    drop_path_rate=0.0,
    trans_dim=768,
    embed_dim=1024,
    num_group=512,
    group_size=64,
    encoder_dim=256,
    patch_dropout=0.0,
    num_classes=1000,
    prompt_length=5,
    embedding_key='cls',
    prompt_init='uniform',
    prompt_pool=False,
    prompt_key=False,
    pool_size=None,
    top_k=None,
    batchwise_prompt=False,
    prompt_key_init='uniform',
    use_prompt_mask: bool = False,
):
    """
    Create Uni3D model with L2P prompt mechanism
    
    Args:
        pc_model: Point cloud transformer model name (from timm)
        pretrained_pc: Path to pretrained checkpoint
        drop_path_rate: Drop path rate for transformer
        trans_dim: Transformer dimension (should match pc_model embed_dim)
        embed_dim: Final embedding dimension
        num_group: Number of groups for point cloud grouping
        group_size: Size of each group
        encoder_dim: Encoder output dimension
        patch_dropout: Patch dropout probability
        num_classes: Number of classes for classification head
        prompt_length: Length of each prompt
        embedding_key: Key for prompt selection ('mean', 'max', 'cls')
        prompt_init: Prompt initialization method
        prompt_pool: Whether to use prompt pool
        prompt_key: Whether to use learnable prompt keys
        pool_size: Size of prompt pool
        top_k: Top-k prompts to select
        batchwise_prompt: Whether to use batchwise prompt selection
        prompt_key_init: Prompt key initialization method
    """
    # Create transformer blocks for point cloud via timm
    # Handle pretrained_pc: 
    # - If it's a .pt file (Uni3D format), don't pass to timm (we'll load manually later)
    # - Otherwise, pass to timm as checkpoint_path
    is_uni3d_pt = pretrained_pc and pretrained_pc.endswith('.pt')
    checkpoint_path = pretrained_pc if (pretrained_pc and not is_uni3d_pt) else ''
    
    # Try to create model - if it fails, it might need a checkpoint_path
    try:
        point_transformer = timm.create_model(
            pc_model,
            checkpoint_path=checkpoint_path if checkpoint_path else None,
            drop_path_rate=drop_path_rate,
        )
    except RuntimeError as e:
        if "Unknown model" in str(e):
            # If model not found and no checkpoint provided, try with pretrained=False
            # This might work if the model architecture is registered but needs weights
            if not checkpoint_path:
                raise RuntimeError(
                    f"Model '{pc_model}' not found in timm. "
                    f"Please provide a checkpoint_path via pretrained_pc parameter, "
                    f"or ensure timm version >= 0.9.7 is installed. "
                    f"Current timm version: {timm.__version__}"
                ) from e
            else:
                # Try with checkpoint_path explicitly
                point_transformer = timm.create_model(
                    pc_model,
                    checkpoint_path=checkpoint_path,
                    drop_path_rate=drop_path_rate,
                    pretrained=False,  # Don't try to download pretrained weights
                )
        else:
            raise
    
    # Create point cloud encoder with L2P
    point_encoder = PointcloudEncoderL2P(
        point_transformer=point_transformer,
        trans_dim=trans_dim,
        embed_dim=embed_dim,
        num_group=num_group,
        group_size=group_size,
        encoder_dim=encoder_dim,
        patch_dropout=patch_dropout,
        prompt_length=prompt_length,
        embedding_key=embedding_key,
        prompt_init=prompt_init,
        prompt_pool=prompt_pool,
        prompt_key=prompt_key,
        pool_size=pool_size,
        top_k=top_k,
        batchwise_prompt=batchwise_prompt,
        prompt_key_init=prompt_key_init,
        use_prompt_mask=use_prompt_mask,
    )
    
    # Load pretrained weights if provided (Uni3D .pt format)
    if is_uni3d_pt:
        _load_uni3d_weights(point_encoder, pretrained_pc)
    
    return point_encoder


@timm.models.register_model
def uni3d_base_patch14_448_l2p(pretrained=False, **kwargs):
    """Uni3D Base with L2P"""
    model_kwargs = dict(
        pc_model='eva02_base_patch14_448',
        trans_dim=768,
        embed_dim=1024,
        num_group=512,
        group_size=64,
        encoder_dim=256,
        **kwargs
    )
    return create_uni3d_l2p(**model_kwargs)


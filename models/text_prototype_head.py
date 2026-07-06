"""Frozen CLIP text prototypes for Uni3D-aligned zero-shot classification.

Lightweight alternative to 3D-FLEG: template prompts + cosine similarity in the
pretrained 1024-d alignment space, without DGPC / spectral clustering / fusion MLPs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_PROMPT_TEMPLATE = "a point cloud model of a {}"
DEFAULT_CLIP_MODEL = "EVA02-E-14-plus"
DEFAULT_CLIP_PRETRAINED = "laion2b_s9b_b144k"


def modelnet40_shape_names(data_path: str | Path) -> list[str]:
    """Return ModelNet40 class names in canonical index order (0..39)."""
    root = Path(data_path)
    catfile = root / "modelnet40_shape_names.txt"
    if not catfile.is_file():
        raise FileNotFoundError(f"ModelNet40 class list not found: {catfile}")
    return [line.strip() for line in catfile.read_text().splitlines() if line.strip()]


def ordered_class_names(
    shape_names: Sequence[str],
    class_order: Sequence[int],
    num_classes: int,
) -> list[str]:
    """Map incremental class indices to human-readable names."""
    names: list[str] = []
    for idx in range(num_classes):
        if idx >= len(class_order):
            break
        orig = int(class_order[idx])
        if orig < 0 or orig >= len(shape_names):
            raise IndexError(
                f"class_order[{idx}]={orig} out of range for {len(shape_names)} names"
            )
        names.append(shape_names[orig])
    return names


class TextPrototypeHead(nn.Module):
    """CLIP text embeddings as class prototypes (frozen, no trainable fusion stack)."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        embed_dim: int = 1024,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        clip_model: str = DEFAULT_CLIP_MODEL,
        clip_pretrained: str = DEFAULT_CLIP_PRETRAINED,
        clip_checkpoint: str | None = None,
        logit_scale: float | None = None,
        prototype_path: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = int(embed_dim)
        self.prompt_template = str(prompt_template)
        self.clip_model = str(clip_model)
        self.clip_pretrained = str(clip_pretrained)
        self.clip_checkpoint = clip_checkpoint
        self._clip_model = None
        self._tokenizer = None

        init_scale = 100.0 if logit_scale is None else float(logit_scale)
        self.logit_scale = nn.Parameter(torch.tensor(init_scale).log())

        self.register_buffer(
            "prototypes",
            torch.zeros(0, self.embed_dim),
            persistent=True,
        )
        self._class_names: list[str] = []

        if prototype_path:
            self.load_prototypes(prototype_path)

    @property
    def num_classes(self) -> int:
        return int(self.prototypes.shape[0])

    def load_prototypes(self, path: str | Path) -> None:
        """Load precomputed [C, D] prototypes (see scripts/build_text_prototypes.py)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Text prototype file not found: {path}")
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            protos = payload.get("prototypes", payload.get("text_prototypes"))
            canonical = payload.get("canonical_prototypes")
            names = payload.get("class_names")
            scale = payload.get("logit_scale")
        else:
            protos, canonical, names, scale = payload, None, None, None
        if protos is None and canonical is None:
            raise ValueError(f"No prototypes tensor in {path}")
        if canonical is not None:
            canonical = F.normalize(canonical.float(), dim=-1)
            if canonical.ndim != 2 or canonical.shape[1] != self.embed_dim:
                raise ValueError(
                    f"Expected canonical [{self.embed_dim}], got {tuple(canonical.shape)}"
                )
            if "_canonical_prototypes" in self._buffers:
                self._canonical_prototypes = canonical.to(self.device)
            else:
                self.register_buffer(
                    "_canonical_prototypes", canonical.to(self.device), persistent=True
                )
        elif "_canonical_prototypes" in self._buffers:
            del self._buffers["_canonical_prototypes"]
        if protos is not None:
            protos = protos.float()
            if protos.ndim != 2 or protos.shape[1] != self.embed_dim:
                raise ValueError(
                    f"Expected prototypes [{self.embed_dim}], got {tuple(protos.shape)}"
                )
            protos = F.normalize(protos, dim=-1)
            self.prototypes = protos.to(self.device)
        elif canonical is not None:
            self.prototypes = torch.zeros(0, self.embed_dim, device=self.device)
        if names is not None:
            self._class_names = list(names)
        if scale is not None:
            self.logit_scale.data = torch.tensor(float(scale)).log().to(self.logit_scale)
        logging.info(
            "Loaded text prototypes from %s (%d classes, dim=%d)",
            path,
            self.num_classes,
            self.embed_dim,
        )

    def save_prototypes(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prototypes": self.prototypes.detach().cpu(),
                "canonical_prototypes": getattr(self, "_canonical_prototypes", None),
                "class_names": self._class_names,
                "logit_scale": float(self.logit_scale.exp().item()),
                "prompt_template": self.prompt_template,
                "clip_model": self.clip_model,
            },
            path,
        )

    def _lazy_clip(self):
        if self._clip_model is not None:
            return self._clip_model, self._tokenizer
        try:
            import open_clip
        except ImportError:
            import site
            from pathlib import Path as _Path

            user_site = _Path.home() / ".local/lib/python3.10/site-packages"
            if user_site.is_dir():
                site.addsitedir(str(user_site))
            try:
                import open_clip
            except ImportError as exc:
                raise ImportError(
                    "open_clip is required to encode text prototypes. "
                    "Install with `aprun python -m pip install --user open-clip-torch` "
                    "or pass `text_prototype_path` from scripts/build_text_prototypes.py."
                ) from exc

        kwargs = {}
        if self.clip_checkpoint:
            kwargs["pretrained"] = self.clip_checkpoint
        else:
            kwargs["pretrained"] = self.clip_pretrained
        cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model,
            **kwargs,
        )
        tokenizer = open_clip.get_tokenizer(self.clip_model)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(self.device)
        self._clip_model = model
        self._tokenizer = tokenizer
        if hasattr(model, "logit_scale"):
            self.logit_scale.data = model.logit_scale.detach().float().to(self.logit_scale)
        return model, tokenizer

    @torch.no_grad()
    def encode_prompts(self, class_names: Sequence[str]) -> torch.Tensor:
        """Encode class names with the frozen CLIP text tower."""
        prompts = [self.prompt_template.format(name.replace("_", " ")) for name in class_names]
        model, tokenizer = self._lazy_clip()
        tokens = tokenizer(prompts).to(self.device)
        emb = model.encode_text(tokens).float()
        return F.normalize(emb, dim=-1)

    def set_class_names(
        self,
        class_names: Sequence[str],
        class_order: Sequence[int] | None = None,
    ) -> None:
        """Build or refresh prototypes for incremental class indices 0..len-1."""
        names = list(class_names)
        if not names:
            self.prototypes = torch.zeros(0, self.embed_dim, device=self.device)
            self._class_names = []
            return

        if (
            class_order is not None
            and getattr(self, "_canonical_prototypes", None) is not None
            and self._canonical_prototypes is not None
        ):
            order = [int(i) for i in class_order[: len(names)]]
            canon = self._canonical_prototypes
            if max(order, default=-1) >= canon.shape[0]:
                raise IndexError("class_order references missing canonical prototypes")
            self.prototypes = canon[order].contiguous()
            self._class_names = names
            return

        if (
            self.num_classes == len(names)
            and names == self._class_names
            and self.num_classes > 0
        ):
            return

        protos = self.encode_prompts(names)
        self.prototypes = protos.to(self.device)
        self._class_names = names
        logging.info(
            "Text prototypes: %d classes, template=%r, sample=%r",
            len(names),
            self.prompt_template,
            prompts_sample(names[0], self.prompt_template),
        )

    def classify(self, embeddings: torch.Tensor, num_classes: int) -> torch.Tensor:
        if self.num_classes < num_classes:
            raise RuntimeError(
                f"Text prototypes cover {self.num_classes} classes, need {num_classes}. "
                "Call set_class_names() after each task."
            )
        protos = self.prototypes[:num_classes].to(embeddings.device, embeddings.dtype)
        pc = F.normalize(embeddings.float(), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * pc @ protos.T


def prompts_sample(class_name: str, template: str) -> str:
    return template.format(class_name.replace("_", " "))


def build_prototypes_for_dataset(
    *,
    dataset: str,
    data_path: str,
    class_order: Iterable[int],
    num_classes: int,
    output_path: str | Path,
    **head_kwargs,
) -> Path:
    """One-shot utility used by scripts/build_text_prototypes.py."""
    if dataset != "modelnet40":
        raise NotImplementedError(f"Text prototype builder only supports modelnet40, got {dataset}")
    shape_names = modelnet40_shape_names(data_path)
    names = ordered_class_names(shape_names, list(class_order), num_classes)
    head = TextPrototypeHead(device="cpu", **head_kwargs)
    head.set_class_names(names)
    output_path = Path(output_path)
    head.save_prototypes(output_path)
    return output_path
